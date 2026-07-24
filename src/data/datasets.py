"""Dataset loading and class-stratified train/test/embed splitting.

Pipeline step 1 (see CLAUDE.md): load a torchvision dataset, pool ALL of its
native partitions (train/test, or train/val/test, or a single unsplit
partition — whatever the dataset natively offers), then re-split that full
pool into train/test/embed at configurable fractions (default 80/10/10) with
equal per-class representation in each split.

Every torchvision classification dataset exposes its per-sample labels under a
different attribute name (`targets`, `labels`, `_labels`, `y`, or — for GTSRB —
only inside a list of (path, label) tuples). `DATASET_REGISTRY` records, per
dataset, which native partitions to pool and where to find its labels, so
adding a new dataset is a one-line registry entry rather than a new branch of
loading logic.

Not included: CelebA. Its labels are 40 binary attributes per image, not a
single class label, so the stratified-by-class splitting here doesn't apply
to it as-is.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class DatasetSpec:
    ctor: Callable[..., Dataset]
    # One kwargs dict per native partition to load and pool (e.g. train/test,
    # or train/val/test, or a single `{}` for datasets with no native split).
    partitions: Tuple[dict, ...]
    # Attribute holding integer labels: "targets" | "labels" | "_labels" | "y",
    # or the sentinel "samples" for datasets that only expose (path, label) pairs.
    label_attr: str = "targets"
    # Extra kwargs merged into every partition's constructor call (e.g. EMNIST's
    # required `split=` for which EMNIST subset, distinct from train/test).
    extra_kwargs: dict = field(default_factory=dict)
    # False for datasets torchvision cannot auto-download (ImageNet).
    auto_download: bool = True


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(datasets.CIFAR10, ({"train": True}, {"train": False})),
    "cifar100": DatasetSpec(datasets.CIFAR100, ({"train": True}, {"train": False})),
    "svhn": DatasetSpec(datasets.SVHN, ({"split": "train"}, {"split": "test"}), label_attr="labels"),
    "stl10": DatasetSpec(datasets.STL10, ({"split": "train"}, {"split": "test"}), label_attr="labels"),
    "fashionmnist": DatasetSpec(datasets.FashionMNIST, ({"train": True}, {"train": False})),
    "mnist": DatasetSpec(datasets.MNIST, ({"train": True}, {"train": False})),
    "kmnist": DatasetSpec(datasets.KMNIST, ({"train": True}, {"train": False})),
    "emnist": DatasetSpec(
        datasets.EMNIST,
        ({"train": True}, {"train": False}),
        extra_kwargs={"split": "balanced"},
    ),
    "qmnist": DatasetSpec(datasets.QMNIST, ({"what": "train"}, {"what": "test"})),
    "usps": DatasetSpec(datasets.USPS, ({"train": True}, {"train": False})),
    "gtsrb": DatasetSpec(
        datasets.GTSRB, ({"split": "train"}, {"split": "test"}), label_attr="samples"
    ),
    "caltech101": DatasetSpec(
        datasets.Caltech101, ({},), label_attr="y", extra_kwargs={"target_type": "category"}
    ),
    "caltech256": DatasetSpec(datasets.Caltech256, ({},), label_attr="y"),
    "food101": DatasetSpec(
        datasets.Food101, ({"split": "train"}, {"split": "test"}), label_attr="_labels"
    ),
    "oxfordiiitpet": DatasetSpec(
        datasets.OxfordIIITPet,
        ({"split": "trainval"}, {"split": "test"}),
        label_attr="_labels",
    ),
    "flowers102": DatasetSpec(
        datasets.Flowers102,
        ({"split": "train"}, {"split": "val"}, {"split": "test"}),
        label_attr="_labels",
    ),
    "sun397": DatasetSpec(datasets.SUN397, ({},), label_attr="_labels"),
    "dtd": DatasetSpec(
        datasets.DTD,
        ({"split": "train"}, {"split": "val"}, {"split": "test"}),
        label_attr="_labels",
    ),
    "fgvcaircraft": DatasetSpec(
        datasets.FGVCAircraft,
        ({"split": "train"}, {"split": "val"}, {"split": "test"}),
        label_attr="_labels",
    ),
    "imagenet": DatasetSpec(
        datasets.ImageNet,
        ({"split": "train"}, {"split": "val"}),
        auto_download=False,
    ),
}


@dataclass
class SplitConfig:
    train_frac: float = 0.8
    test_frac: float = 0.1
    embed_frac: float = 0.1
    # None = np.random.default_rng(None), i.e. seed from OS entropy -- a fresh,
    # non-reproducible train/test/embed partition each call.
    seed: Optional[int] = 42

    def __post_init__(self) -> None:
        total = self.train_frac + self.test_frac + self.embed_frac
        if not np.isclose(total, 1.0):
            raise ValueError(f"train/test/embed fractions must sum to 1.0, got {total}")


def _extract_labels(dataset: Dataset, label_attr: str) -> np.ndarray:
    if label_attr == "samples":
        samples = dataset._samples  # GTSRB: list of (path, label) tuples
        return np.asarray([label for _, label in samples])
    if not hasattr(dataset, label_attr):
        raise AttributeError(f"{type(dataset).__name__} has no attribute '{label_attr}'")
    return np.asarray(getattr(dataset, label_attr))


def get_labels(dataset: Dataset) -> np.ndarray:
    """Best-effort label extraction for a raw (non-registry) torchvision dataset."""
    for attr in ("targets", "labels", "_labels", "y"):
        if hasattr(dataset, attr):
            return np.asarray(getattr(dataset, attr))
    if hasattr(dataset, "_samples"):
        return _extract_labels(dataset, "samples")
    raise AttributeError(
        f"Dataset {type(dataset).__name__} exposes no recognized label attribute"
    )


def stratified_split_indices(
    labels: np.ndarray, config: SplitConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split sample indices into (train, test, embed), preserving per-class proportions.

    Any samples left over after rounding train/test counts are assigned to embed,
    so every sample is used exactly once and class balance is kept within each split.
    """
    rng = np.random.default_rng(config.seed)
    train_idx, test_idx, embed_idx = [], [], []

    for class_id in np.unique(labels):
        class_idx = np.where(labels == class_id)[0]
        rng.shuffle(class_idx)

        n = len(class_idx)
        n_train = min(int(round(n * config.train_frac)), n)
        n_test = min(int(round(n * config.test_frac)), n - n_train)

        train_idx.append(class_idx[:n_train])
        test_idx.append(class_idx[n_train : n_train + n_test])
        embed_idx.append(class_idx[n_train + n_test :])

    return (
        np.concatenate(train_idx),
        np.concatenate(test_idx),
        np.concatenate(embed_idx),
    )


def load_full_dataset(
    name: str,
    root: str = "./data",
    download: bool = True,
    transform: Optional[Callable] = None,
) -> Tuple[Dataset, np.ndarray]:
    """Load `name`, pool all of its native partitions, and return (pooled_dataset, labels).

    `transform` defaults to `ToTensor()`; pass your own for model-specific
    preprocessing (e.g. resizing variable-sized images like Caltech/Food101/SUN397).
    """
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset '{name}'. Supported: {sorted(DATASET_REGISTRY)}"
        )
    spec = DATASET_REGISTRY[key]
    transform = transform or transforms.Compose([transforms.ToTensor()])

    partitions, label_arrays = [], []
    for part_kwargs in spec.partitions:
        kwargs = {"root": root, "transform": transform, **spec.extra_kwargs, **part_kwargs}
        if spec.auto_download:
            kwargs["download"] = download
        ds = spec.ctor(**kwargs)
        partitions.append(ds)
        label_arrays.append(_extract_labels(ds, spec.label_attr))

    full = partitions[0] if len(partitions) == 1 else ConcatDataset(partitions)
    labels = np.concatenate(label_arrays)
    return full, labels


@dataclass
class DatasetMetadata:
    num_classes: int
    in_channels: int


def infer_metadata(full: Dataset, labels: np.ndarray) -> DatasetMetadata:
    """Derive (num_classes, in_channels) from already-loaded data — torchvision
    datasets encode labels as contiguous ints 0..num_classes-1, so no per-dataset
    hardcoded table is needed."""
    sample_image, _ = full[0]
    return DatasetMetadata(num_classes=len(np.unique(labels)), in_channels=sample_image.shape[0])


def build_splits(
    name: str,
    root: str = "./data",
    config: Optional[SplitConfig] = None,
    download: bool = True,
    transform: Optional[Callable] = None,
) -> Tuple[Subset, Subset, Subset, DatasetMetadata]:
    """Load `name` and return (train, test, embed, metadata), class-stratified per SplitConfig."""
    config = config or SplitConfig()
    full, labels = load_full_dataset(name, root=root, download=download, transform=transform)
    train_idx, test_idx, embed_idx = stratified_split_indices(labels, config)
    metadata = infer_metadata(full, labels)

    return (
        Subset(full, train_idx),
        Subset(full, test_idx),
        Subset(full, embed_idx),
        metadata,
    )
