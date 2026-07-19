"""CNN backbone builder: basic CNN, ResNet, VGG, DenseNet — swappable via a registry.

All architectures except `simplecnn` are torchvision models built with
`weights=None` (trained from scratch, per the pipeline in CLAUDE.md — there is
no pretrained-weight reuse here).

Two adaptations are applied on top of the raw torchvision model so the same
architecture works across every dataset in `src/data/datasets.py`:

- `in_channels`: the first conv layer is rebuilt if a dataset is grayscale
  (MNIST family) instead of RGB.
- `small_inputs`: ResNet/DenseNet were designed for 224x224 ImageNet inputs —
  their stem (7x7/stride-2 conv + maxpool) discards too much spatial
  information on 28-96px images (MNIST family, CIFAR, SVHN, STL10). When
  `small_inputs=True`, the stem conv shrinks to 3x3/stride-1 and the stem
  maxpool is dropped — the standard "CIFAR-style" adaptation used throughout
  CNN-pruning literature. VGG has no single stem to swap (its downsampling is
  spread across 5 maxpools); instead its classifier is replaced with a single
  compact Linear layer sized to the actual (much smaller) flattened feature
  count, since torchvision's default 3-layer/4096-unit classifier is sized for
  224x224 inputs and does not fit a CIFAR-sized feature map.
"""

from dataclasses import dataclass
from typing import Callable, Dict

import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """A plain 4-conv-block CNN — the "most basic" tier of the model registry."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_width: int = 32):
        super().__init__()
        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, widths[0]),
            block(widths[0], widths[1]),
            block(widths[1], widths[2]),
            block(widths[2], widths[3]),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _replace_conv_in_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv
    return nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )


def _shrink_stem_conv(conv: nn.Conv2d) -> nn.Conv2d:
    """7x7/stride-2 ImageNet-style stem conv -> 3x3/stride-1 for small inputs."""
    return nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=conv.bias is not None,
    )


def _replace_first_conv(seq: nn.Sequential, in_channels: int) -> None:
    for i, m in enumerate(seq):
        if isinstance(m, nn.Conv2d):
            seq[i] = _replace_conv_in_channels(m, in_channels)
            return
    raise ValueError("no Conv2d layer found in sequential")


def _last_conv_out_channels(seq: nn.Sequential) -> int:
    convs = [m for m in seq if isinstance(m, nn.Conv2d)]
    return convs[-1].out_channels


def _drop_last_maxpool(seq: nn.Sequential) -> None:
    for i in range(len(seq) - 1, -1, -1):
        if isinstance(seq[i], nn.MaxPool2d):
            seq[i] = nn.Identity()
            return
    raise ValueError("no MaxPool2d layer found in sequential")


def _adapt_resnet(model: nn.Module, in_channels: int, num_classes: int, small_inputs: bool) -> nn.Module:
    if small_inputs:
        model.conv1 = _shrink_stem_conv(model.conv1)
        model.maxpool = nn.Identity()
    if in_channels != 3:
        model.conv1 = _replace_conv_in_channels(model.conv1, in_channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _adapt_densenet(model: nn.Module, in_channels: int, num_classes: int, small_inputs: bool) -> nn.Module:
    stem = model.features
    if small_inputs:
        stem.conv0 = _shrink_stem_conv(stem.conv0)
        stem.pool0 = nn.Identity()
    if in_channels != 3:
        stem.conv0 = _replace_conv_in_channels(stem.conv0, in_channels)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def _adapt_vgg(model: nn.Module, in_channels: int, num_classes: int, small_inputs: bool) -> nn.Module:
    if in_channels != 3:
        _replace_first_conv(model.features, in_channels)
    if small_inputs:
        # VGG's 5 stride-2 maxpools collapse inputs as small as 28x28 (MNIST family)
        # below 1x1 before the last stage. AdaptiveAvgPool2d(1) below already handles
        # any leftover spatial size, so dropping the last maxpool costs nothing and
        # avoids that crash.
        _drop_last_maxpool(model.features)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Linear(_last_conv_out_channels(model.features), num_classes)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


@dataclass
class ModelSpec:
    ctor: Callable[..., nn.Module]
    family: str  # "simple" | "resnet" | "vgg" | "densenet"


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "simplecnn": ModelSpec(SimpleCNN, family="simple"),
    "resnet18": ModelSpec(models.resnet18, family="resnet"),
    "resnet34": ModelSpec(models.resnet34, family="resnet"),
    "resnet50": ModelSpec(models.resnet50, family="resnet"),
    "vgg11_bn": ModelSpec(models.vgg11_bn, family="vgg"),
    "vgg13_bn": ModelSpec(models.vgg13_bn, family="vgg"),
    "vgg16_bn": ModelSpec(models.vgg16_bn, family="vgg"),
    "vgg19_bn": ModelSpec(models.vgg19_bn, family="vgg"),
    "densenet121": ModelSpec(models.densenet121, family="densenet"),
    "densenet169": ModelSpec(models.densenet169, family="densenet"),
    "densenet201": ModelSpec(models.densenet201, family="densenet"),
}

_ADAPTERS = {
    "resnet": _adapt_resnet,
    "vgg": _adapt_vgg,
    "densenet": _adapt_densenet,
}


def build_model(
    name: str,
    num_classes: int,
    in_channels: int = 3,
    small_inputs: bool = True,
) -> nn.Module:
    """Build a CNN by registry name, adapted for `in_channels`/`num_classes`/`small_inputs`."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{name}'. Supported: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[key]

    if spec.family == "simple":
        return spec.ctor(in_channels=in_channels, num_classes=num_classes)

    model = spec.ctor(weights=None)
    return _ADAPTERS[spec.family](model, in_channels, num_classes, small_inputs)
