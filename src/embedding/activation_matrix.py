"""Builds the `activation_matrix` (CLAUDE.md Step 3): run the trained model
over the embed split and, for every conv filter, collapse its feature map
into the configured statistics.

Memory note: raw activations are never accumulated for the whole embed set —
each batch's feature maps are reduced to (batch, channels) scalars immediately
via `compute_stats`, and only those scalars are kept, so peak memory is one
batch of activations, not `num_samples` worth.
"""

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .stats import STAT_NAMES, compute_stats


@dataclass
class ActivationConfig:
    stats: Sequence[str] = field(default_factory=lambda: list(STAT_NAMES))
    batch_size: int = 128
    entropy_bins: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2


@dataclass
class ActivationMatrix:
    representation: np.ndarray  # (num_filters, num_samples, num_stats)
    stats: List[str]
    layer_names: List[str]  # one entry per conv layer, in filter order
    filters_per_layer: List[int]  # out_channels per layer, same order as layer_names
    labels: np.ndarray  # (num_samples,) embed-set labels, aligned to the sample axis


def _conv_layers(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    return [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]


def build_activation_matrix(
    model: nn.Module,
    embed_dataset: Dataset,
    config: ActivationConfig = None,
) -> ActivationMatrix:
    config = config or ActivationConfig()
    device = torch.device(config.device)
    model.to(device)
    model.eval()

    conv_layers = _conv_layers(model)
    if not conv_layers:
        raise ValueError("model has no Conv2d layers to extract activations from")

    captured = {}

    def make_hook(name: str):
        def hook(module, inputs, output):
            captured[name] = output.detach()

        return hook

    handles = [conv.register_forward_hook(make_hook(name)) for name, conv in conv_layers]

    chunks = {name: {s: [] for s in config.stats} for name, _ in conv_layers}
    all_labels = []

    loader = DataLoader(
        embed_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    try:
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                captured.clear()
                model(x)
                all_labels.append(y.numpy())
                for name, _ in conv_layers:
                    stat_values = compute_stats(captured[name], config.stats, entropy_bins=config.entropy_bins)
                    for s in config.stats:
                        chunks[name][s].append(stat_values[s].cpu().numpy())
    finally:
        for h in handles:
            h.remove()

    labels = np.concatenate(all_labels)
    layer_names = [name for name, _ in conv_layers]
    filters_per_layer = [conv.out_channels for _, conv in conv_layers]

    per_layer_arrays = []
    for name in layer_names:
        per_stat = [np.concatenate(chunks[name][s], axis=0) for s in config.stats]  # each (num_samples, C_l)
        stacked = np.stack(per_stat, axis=-1)  # (num_samples, C_l, num_stats)
        per_layer_arrays.append(np.transpose(stacked, (1, 0, 2)))  # (C_l, num_samples, num_stats)

    representation = np.concatenate(per_layer_arrays, axis=0)  # (num_filters, num_samples, num_stats)

    return ActivationMatrix(
        representation=representation,
        stats=list(config.stats),
        layer_names=layer_names,
        filters_per_layer=filters_per_layer,
        labels=labels,
    )
