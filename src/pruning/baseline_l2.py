"""L2-norm magnitude pruning baseline (CLAUDE.md Step 9's comparator), via
torch-pruning's MetaPruner directly -- deliberately the *same* mechanical
removal engine as our own method (`prune.py`), so the comparison isolates the
one thing that differs: which filters get selected, not how removal happens.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch_pruning as tp


@dataclass
class L2PruneConfig:
    pruning_ratio: float = 0.3
    global_pruning: bool = True  # network-wide selection, not a fixed per-layer quota --
    # matches our method's own network-wide (cluster-driven) selection for a fair comparison


def apply_l2_baseline_prune(
    model: nn.Module, example_inputs: torch.Tensor, config: L2PruneConfig = None
) -> nn.Module:
    config = config or L2PruneConfig()
    importance = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=importance,
        pruning_ratio=config.pruning_ratio,
        global_pruning=config.global_pruning,
        ignored_layers=ignored_layers,
    )
    pruner.step()
    return model
