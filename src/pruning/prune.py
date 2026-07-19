"""Apply our clustering-derived prune plan to a model via torch-pruning
(CLAUDE.md Step 8's "reinstantiate" half — retraining is `src/train.py`'s job).

The graph's node `layer`/`channel` attributes (`src/graph/similarity_graph.py`)
map each selected global filter index back to an (original conv layer name,
original channel index) pair, computed when activations were extracted from
the *unpruned* model.

Coupled architectures (ResNet's residual adds, etc.) make this non-trivial:
removing an output channel from one conv layer can force torch-pruning to
also remove the corresponding channel from every other layer tied to it by a
shape constraint (verified empirically — pruning one channel from a
`resnet18` residual block's conv2 cascades through the block's downsample
BatchNorm and into the *next* block's conv2 too, since they share the same
residual-sum channel dimension). Two consequences that this module has to
handle correctly:

1. A single requested filter can silently remove channels from OTHER layers
   as a structural side effect, including layers we independently intended to
   prune specific channels from ourselves.
2. Channel indices shift after every removal, so a later request for "prune
   these original channel numbers from layer X" is wrong once X has already
   been touched by an earlier group's coupling side effect.

We handle this by tracking, per conv layer, a live mapping from current
position -> original channel id, seeded from every conv layer up front and
updated after each executed group using exactly the (layer, current-index)
pairs torch-pruning reports as touched — not just the layer we explicitly
targeted. A request is translated through this live map each time, so
already-removed channels are automatically skipped rather than double-removed
or mis-indexed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import networkx as nx
import torch
import torch.nn as nn
import torch_pruning as tp


@dataclass
class PruneReport:
    skipped_layers: List[str] = field(default_factory=list)  # requested prune was invalid, left untouched


def _out_channel_conv_updates(group):
    """Yields (conv_module, current_idxs) for every Conv2d whose own out_channels
    this group actually reduces (as opposed to e.g. a downstream conv's in_channels,
    or a BatchNorm mirroring its preceding conv — neither represents "a filter removed
    from this layer" in the sense our index tracking cares about)."""
    for dep, idxs in group:
        module = dep.target.module
        if isinstance(module, nn.Conv2d) and dep.handler.__func__.__qualname__ == "ConvPruner.prune_out_channels":
            yield module, idxs


def apply_prune_plan(
    model: nn.Module,
    graph: nx.Graph,
    filter_indices_to_prune: Sequence[int],
    example_inputs: torch.Tensor,
) -> PruneReport:
    """Mutates `model` in place, removing the requested filters (plus whatever
    else coupling structurally requires). Returns which requested layers had to
    be skipped entirely (e.g. would have reduced a layer to 0 channels)."""
    example_inputs = example_inputs.to(next(model.parameters()).device)
    conv_layers: Dict[str, nn.Conv2d] = {
        name: m for name, m in model.named_modules() if isinstance(m, nn.Conv2d)
    }
    module_to_name = {m: n for n, m in conv_layers.items()}

    # current position -> original channel id, per conv layer; seeded for every
    # conv layer (not just requested ones) since coupling can touch any of them.
    index_maps: Dict[str, List[int]] = {name: list(range(m.out_channels)) for name, m in conv_layers.items()}

    plan_by_layer: Dict[str, set] = {}
    for gidx in filter_indices_to_prune:
        layer_name = graph.nodes[gidx]["layer"]
        channel_idx = graph.nodes[gidx]["channel"]
        plan_by_layer.setdefault(layer_name, set()).add(channel_idx)

    dg = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
    report = PruneReport()

    for layer_name, original_targets in plan_by_layer.items():
        conv = conv_layers[layer_name]
        current_map = index_maps[layer_name]
        current_idxs = [pos for pos, orig in enumerate(current_map) if orig in original_targets]
        if not current_idxs:
            continue  # already fully removed as a side effect of an earlier group

        group = dg.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=current_idxs)
        if not dg.check_pruning_group(group):
            report.skipped_layers.append(layer_name)
            continue

        group.prune()

        for touched_conv, touched_idxs in _out_channel_conv_updates(group):
            touched_name = module_to_name[touched_conv]
            keep = set(range(len(index_maps[touched_name]))) - set(touched_idxs)
            index_maps[touched_name] = [index_maps[touched_name][pos] for pos in sorted(keep)]

    return report
