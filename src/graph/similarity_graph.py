"""Filter-similarity graph construction (CLAUDE.md Step 4).

Node = one filter. Its feature vector is that filter's full
`(num_samples, num_stats)` slice from `activation_matrix.representation`,
flattened. An edge is added between two filters only if their cosine
similarity clears a threshold; edge weight is that similarity, min-max
renormalized to [0, 1] across all kept edges afterward.

Each node carries `layer`/`channel` attributes so a node can be traced back
to an actual (Conv2d layer, output channel) pair later, when pruning acts on
the graph's clustering output.

Cross-layer vs. same-layer comparison is a config choice, not a fixed design
decision (`GraphConfig.same_layer_only` / `cross_layer_threshold`):

- Every filter is reduced to the same-length statistic vector regardless of
  its source layer's spatial size, so cosine similarity is always numerically
  well-defined across layers — but that doesn't mean it's semantically
  meaningful. Shallow and deep filters differ in receptive field, input
  distribution, and (with ReLU) sparsity, so a high similarity score between
  them can be coincidental rather than true redundancy.
- `same_layer_only=True` restricts edges to filter pairs from the same conv
  layer — the safer, literature-aligned default, since most filter-pruning
  methods compare filters only within a layer.
- `same_layer_only=False` with `cross_layer_threshold=None` is the fully
  global graph (any pair may connect at `similarity_threshold`) — can surface
  genuine cross-depth redundancy (relevant for e.g. DenseNet's explicit
  feature reuse) but risks spurious cross-layer edges.
- `same_layer_only=False` with `cross_layer_threshold` set is a hybrid: normal
  `similarity_threshold` within a layer, but a stricter (typically higher)
  `cross_layer_threshold` required for a cross-layer edge — cross-layer
  redundancy is admitted only on stronger evidence.

Which of these performs best for a given architecture/dataset is an open,
empirical question — see "Known iteration points" in CLAUDE.md.
"""

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
import torch

from src.embedding.activation_matrix import ActivationMatrix


@dataclass
class GraphConfig:
    similarity_threshold: float = 0.7
    same_layer_only: bool = False
    # Only used when same_layer_only=False. If set, cross-layer edges require
    # this (typically stricter) threshold instead of similarity_threshold.
    cross_layer_threshold: Optional[float] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _flatten_filters(representation: np.ndarray) -> np.ndarray:
    """(num_filters, num_samples, num_stats) -> (num_filters, num_samples * num_stats)."""
    return representation.reshape(representation.shape[0], -1)


def _cosine_similarity_matrix(vectors: np.ndarray, device: str) -> np.ndarray:
    x = torch.as_tensor(vectors, dtype=torch.float32, device=device)
    x = torch.nn.functional.normalize(x, dim=1, eps=1e-12)
    return (x @ x.T).cpu().numpy()


def _node_layer_channel(activation_matrix: ActivationMatrix):
    layers, channels = [], []
    for layer_name, n_channels in zip(activation_matrix.layer_names, activation_matrix.filters_per_layer):
        layers.extend([layer_name] * n_channels)
        channels.extend(range(n_channels))
    return layers, channels


def _normalize_edge_weights(graph: nx.Graph) -> None:
    weights = np.array([d["weight"] for _, _, d in graph.edges(data=True)])
    if len(weights) == 0:
        return
    lo, hi = weights.min(), weights.max()
    span = hi - lo
    for _, _, d in graph.edges(data=True):
        d["weight"] = 1.0 if span == 0 else (d["weight"] - lo) / span


def build_similarity_graph(activation_matrix: ActivationMatrix, config: GraphConfig = None) -> nx.Graph:
    config = config or GraphConfig()
    vectors = _flatten_filters(activation_matrix.representation)
    sim = _cosine_similarity_matrix(vectors, config.device)
    num_filters = vectors.shape[0]
    layers, channels = _node_layer_channel(activation_matrix)

    graph = nx.Graph()
    for i in range(num_filters):
        graph.add_node(i, layer=layers[i], channel=channels[i])

    iu = np.triu_indices(num_filters, k=1)
    sims = sim[iu]

    layer_ids = np.unique(layers, return_inverse=True)[1]
    same_layer = layer_ids[iu[0]] == layer_ids[iu[1]]

    if config.same_layer_only:
        mask = same_layer & (sims >= config.similarity_threshold)
    elif config.cross_layer_threshold is not None:
        mask = (same_layer & (sims >= config.similarity_threshold)) | (
            ~same_layer & (sims >= config.cross_layer_threshold)
        )
    else:
        mask = sims >= config.similarity_threshold

    for i, j, w in zip(iu[0][mask], iu[1][mask], sims[mask]):
        graph.add_edge(int(i), int(j), weight=float(w))

    _normalize_edge_weights(graph)
    return graph
