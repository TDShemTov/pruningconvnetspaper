"""Graph embedding via node2vec (CLAUDE.md Step 5) — isolated in its own file
so it's a one-file swap if a different graph-embedding method replaces it later.

Custom pure-PyTorch implementation (biased random walk + skip-gram with
negative sampling), not the gensim-based `node2vec` package. Two reasons:

1. Weighted sampling, portable across CPU/GPU. `torch_geometric.nn.Node2Vec`
   (the standard GPU option) turned out to ignore edge weights entirely --
   confirmed by reading its `pos_sample` source, which calls its random-walk
   kernel with no weight tensor at all. That's a real behavioral regression
   for us: `similarity_graph.py`'s whole point is that edge weight (cosine
   similarity strength) should bias which neighbor a walk favors, not just
   gate whether an edge exists. This implementation samples each step via
   `torch.multinomial` over the (possibly GPU-resident) weight tensor, so it
   stays correct AND is a device flag (`config.device`) away from running on
   GPU -- no pyg-lib/torch_cluster version-matched dependency needed either
   (confirmed: no prebuilt wheel existed yet for our torch/CUDA combo).
2. Raw speed, even on CPU. The gensim-based version failed to finish
   embedding a real (untrained) ResNet18's activation graph (4800 filters,
   6.6M edges at similarity_threshold=0.7) after 10+ minutes and 10GB+ RAM --
   its Python-level walk generation doesn't scale to that edge count. This
   implementation completes the same graph in ~134s on CPU, ~66s on GPU,
   because walk generation and skip-gram training are both fully vectorized
   tensor operations (`_random_walks`, `_SkipGram.loss`), not per-node/per-step
   Python loops.

Building the padded per-node neighbor/weight arrays is also fully vectorized
(`_build_padded_adjacency`, via sort + cumulative offsets) rather than a
per-node Python loop, for the same reason -- it's not just node2vec's walk
generation that has to scale to millions of edges, so does turning the graph
into arrays in the first place.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Node2VecConfig:
    embed_dim: int = 64
    walk_length: int = 30
    num_walks: int = 200
    window: int = 10
    p: float = 1.0
    q: float = 1.0
    epochs: int = 5
    batch_size: int = 2048
    lr: float = 0.025
    num_negative_samples: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    # Floor applied to edge weights only for random-walk sampling. similarity_graph's
    # min-max normalization can legitimately assign an edge exactly weight 0.0 (the
    # least similar kept edge) -- if that's a low-degree node's only edge, its total
    # outgoing weight is 0 and torch.multinomial has nothing valid to sample.
    min_walk_weight: float = 1e-6


def _build_padded_adjacency(graph: nx.Graph, min_weight: float, device: torch.device):
    """Per-node neighbor lists/weights, padded to the graph's max degree, plus a
    dense boolean adjacency matrix (for the p/q "is this a common neighbor of the
    previous node" check) -- all built via sort + cumulative offsets, not a loop."""
    num_nodes = graph.number_of_nodes()
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(num_nodes), weight="weight", format="coo")
    sources, targets = adjacency.row, adjacency.col
    weights = np.maximum(adjacency.data, min_weight)

    degrees = np.bincount(sources, minlength=num_nodes)
    max_degree = max(int(degrees.max()) if len(degrees) else 0, 1)

    order = np.argsort(sources, kind="stable")
    sorted_sources = sources[order]
    sorted_targets = targets[order]
    sorted_weights = weights[order]

    offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(degrees, out=offsets[1:])
    row_position = np.arange(len(sorted_sources)) - offsets[sorted_sources]

    neighbors_np = np.full((num_nodes, max_degree), -1, dtype=np.int64)
    weights_np = np.zeros((num_nodes, max_degree), dtype=np.float32)
    neighbors_np[sorted_sources, row_position] = sorted_targets
    weights_np[sorted_sources, row_position] = sorted_weights

    isolated = np.nonzero(degrees == 0)[0]
    neighbors_np[isolated, 0] = isolated  # self-loop fallback so every node has >=1 valid step
    weights_np[isolated, 0] = 1.0

    adjacency_bool = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    adjacency_bool[sources, targets] = True
    adjacency_bool[isolated, isolated] = True

    return (
        torch.from_numpy(neighbors_np).to(device),
        torch.from_numpy(weights_np).to(device),
        adjacency_bool.to(device),
    )


def _random_walks(
    start_nodes: torch.Tensor,
    neighbors: torch.Tensor,
    base_weights: torch.Tensor,
    adjacency_bool: torch.Tensor,
    walk_length: int,
    p: float,
    q: float,
) -> torch.Tensor:
    """Vectorized 2nd-order biased random walk across a whole batch of starting
    nodes at once. Returns (batch, walk_length + 1) node-id tensor."""
    walks = [start_nodes]
    prev = start_nodes.clone()
    current = start_nodes.clone()

    for step in range(walk_length):
        cur_neighbors = neighbors[current]
        cur_weights = base_weights[current].clone()

        if step > 0:
            valid_slot = cur_neighbors >= 0
            is_return = cur_neighbors == prev.unsqueeze(1)
            prev_adj_row = adjacency_bool[prev]
            is_common = torch.gather(prev_adj_row, 1, cur_neighbors.clamp(min=0)) & valid_slot & ~is_return

            bias = torch.ones_like(cur_weights)
            bias = torch.where(is_return, bias / p, bias)
            other = valid_slot & ~is_return & ~is_common
            bias = torch.where(other, bias / q, bias)
            cur_weights = cur_weights * bias

        choice = torch.multinomial(cur_weights, 1).squeeze(1)
        next_node = cur_neighbors.gather(1, choice.unsqueeze(1)).squeeze(1)

        prev = current
        current = next_node
        walks.append(current)

    return torch.stack(walks, dim=1)


def _walks_to_pairs(walks: torch.Tensor, window: int):
    """Skip-gram (center, context) pairs from every position within `window` of
    each other in a walk, both directions."""
    centers, contexts = [], []
    walk_len = walks.shape[1]
    for offset in range(1, window + 1):
        if offset < walk_len:
            centers.append(walks[:, :-offset].reshape(-1))
            contexts.append(walks[:, offset:].reshape(-1))
            centers.append(walks[:, offset:].reshape(-1))
            contexts.append(walks[:, :-offset].reshape(-1))
    return torch.cat(centers), torch.cat(contexts)


class _SkipGram(nn.Module):
    def __init__(self, num_nodes: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        nn.init.uniform_(self.embedding.weight, -0.5 / embed_dim, 0.5 / embed_dim)

    def loss(self, center: torch.Tensor, context: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        center_emb = self.embedding(center)
        context_emb = self.embedding(context)
        neg_emb = self.embedding(negative)

        pos_loss = F.logsigmoid((center_emb * context_emb).sum(-1))
        # elementwise multiply + sum, not torch.bmm: a trailing dim of 1 routes bmm through
        # torch's Triton-JIT "outer product" kernel, which needs a C compiler at runtime --
        # this form is equivalent and avoids that dependency entirely.
        neg_loss = F.logsigmoid(-(neg_emb * center_emb.unsqueeze(1)).sum(-1)).sum(-1)

        return -(pos_loss + neg_loss).mean()


def compute_node2vec_embeddings(graph: nx.Graph, config: Node2VecConfig = None) -> np.ndarray:
    """Returns (num_nodes, embed_dim), row i is node i's embedding (node ids are 0..num_nodes-1)."""
    config = config or Node2VecConfig()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    num_nodes = graph.number_of_nodes()

    neighbors, base_weights, adjacency_bool = _build_padded_adjacency(graph, config.min_walk_weight, device)
    node_ids = torch.arange(num_nodes, device=device)
    start_nodes = node_ids.repeat(config.num_walks)
    walks = _random_walks(start_nodes, neighbors, base_weights, adjacency_bool, config.walk_length, config.p, config.q)
    centers, contexts = _walks_to_pairs(walks, config.window)

    model = _SkipGram(num_nodes, config.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    num_pairs = centers.shape[0]
    for _ in range(config.epochs):
        perm = torch.randperm(num_pairs, device=device)
        for start in range(0, num_pairs, config.batch_size):
            idx = perm[start : start + config.batch_size]
            negatives = torch.randint(0, num_nodes, (idx.shape[0], config.num_negative_samples), device=device)

            optimizer.zero_grad()
            loss = model.loss(centers[idx], contexts[idx], negatives)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        return model.embedding.weight.detach().cpu().numpy().astype(np.float32)
