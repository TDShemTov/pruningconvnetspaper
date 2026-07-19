"""Graph embedding via node2vec (CLAUDE.md Step 5) — isolated in its own file
so it's a one-file swap if a different graph-embedding method replaces it later.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from node2vec import Node2Vec


@dataclass
class Node2VecConfig:
    embed_dim: int = 64
    walk_length: int = 30
    num_walks: int = 200
    window: int = 10
    min_count: int = 1
    p: float = 1.0
    q: float = 1.0
    workers: int = 1
    seed: int = 42
    # Floor applied to edge weights only for random-walk sampling (see _floor_weights).
    min_walk_weight: float = 1e-6


def _floor_weights(graph: nx.Graph, min_weight: float) -> nx.Graph:
    """node2vec uses edge weights as walk transition probabilities. similarity_graph's
    min-max normalization can legitimately assign an edge exactly weight 0.0 (the least
    similar kept edge) — if that's a low-degree node's only edge, its total outgoing
    weight is 0 and the walk generator divides by zero. Floor a *copy* of the graph's
    weights so walks stay well-defined; the original graph's true [0,1] weights are untouched.
    """
    floored = graph.copy()
    for _, _, d in floored.edges(data=True):
        d["weight"] = max(d["weight"], min_weight)
    return floored


def compute_node2vec_embeddings(graph: nx.Graph, config: Node2VecConfig = None) -> np.ndarray:
    """Returns (num_nodes, embed_dim), row i is node i's embedding (node ids are 0..num_nodes-1)."""
    config = config or Node2VecConfig()
    walk_graph = _floor_weights(graph, config.min_walk_weight)
    n2v = Node2Vec(
        walk_graph,
        dimensions=config.embed_dim,
        walk_length=config.walk_length,
        num_walks=config.num_walks,
        p=config.p,
        q=config.q,
        workers=config.workers,
        seed=config.seed,
        weight_key="weight",
        quiet=True,
    )
    model = n2v.fit(window=config.window, min_count=config.min_count)

    num_nodes = graph.number_of_nodes()
    embeddings = np.zeros((num_nodes, config.embed_dim), dtype=np.float32)
    for node in graph.nodes():
        embeddings[node] = model.wv[str(node)]
    return embeddings
