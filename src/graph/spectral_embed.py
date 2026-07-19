"""Graph embedding via spectral embedding / Laplacian eigenmaps — a
deterministic alternative to node2vec with no random-walk hyperparameters
(walk_length, num_walks, p, q, window) to tune.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.manifold import SpectralEmbedding


@dataclass
class SpectralEmbedConfig:
    embed_dim: int = 64
    seed: int = 42


def compute_spectral_embeddings(graph: nx.Graph, config: SpectralEmbedConfig = None) -> np.ndarray:
    """Returns (num_nodes, embed_dim), row i is node i's embedding (node ids are 0..num_nodes-1)."""
    config = config or SpectralEmbedConfig()
    nodelist = sorted(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodelist, weight="weight")

    n_components = min(config.embed_dim, len(nodelist) - 1) if len(nodelist) > 1 else 1
    model = SpectralEmbedding(n_components=n_components, affinity="precomputed", random_state=config.seed)
    embeddings = model.fit_transform(adjacency)
    return embeddings.astype(np.float32)
