"""Graph embedding via SGC-style feature diffusion -- propagate each filter's
raw statistic vector across the similarity graph, then optionally PCA-reduce.
No training loop, no random walks: the only "learning" step is propagation, a
fixed number of sparse matrix multiplications (`A_norm^k @ X`), following
Simple Graph Convolution (Wu et al., 2019)'s observation that most of a GCN's
benefit comes from the propagation step itself, not the learned weights
between layers.

Sits between raw_embed.py (content only, no graph at all) and
node2vec_embed.py/spectral_embed.py (topology only, no node content): this
smooths each filter's own statistic vector using its graph neighbors'
vectors, so two filters that are mutually similar (and so share much of the
same neighborhood) end up with more similar embeddings than their raw
vectors alone would give -- with no trainable parameters, walk generation,
or negative sampling at all.

`use_edge_weights` toggles whether propagation is weighted by cosine
similarity strength (the edge's weight) or treats every kept edge as equally
strong (binarized adjacency, edge presence only). The same toggle exists on
gcn_embed.py's GAE encoder, so both new content+topology methods support the
same weighted-vs-unweighted ablation.
"""

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA

from src.embedding.activation_matrix import ActivationMatrix
from src.graph.similarity_graph import _flatten_filters


@dataclass
class DiffusionEmbedConfig:
    # PCA-reduce the diffused vector to this many dims; None keeps the full
    # flattened (samples * stats) dimensionality.
    embed_dim: Optional[int] = 64
    # Number of propagation hops (applications of the normalized adjacency).
    k: int = 2
    # Weighted (cosine-similarity strength) vs unweighted (every kept edge
    # counted equally) propagation.
    use_edge_weights: bool = True
    seed: int = 42


def _normalized_adjacency_with_self_loops(graph: nx.Graph, use_edge_weights: bool) -> sparse.csr_matrix:
    """Symmetric-normalized D^-1/2 (A + I) D^-1/2. Self-loops are added before
    computing degree, so every node has nonzero degree -- unlike node2vec's
    walk sampling, no isolated-node epsilon-floor guard is needed here."""
    num_nodes = graph.number_of_nodes()
    weight_attr = "weight" if use_edge_weights else None
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(num_nodes), weight=weight_attr, format="csr")
    adjacency = adjacency + sparse.eye(num_nodes, format="csr")

    degree = np.asarray(adjacency.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt_mat = sparse.diags(d_inv_sqrt)

    return d_inv_sqrt_mat @ adjacency @ d_inv_sqrt_mat


def compute_diffusion_embeddings(
    graph: nx.Graph, activation_matrix: ActivationMatrix, config: DiffusionEmbedConfig = None
) -> np.ndarray:
    """Returns (num_filters, embed_dim), row i is filter i's diffused (and
    optionally PCA-reduced) raw vector."""
    config = config or DiffusionEmbedConfig()
    vectors = _flatten_filters(activation_matrix.representation)
    normalized_adjacency = _normalized_adjacency_with_self_loops(graph, config.use_edge_weights)

    diffused = vectors
    for _ in range(config.k):
        diffused = normalized_adjacency @ diffused

    if config.embed_dim is None:
        return diffused.astype(np.float32)

    n_components = min(config.embed_dim, diffused.shape[0], diffused.shape[1])
    pca = PCA(n_components=n_components, random_state=config.seed)
    return pca.fit_transform(diffused).astype(np.float32)
