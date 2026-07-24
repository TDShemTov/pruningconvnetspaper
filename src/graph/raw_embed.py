"""No graph embedding at all — cluster directly on each filter's raw statistic
vector. The simplest possible alternative to node2vec: sidesteps every
graph-embedding quirk (disconnected components, zero-weight edges,
negative-sampling cross-talk between components) but loses the graph's
transitive/community signal — two filters each strongly similar to a shared
third filter, but only moderately similar to each other directly, won't be
pulled together the way a graph embedding does. Also useful as an ablation
baseline: does the graph-embedding step help at all over this?
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from src.embedding.activation_matrix import ActivationMatrix
from src.graph.similarity_graph import _flatten_filters


@dataclass
class RawEmbedConfig:
    # PCA-reduce to this many dims; None keeps the full flattened (samples * stats) vector.
    embed_dim: Optional[int] = 64
    # None passes random_state=None to sklearn's PCA -- non-reproducible across calls.
    seed: Optional[int] = 42
    # See _flatten_filters (similarity_graph.py) -- z-scores each of the 7
    # stats independently before PCA, since PCA's variance-maximizing
    # components would otherwise be dominated by whichever raw stat has the
    # largest magnitude (mean/max/std/median) at the expense of the rest
    # (skew/kurtosis/entropy).
    standardize_stats: bool = True


def compute_raw_embeddings(activation_matrix: ActivationMatrix, config: RawEmbedConfig = None) -> np.ndarray:
    """Returns (num_filters, embed_dim), row i is filter i's (optionally PCA-reduced) raw vector."""
    config = config or RawEmbedConfig()
    vectors = _flatten_filters(activation_matrix.representation, standardize=config.standardize_stats)

    if config.embed_dim is None:
        return vectors.astype(np.float32)

    n_components = min(config.embed_dim, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=n_components, random_state=config.seed)
    return pca.fit_transform(vectors).astype(np.float32)
