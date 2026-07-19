"""Hierarchical clustering over filter embeddings -> prune-candidate ranking
(CLAUDE.md Step 7).

Intuition: filters that cluster tightly together are redundant with each
other. Larger clusters mean more redundancy, so they're pruned from first.
Within a chosen cluster, the most mutually-similar (most redundant, least
unique) members are pruned first, and at least one representative — the
*least* redundant member — is always kept per eligible cluster.

This module is agnostic to what a "filter" is architecturally: it operates
purely on an `(num_filters, embed_dim)` embedding array and returns global
filter indices in prune-priority order. Mapping those indices back to actual
(layer, channel) pairs is the pruning module's job (uses the graph's node
attributes), not this one's.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


@dataclass
class ClusterConfig:
    method: str = "ward"  # scipy linkage method: ward/single/complete/average/...
    metric: str = "euclidean"  # ward requires euclidean; other methods can use e.g. cosine
    n_clusters: Optional[int] = None  # exactly one of n_clusters / distance_threshold
    distance_threshold: Optional[float] = None
    min_cluster_size: int = 3  # clusters smaller than this are never touched (too "unique")

    def __post_init__(self) -> None:
        if (self.n_clusters is None) == (self.distance_threshold is None):
            raise ValueError("exactly one of n_clusters or distance_threshold must be set")


@dataclass
class PruneCandidate:
    filter_idx: int
    cluster_id: int
    cluster_size: int
    redundancy_rank: int  # 0 = most redundant (most similar to its cluster-mates)


def cluster_filters(embeddings: np.ndarray, config: ClusterConfig) -> np.ndarray:
    """Returns a cluster label per filter, shape (num_filters,)."""
    z = linkage(embeddings, method=config.method, metric=config.metric)
    if config.n_clusters is not None:
        return fcluster(z, t=config.n_clusters, criterion="maxclust")
    return fcluster(z, t=config.distance_threshold, criterion="distance")


def _rank_by_redundancy(embeddings: np.ndarray, member_idxs: np.ndarray) -> np.ndarray:
    """Reorders member_idxs so index 0 is the most redundant (highest average
    cosine similarity to the rest of the cluster — prune first) and the last
    entry is the most unique (keep as the cluster's representative)."""
    if len(member_idxs) == 1:
        return member_idxs

    vecs = embeddings[member_idxs]
    unit = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    sim = unit @ unit.T
    np.fill_diagonal(sim, 0.0)
    avg_sim = sim.sum(axis=1) / (len(member_idxs) - 1)
    order = np.argsort(-avg_sim)
    return member_idxs[order]


def rank_prune_candidates(
    embeddings: np.ndarray, cluster_labels: np.ndarray, config: ClusterConfig
) -> List[PruneCandidate]:
    """Cluster-size-first, most-redundant-within-cluster-second prune priority.
    Clusters smaller than config.min_cluster_size are skipped entirely. Every
    eligible cluster always keeps at least one (its least redundant) member.
    """
    candidates: List[PruneCandidate] = []

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    order = np.argsort(-counts)  # biggest cluster first

    for cluster_id, size in zip(unique_labels[order], counts[order]):
        if size < config.min_cluster_size:
            continue

        member_idxs = np.where(cluster_labels == cluster_id)[0]
        ranked = _rank_by_redundancy(embeddings, member_idxs)
        prunable = ranked[:-1]  # keep the least redundant member as representative

        for rank, filter_idx in enumerate(prunable):
            candidates.append(
                PruneCandidate(
                    filter_idx=int(filter_idx),
                    cluster_id=int(cluster_id),
                    cluster_size=int(size),
                    redundancy_rank=rank,
                )
            )

    return candidates


def select_filters_to_prune(
    candidates: List[PruneCandidate], amount: Union[int, float], total_num_filters: int
) -> List[int]:
    """`amount` is either a fixed count, or a float in (0, 1] interpreted as a
    fraction of `total_num_filters` (the whole network's filter count, not just
    the eligible/prunable pool — matching how compression ratios are usually
    reported). The result is capped at len(candidates): if min_cluster_size
    protects most of the network, you may get fewer prunes than requested.
    """
    if isinstance(amount, float):
        if not 0.0 < amount <= 1.0:
            raise ValueError(f"fractional amount must be in (0, 1], got {amount}")
        k = int(round(total_num_filters * amount))
    else:
        k = amount

    k = min(k, len(candidates))
    return [c.filter_idx for c in candidates[:k]]
