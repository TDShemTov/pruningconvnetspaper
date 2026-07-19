"""Per-filter statistics collapsing a feature map's spatial dimensions to a
scalar (see CLAUDE.md Step 3). Each function operates on a batch of feature
maps at once — shape (B, C, H, W) in, shape (B, C) out — so a whole batch's
worth of filters is reduced in one vectorized pass rather than looping over
samples/channels.
"""

from typing import Dict, Sequence

import torch

STAT_NAMES = ("mean", "max", "std", "median", "skew", "kurtosis", "entropy")


def _entropy(flat: torch.Tensor, n_bins: int, eps: float) -> torch.Tensor:
    """Shannon entropy of each channel's value histogram. `flat` is (B, C, N)."""
    mn = flat.amin(dim=-1, keepdim=True)
    mx = flat.amax(dim=-1, keepdim=True)
    scale = (mx - mn).clamp(min=eps)
    bin_idx = ((flat - mn) / scale * n_bins).long().clamp(0, n_bins - 1)  # (B, C, N)

    counts = torch.zeros(*flat.shape[:2], n_bins, device=flat.device, dtype=flat.dtype)
    counts.scatter_add_(-1, bin_idx, torch.ones_like(flat))
    p = counts / flat.shape[-1]
    return -(p * torch.log(p + eps)).sum(dim=-1)


def compute_stats(
    feature_map: torch.Tensor,
    stats: Sequence[str],
    entropy_bins: int = 32,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """feature_map: (B, C, H, W). Returns {stat_name: (B, C)} for each name in `stats`."""
    flat = feature_map.flatten(2)  # (B, C, N)
    mean = flat.mean(dim=-1)

    results: Dict[str, torch.Tensor] = {}
    if "mean" in stats:
        results["mean"] = mean
    if "max" in stats:
        results["max"] = flat.amax(dim=-1)
    if "median" in stats:
        # torch's .median() returns the lower of the two middle values on even-length
        # input rather than interpolating; quantile(0.5) matches the standard/numpy definition.
        results["median"] = torch.quantile(flat, 0.5, dim=-1)

    if any(s in stats for s in ("std", "skew", "kurtosis")):
        centered = flat - mean.unsqueeze(-1)
        var = (centered**2).mean(dim=-1)
        std = torch.sqrt(var + eps)
        if "std" in stats:
            results["std"] = std
        if "skew" in stats:
            m3 = (centered**3).mean(dim=-1)
            results["skew"] = m3 / (std**3 + eps)
        if "kurtosis" in stats:
            m4 = (centered**4).mean(dim=-1)
            results["kurtosis"] = m4 / (std**4 + eps) - 3.0

    if "entropy" in stats:
        results["entropy"] = _entropy(flat, n_bins=entropy_bins, eps=eps)

    return results
