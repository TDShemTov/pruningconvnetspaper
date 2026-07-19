"""FLOPs/param counting and inference-speed/memory measurement (CLAUDE.md Step 9).

FLOPs/params via torch-pruning's own counter (already a dependency, and it's
the same library doing the actual pruning, so counts stay consistent with
whatever layer types it understands as prunable).
"""

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp


@dataclass
class FlopsParamsReport:
    ops: float
    params: int


def count_ops_and_params(model: nn.Module, example_inputs: torch.Tensor) -> FlopsParamsReport:
    ops, params = tp.utils.count_ops_and_params(model, example_inputs)
    return FlopsParamsReport(ops=ops, params=params)


@dataclass
class TimingConfig:
    batch_size: int = 1
    num_warmup: int = 10
    num_trials: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class InferenceBenchmark:
    mean_time_s: float
    std_time_s: float
    peak_memory_bytes: Optional[float]  # None on CPU (torch's peak-memory API is CUDA-only)


def measure_inference(model: nn.Module, input_shape: Sequence[int], config: TimingConfig = None) -> InferenceBenchmark:
    """Times `num_trials` independent single-forward-pass calls (each its own
    timed+synchronized trial, not one aggregate loop divided by iteration
    count), on a fixed-size synthetic batch -- isolates the model's own
    inference cost from DataLoader/I/O overhead, matching the standard
    "inference latency" metric pruning papers report. Warm-up before timing
    matters on GPU (cuDNN kernel-selection/autotuning on the first few calls),
    and `torch.cuda.synchronize()` matters because CUDA calls are async from
    the CPU's perspective -- without it you'd measure kernel-launch overhead,
    not actual execution time.
    """
    config = config or TimingConfig()
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    x = torch.randn(config.batch_size, *input_shape, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(config.num_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(config.num_trials):
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    peak_memory = float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    times_arr = np.array(times)

    return InferenceBenchmark(
        mean_time_s=float(times_arr.mean()),
        std_time_s=float(times_arr.std()),
        peak_memory_bytes=peak_memory,
    )
