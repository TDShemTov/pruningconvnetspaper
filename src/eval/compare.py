"""Orchestrates CLAUDE.md Step 9: recalibrate a pruned model, then evaluate
it (accuracy/balanced accuracy/F1/AUC, FLOPs/params, inference speed/memory)
so our method, the L2 baseline, and the unpruned model can be compared side
by side.

Recalibration -- retraining a pruned model on the train split for a few
epochs before evaluating it, since pruning always costs some accuracy that
a bit of retraining recovers -- is a `TrainConfig` passed in here, not a
hardcoded constant: `recalibration_config.epochs` (and every other training
hyperparameter) is tunable per run, and how much recalibration a given prune
ratio actually needs is itself an open empirical question.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.eval.flops_timing import (
    FlopsParamsReport,
    InferenceBenchmark,
    TimingConfig,
    count_ops_and_params,
    measure_inference,
)
from src.train import TrainConfig, evaluate as evaluate_loop, train_model


@dataclass
class ModelReport:
    name: str
    test_metrics: dict
    flops_params: FlopsParamsReport
    inference: InferenceBenchmark
    # Per-epoch recalibration metrics (loss/acc/balanced_acc/f1/auc), None when
    # recalibration_config was None (the already-trained unpruned baseline).
    recalibration_history: Optional[List[dict]] = field(default=None)


def evaluate_model(
    name: str,
    model: nn.Module,
    train_dataset: Dataset,
    test_dataset: Dataset,
    example_inputs: torch.Tensor,
    input_shape: Sequence[int],
    recalibration_config: Optional[TrainConfig],
    timing_config: TimingConfig = None,
    eval_batch_size: int = 128,
) -> ModelReport:
    """`recalibration_config=None` skips retraining entirely -- use that for
    the already-fully-trained unpruned baseline. Pass a `TrainConfig` (its
    `epochs` field is the tunable recalibration length) to retrain a pruned
    model before evaluating it.
    """
    timing_config = timing_config or TimingConfig()

    recalibration_history = None
    if recalibration_config is not None:
        train_result = train_model(model, train_dataset, test_dataset, recalibration_config)
        recalibration_history = train_result["history"]

    device = torch.device(timing_config.device)
    model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    test_metrics = evaluate_loop(model, test_loader, device)

    flops_params = count_ops_and_params(model, example_inputs.to(device))
    inference = measure_inference(model, input_shape, timing_config)

    return ModelReport(
        name=name,
        test_metrics=test_metrics,
        flops_params=flops_params,
        inference=inference,
        recalibration_history=recalibration_history,
    )


def compression_ratios(baseline: ModelReport, other: ModelReport) -> Dict[str, float]:
    """>1 means `other` is smaller/faster than `baseline`."""
    return {
        "params_compression": baseline.flops_params.params / other.flops_params.params,
        "ops_compression": baseline.flops_params.ops / other.flops_params.ops,
        "speedup": baseline.inference.mean_time_s / other.inference.mean_time_s,
    }
