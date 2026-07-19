"""Single config-driven entry point wiring every module built so far into one
call: data -> baseline training -> activations -> similarity graph -> graph
embedding -> clustering -> pruning (ours + L2 baseline) -> recalibration ->
comparison eval.

This is the code realization of the "Config surface" section in CLAUDE.md --
that section has been a documentation-only draft throughout the project;
`PipelineConfig` is exactly that surface, and `run_pipeline` is what actually
runs it. The Colab notebook is meant to be a thin wrapper around this: clone,
install requirements, construct one `PipelineConfig`, call `run_pipeline`.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch.utils.data import Subset

from src.clustering import ClusterConfig, cluster_filters, rank_prune_candidates, select_filters_to_prune
from src.data.datasets import SplitConfig, build_splits
from src.embedding import ActivationConfig, build_activation_matrix
from src.eval.compare import ModelReport, compression_ratios, evaluate_model
from src.eval.flops_timing import TimingConfig
from src.graph import (
    GraphConfig,
    Node2VecConfig,
    RawEmbedConfig,
    SpectralEmbedConfig,
    build_similarity_graph,
    compute_node2vec_embeddings,
    compute_raw_embeddings,
    compute_spectral_embeddings,
)
from src.models.builder import build_model
from src.pruning import L2PruneConfig, apply_l2_baseline_prune, apply_prune_plan
from src.run_log import timed_step, write_run_log
from src.train import TrainConfig, train_model


@dataclass
class PipelineConfig:
    # --- data ---
    dataset_name: str = "cifar10"
    data_root: str = "./data"
    split_config: SplitConfig = field(default_factory=SplitConfig)
    embed_sample_limit: Optional[int] = None  # None = use the whole embed split

    # --- model ---
    model_name: str = "resnet18"
    small_inputs: bool = True
    input_size: int = 32  # spatial H=W, for both example_inputs and the timing benchmark

    # --- baseline training ---
    baseline_train_config: TrainConfig = field(default_factory=lambda: TrainConfig(epochs=20))

    # --- activation extraction ---
    activation_config: ActivationConfig = field(default_factory=ActivationConfig)

    # --- similarity graph ---
    graph_config: GraphConfig = field(default_factory=GraphConfig)

    # --- graph embedding (swap point: which method actually runs) ---
    graph_embedding_method: str = "node2vec"  # "node2vec" | "spectral" | "raw"
    node2vec_config: Node2VecConfig = field(default_factory=Node2VecConfig)
    spectral_config: SpectralEmbedConfig = field(default_factory=SpectralEmbedConfig)
    raw_embed_config: RawEmbedConfig = field(default_factory=RawEmbedConfig)

    # --- clustering ---
    cluster_config: ClusterConfig = field(default_factory=lambda: ClusterConfig(n_clusters=20, min_cluster_size=3))

    # --- pruning: one shared fraction for both methods, so the comparison is
    # at a matched pruning ratio, not an apples-to-oranges one ---
    prune_fraction: float = 0.3
    l2_global_pruning: bool = True

    # --- recalibration: retraining a pruned model before evaluation. Separate
    # from baseline_train_config on purpose -- recalibration typically wants
    # fewer epochs / a lower lr than training from scratch, and how much it
    # needs for a given prune ratio is itself an open empirical question. ---
    recalibration_config: TrainConfig = field(default_factory=lambda: TrainConfig(epochs=5, lr=0.005))

    # --- comparison eval ---
    timing_config: TimingConfig = field(default_factory=TimingConfig)
    eval_batch_size: int = 128

    seed: int = 42

    # --- run logging ---
    # Writes log_dir/<timestamp>/log.json with this config, per-step wall-clock
    # durations, and the result, so a run is traceable/self-diagnosing without
    # ad-hoc timing instrumentation. Set to None to skip writing a log.
    log_dir: Optional[str] = "experiments"


@dataclass
class PipelineResult:
    baseline: ModelReport
    ours: ModelReport
    l2_baseline: ModelReport
    ours_vs_baseline: Dict[str, float]
    l2_vs_baseline: Dict[str, float]
    num_filters: int
    num_prune_candidates: int
    num_pruned: int
    graph_num_edges: int
    step_durations_seconds: Dict[str, float] = field(default_factory=dict)
    log_path: Optional[str] = None


def _compute_embeddings(config: PipelineConfig, activation_matrix, graph):
    if config.graph_embedding_method == "node2vec":
        return compute_node2vec_embeddings(graph, config.node2vec_config)
    if config.graph_embedding_method == "spectral":
        return compute_spectral_embeddings(graph, config.spectral_config)
    if config.graph_embedding_method == "raw":
        return compute_raw_embeddings(activation_matrix, config.raw_embed_config)
    raise ValueError(
        f"unknown graph_embedding_method '{config.graph_embedding_method}', "
        "expected 'node2vec', 'spectral', or 'raw'"
    )


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    torch.manual_seed(config.seed)
    durations: Dict[str, float] = {}

    # 1. data
    with timed_step(durations, "data"):
        train_ds, test_ds, embed_ds, metadata = build_splits(
            config.dataset_name, root=config.data_root, config=config.split_config
        )
        if config.embed_sample_limit is not None:
            limit = min(config.embed_sample_limit, len(embed_ds))
            embed_ds = Subset(embed_ds, list(range(limit)))

    # 2. model + baseline training
    with timed_step(durations, "baseline_training"):
        model = build_model(
            config.model_name,
            num_classes=metadata.num_classes,
            in_channels=metadata.in_channels,
            small_inputs=config.small_inputs,
        )
        train_model(model, train_ds, test_ds, config.baseline_train_config)

    input_shape = (metadata.in_channels, config.input_size, config.input_size)
    example_inputs = torch.randn(1, *input_shape)

    with timed_step(durations, "baseline_eval"):
        baseline_report = evaluate_model(
            "baseline",
            copy.deepcopy(model),
            train_ds,
            test_ds,
            example_inputs,
            input_shape,
            recalibration_config=None,
            timing_config=config.timing_config,
            eval_batch_size=config.eval_batch_size,
        )

    # 3. activation extraction
    with timed_step(durations, "activation_extraction"):
        activation_matrix = build_activation_matrix(model, embed_ds, config.activation_config)

    # 4. similarity graph
    with timed_step(durations, "similarity_graph"):
        graph = build_similarity_graph(activation_matrix, config.graph_config)

    # 5. graph embedding
    with timed_step(durations, "graph_embedding"):
        embeddings = _compute_embeddings(config, activation_matrix, graph)
        total_filters = embeddings.shape[0]

    # 6. clustering + prune-candidate ranking (ours)
    with timed_step(durations, "clustering"):
        labels = cluster_filters(embeddings, config.cluster_config)
        candidates = rank_prune_candidates(embeddings, labels, config.cluster_config)
        to_prune = select_filters_to_prune(candidates, amount=config.prune_fraction, total_num_filters=total_filters)

    # 7. our method: prune -> recalibrate -> evaluate
    with timed_step(durations, "ours_prune_recalibrate_eval"):
        ours_model = copy.deepcopy(model)
        apply_prune_plan(ours_model, graph, to_prune, example_inputs)
        ours_report = evaluate_model(
            "ours",
            ours_model,
            train_ds,
            test_ds,
            example_inputs,
            input_shape,
            recalibration_config=config.recalibration_config,
            timing_config=config.timing_config,
            eval_batch_size=config.eval_batch_size,
        )

    # 8. L2 baseline: prune (same fraction) -> SAME recalibration -> evaluate
    with timed_step(durations, "l2_prune_recalibrate_eval"):
        l2_model = copy.deepcopy(model)
        apply_l2_baseline_prune(
            l2_model,
            example_inputs,
            L2PruneConfig(pruning_ratio=config.prune_fraction, global_pruning=config.l2_global_pruning),
        )
        l2_report = evaluate_model(
            "l2_baseline",
            l2_model,
            train_ds,
            test_ds,
            example_inputs,
            input_shape,
            recalibration_config=config.recalibration_config,
            timing_config=config.timing_config,
            eval_batch_size=config.eval_batch_size,
        )

    result = PipelineResult(
        baseline=baseline_report,
        ours=ours_report,
        l2_baseline=l2_report,
        ours_vs_baseline=compression_ratios(baseline_report, ours_report),
        l2_vs_baseline=compression_ratios(baseline_report, l2_report),
        num_filters=total_filters,
        num_prune_candidates=len(candidates),
        num_pruned=len(to_prune),
        graph_num_edges=graph.number_of_edges(),
        step_durations_seconds=durations,
    )

    if config.log_dir is not None:
        result.log_path = write_run_log(config, durations, result, config.log_dir)

    return result
