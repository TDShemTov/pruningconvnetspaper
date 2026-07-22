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

import numpy as np
import torch
from torch.utils.data import Subset

from src.clustering import ClusterConfig, cluster_filters, rank_prune_candidates, select_filters_to_prune
from src.data.datasets import SplitConfig, build_splits
from src.embedding import ActivationConfig, build_activation_matrix
from src.eval.compare import ModelReport, compression_ratios, evaluate_model
from src.eval.flops_timing import TimingConfig
from src.graph import (
    DiffusionEmbedConfig,
    GCNEmbedConfig,
    GraphConfig,
    Node2VecConfig,
    RawEmbedConfig,
    SpectralEmbedConfig,
    build_similarity_graph,
    compute_diffusion_embeddings,
    compute_gcn_embeddings,
    compute_node2vec_embeddings,
    compute_raw_embeddings,
    compute_spectral_embeddings,
)
from src.models.builder import build_model
from src.pruning import L2PruneConfig, apply_l2_baseline_prune, apply_prune_plan
from src.run_log import RunLog
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
    graph_embedding_method: str = "node2vec"  # "node2vec" | "spectral" | "raw" | "diffusion" | "gcn"
    node2vec_config: Node2VecConfig = field(default_factory=Node2VecConfig)
    spectral_config: SpectralEmbedConfig = field(default_factory=SpectralEmbedConfig)
    raw_embed_config: RawEmbedConfig = field(default_factory=RawEmbedConfig)
    diffusion_config: DiffusionEmbedConfig = field(default_factory=DiffusionEmbedConfig)
    gcn_config: GCNEmbedConfig = field(default_factory=GCNEmbedConfig)

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
    # Writes log_dir/<timestamp>/log.json plus one steps/<NN>_<name>.json per
    # step -- each written the moment that step finishes, not batched up until
    # the whole pipeline is done, so a long run (baseline training alone can
    # take 10+ minutes) is traceable/self-diagnosing while it's still running,
    # not only after it completes. Set log_dir to None to skip writing a log.
    log_dir: Optional[str] = "experiments"
    # Prints a one-line start/done message (with duration) per step, so a long
    # step shows visible progress instead of going silent until it finishes.
    log_verbose: bool = True


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
    if config.graph_embedding_method == "diffusion":
        return compute_diffusion_embeddings(graph, activation_matrix, config.diffusion_config)
    if config.graph_embedding_method == "gcn":
        return compute_gcn_embeddings(graph, activation_matrix, config.gcn_config)
    raise ValueError(
        f"unknown graph_embedding_method '{config.graph_embedding_method}', "
        "expected 'node2vec', 'spectral', 'raw', 'diffusion', or 'gcn'"
    )


def _embedding_method_config(config: PipelineConfig):
    return {
        "node2vec": config.node2vec_config,
        "spectral": config.spectral_config,
        "raw": config.raw_embed_config,
        "diffusion": config.diffusion_config,
        "gcn": config.gcn_config,
    }[config.graph_embedding_method]


def _report_summary(report: ModelReport) -> dict:
    return {
        "test_metrics": report.test_metrics,
        "flops_params": report.flops_params,
        "inference": report.inference,
        "recalibration_history": report.recalibration_history,
    }


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    torch.manual_seed(config.seed)
    run_log = RunLog(config=config, log_dir=config.log_dir, verbose=config.log_verbose)

    # 1. data
    with run_log.step("data") as info:
        train_ds, test_ds, embed_ds, metadata = build_splits(
            config.dataset_name, root=config.data_root, config=config.split_config
        )
        if config.embed_sample_limit is not None:
            limit = min(config.embed_sample_limit, len(embed_ds))
            embed_ds = Subset(embed_ds, list(range(limit)))
        info["dataset_name"] = config.dataset_name
        info["num_train"] = len(train_ds)
        info["num_test"] = len(test_ds)
        info["num_embed"] = len(embed_ds)
        info["num_classes"] = metadata.num_classes
        info["in_channels"] = metadata.in_channels

    # 2. model + baseline training
    with run_log.step("baseline_training") as info:
        model = build_model(
            config.model_name,
            num_classes=metadata.num_classes,
            in_channels=metadata.in_channels,
            small_inputs=config.small_inputs,
        )
        train_result = train_model(model, train_ds, test_ds, config.baseline_train_config)
        info["model_name"] = config.model_name
        info["train_config"] = config.baseline_train_config
        info["history"] = train_result["history"]
        info["final_metrics"] = train_result["final_metrics"]

    input_shape = (metadata.in_channels, config.input_size, config.input_size)
    example_inputs = torch.randn(1, *input_shape)

    with run_log.step("baseline_eval") as info:
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
        info.update(_report_summary(baseline_report))

    # 3. activation extraction
    with run_log.step("activation_extraction") as info:
        activation_matrix = build_activation_matrix(model, embed_ds, config.activation_config)
        info["stats"] = activation_matrix.stats
        info["num_filters"] = activation_matrix.representation.shape[0]
        info["num_samples"] = activation_matrix.representation.shape[1]
        info["num_stats"] = activation_matrix.representation.shape[2]
        info["layer_names"] = activation_matrix.layer_names
        info["filters_per_layer"] = activation_matrix.filters_per_layer

    # 4. similarity graph
    with run_log.step("similarity_graph") as info:
        graph = build_similarity_graph(activation_matrix, config.graph_config)
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        degrees = [d for _, d in graph.degree()]
        info["graph_config"] = config.graph_config
        info["num_nodes"] = num_nodes
        info["num_edges"] = num_edges
        info["density"] = (2 * num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0
        info["avg_degree"] = (sum(degrees) / num_nodes) if num_nodes else 0.0
        info["isolated_nodes"] = sum(1 for d in degrees if d == 0)

    # 5. graph embedding
    with run_log.step("graph_embedding") as info:
        embeddings = _compute_embeddings(config, activation_matrix, graph)
        total_filters = embeddings.shape[0]
        info["method"] = config.graph_embedding_method
        info["method_config"] = _embedding_method_config(config)
        info["embed_dim"] = embeddings.shape[1]
        info["num_filters"] = total_filters

    # 6. clustering + prune-candidate ranking (ours)
    with run_log.step("clustering") as info:
        labels = cluster_filters(embeddings, config.cluster_config)
        candidates = rank_prune_candidates(embeddings, labels, config.cluster_config)
        to_prune = select_filters_to_prune(candidates, amount=config.prune_fraction, total_num_filters=total_filters)
        _, counts = np.unique(labels, return_counts=True)
        info["cluster_config"] = config.cluster_config
        info["num_clusters"] = int(len(counts))
        info["cluster_sizes_desc"] = sorted(counts.tolist(), reverse=True)
        info["num_prune_candidates"] = len(candidates)
        info["prune_fraction"] = config.prune_fraction
        info["num_selected_to_prune"] = len(to_prune)

    # 7. our method: prune -> recalibrate -> evaluate
    with run_log.step("ours_prune_recalibrate_eval") as info:
        ours_model = copy.deepcopy(model)
        prune_report = apply_prune_plan(ours_model, graph, to_prune, example_inputs)
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
        info["recalibration_config"] = config.recalibration_config
        info["skipped_layers"] = prune_report.skipped_layers
        info.update(_report_summary(ours_report))

    # 8. L2 baseline: prune (same fraction) -> SAME recalibration -> evaluate
    with run_log.step("l2_prune_recalibrate_eval") as info:
        l2_model = copy.deepcopy(model)
        l2_config = L2PruneConfig(pruning_ratio=config.prune_fraction, global_pruning=config.l2_global_pruning)
        apply_l2_baseline_prune(l2_model, example_inputs, l2_config)
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
        info["l2_config"] = l2_config
        info["recalibration_config"] = config.recalibration_config
        info.update(_report_summary(l2_report))

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
        step_durations_seconds=run_log.durations,
    )

    result.log_path = run_log.finalize(result)

    return result
