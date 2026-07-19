# pruningconvnetspaper

Graph-embedding-based structured CNN pruning. Each filter is represented as a
statistical "signature" over a held-out embed split, filters are connected
into a similarity graph (cosine similarity, thresholded), graph embeddings
group filters by representational redundancy, and the most redundant filters
are pruned first — compared against a standard magnitude-based baseline
(torch-pruning L2, matched pruning ratio, same recalibrate-then-evaluate path).

## Quickstart

`notebooks/run_pipeline.ipynb` is a thin Colab wrapper around
`run_pipeline(config)` (`src/pipeline.py`) — clone, `pip install -r
requirements.txt`, build a `PipelineConfig`, run, inspect the results table.
All the actual logic lives in `src/`; the notebook doesn't reimplement
anything.

## Config reference

`run_pipeline(config: PipelineConfig)` is the single entry point — every
section below is a `PipelineConfig` field (or a field on a sub-config it
holds), and every field is set explicitly in the notebook's config cell.

### Data
- `dataset_name` (`"cifar10"`) — any of the 20 datasets in
  `src/data/datasets.py`'s `DATASET_REGISTRY`.
- `split_config: SplitConfig` — `train_frac`/`test_frac`/`embed_frac`
  (default `0.8`/`0.1`/`0.1`, must sum to 1.0), `seed`. Splits are
  class-stratified.
- `embed_sample_limit` — cap how many embed-split samples are used for
  activation extraction (`None` = use the whole split).

### Model
- `model_name` (`"resnet18"`) — `simplecnn` / `resnet18`/`34`/`50` /
  `vgg11_bn`…`vgg19_bn` / `densenet121`/`169`/`201`.
- `small_inputs` (`True`) — CIFAR-style stem adaptation for ≤96px images.
- `input_size` (`32`) — spatial H=W, for both `example_inputs` and timing.

### Training (`TrainConfig`)
Used twice, as two separate instances: `baseline_train_config` (training
from scratch) and `recalibration_config` (retraining a pruned model before
evaluation — typically fewer epochs / a lower lr).
- `epochs`, `batch_size`, `lr`, `momentum` (only used by `"sgd"`), `weight_decay`
- `optimizer` (`"sgd"` / `"adam"` / `"adamw"`), `scheduler` (`"cosine"` /
  `"step"` / `"none"`)
- `device`, `num_workers`, `seed`

`lr`/`weight_decay` defaults (`0.1`/`5e-4`) are SGD-tuned (standard
ResNet/CIFAR values) and don't auto-adjust per optimizer. Adam/AdamW
typically want a much smaller `lr` (`~1e-3`-`4e-3`); AdamW specifically also
tends to want a larger `weight_decay` (`~0.01`-`0.05`), since its decay isn't
entangled with Adam's per-parameter adaptive scaling the way plain Adam's is.
Switching `optimizer` without re-tuning both is a common way to get a
silently-diverging or stalled run.

### Activation extraction (`activation_config: ActivationConfig`)
- `stats` — any subset of `{mean, max, std, median, skew, kurtosis, entropy}`.
- `batch_size`, `entropy_bins`, `device`, `num_workers`.

### Similarity graph (`graph_config: GraphConfig`)
- `similarity_threshold` (`0.7`)
- `same_layer_only` (`False`) / `cross_layer_threshold` (`None`) — same-layer
  vs. global vs. hybrid cross-layer topology.
- `device`

### Graph embedding — swap point
`graph_embedding_method` selects which method runs; each has its own config
field on `PipelineConfig`:

| `graph_embedding_method` | Config class | Uses node content? | Uses topology? | Notes |
|---|---|---|---|---|
| `"node2vec"` (default) | `Node2VecConfig` | No | Yes | Custom GPU/CPU biased random walk + skip-gram. Params: `embed_dim`, `walk_length`, `num_walks`, `window`, `p`, `q`, `epochs`, `batch_size`, `walk_batch_size`, `lr`, `num_negative_samples`, `device`, `seed`, `min_walk_weight` |
| `"spectral"` | `SpectralEmbedConfig` | No | Yes | Deterministic Laplacian eigenmap, no walk hyperparameters. Params: `embed_dim`, `seed` |
| `"raw"` | `RawEmbedConfig` | Yes | No | No graph at all — optionally PCA-reduced raw statistic vector. Params: `embed_dim` (PCA target, `None` = no reduction), `seed` |
| `"diffusion"` | `DiffusionEmbedConfig` | Yes | Yes | SGC-style: propagate the raw vector across the graph (`Â^k · X`), no training loop. Params: `embed_dim`, `k` (propagation hops), `use_edge_weights`, `seed` |
| `"gcn"` | `GCNEmbedConfig` | Yes | Yes | Graph autoencoder, 2-layer `GCNConv` encoder. Params: `embed_dim`, `hidden_dim`, `epochs`, `lr`, `use_edge_weights`, `device`, `seed` |

`"diffusion"` and `"gcn"` both expose `use_edge_weights` (cosine-similarity
strength vs. every kept edge treated equally) for a matched
weighted-vs-unweighted ablation.

### Clustering (`cluster_config: ClusterConfig`)
- `method` (`"ward"`) — any scipy linkage method (`single`/`complete`/`average`/…).
- `metric` (`"euclidean"`) — ward requires euclidean.
- Exactly one of `n_clusters` / `distance_threshold`.
- `min_cluster_size` (`3`) — clusters smaller than this are never pruned.

### Pruning
- `prune_fraction` (`0.3`) — shared between our method and the L2 baseline,
  so the comparison is at a matched pruning ratio.
- `l2_global_pruning` (`True`) — network-wide L2 selection, not a fixed
  per-layer quota.

### Comparison eval
- `timing_config: TimingConfig` — `batch_size`, `num_warmup`, `num_trials`, `device`.
- `eval_batch_size` (`128`).

### Misc
- `seed` (`42`).
- `log_dir` (`"experiments"`) — writes `<log_dir>/<timestamp>/log.json` (full
  config + per-step durations + result) on every run; `None` skips writing.

## Results

`run_pipeline` returns a `PipelineResult`: `baseline` / `ours` / `l2_baseline`
(`ModelReport` — accuracy/balanced accuracy/F1/AUC, FLOPs/params, inference
timing + peak memory), `ours_vs_baseline` / `l2_vs_baseline` (compression
ratios, >1 means smaller/faster than baseline), plus filter/edge counts and
per-step timings.
