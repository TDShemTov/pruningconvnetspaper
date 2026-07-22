"""Per-run logging: one JSON file per pipeline step (time + whatever
step-specific metrics that step attaches -- training epoch history,
activation-matrix shape, graph size, embedding-method config, pruning
results, ...) plus one top-level log.json tying them together and ranking
steps by duration.

Written incrementally -- each step's file (and the top-level log.json
summary) hits disk the moment that step finishes, not batched up until the
whole pipeline is done. A single `run_pipeline` call can easily run 10+
minutes (baseline training alone defaults to 20 epochs), and writing only at
the very end meant a long run produced zero visible output the entire time
it was running, on Colab or anywhere else -- if it hung or crashed midway,
there was nothing on disk to diagnose from either. `RunLog.step` also prints
a one-line start/done message per step (toggle via `verbose=False`) for the
same reason: something should show up on screen well before a 10-minute step
finishes.

The point is diagnosing *where a run's time actually goes* without
re-instrumenting anything ad hoc (as happened while chasing down the
node2vec/graph-construction bottleneck): "what's slow" is answered by
log.json's `steps_by_duration_desc`, "what happened, roughly" by its `steps`
(execution order, condensed per-step content) plus `result` (the final
pipeline-wide comparison) in that same file, and "exactly what happened in
step X" by that step's own full-detail steps/<NN>_<name>.json.
"""

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _summarize(obj: Any, max_list_len: int) -> Any:
    """Condenses an already-jsonable structure for the top-level log.json
    overview: long lists (e.g. a 20-epoch training history, a cluster-size
    distribution) collapse to their length plus first/last entries -- for a
    history list that's exactly the initial and final epoch's metrics, the
    two points you actually want at a glance. Full detail always stays
    available in that step's own steps/<NN>_<name>.json file; this is purely
    a smaller view for the combined summary.
    """
    if isinstance(obj, dict):
        return {k: _summarize(v, max_list_len) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > max_list_len:
            return {
                "length": len(obj),
                "first": _summarize(obj[0], max_list_len),
                "last": _summarize(obj[-1], max_list_len),
            }
        return [_summarize(v, max_list_len) for v in obj]
    return obj


class RunLog:
    """Accumulates one entry per pipeline step, in the order they run, and
    writes each one to disk as soon as it finishes.

    Usage: `with run_log.step("name") as info:` times the block; the block
    writes whatever metrics it wants into `info` (dicts, dataclasses, arrays
    -- anything `_to_jsonable` can walk) and they're captured alongside the
    duration. If `log_dir` is set, `log_dir/<timestamp>/steps/<NN>_<name>.json`
    and a running `log_dir/<timestamp>/log.json` summary are written the
    moment the `with` block exits -- call `finalize(result)` once, after the
    last step, to fold the pipeline's final result into that summary.
    """

    def __init__(
        self,
        config: Any = None,
        log_dir: Optional[str] = None,
        verbose: bool = True,
        overview_max_list_len: int = 5,
    ) -> None:
        self.steps: List[Dict[str, Any]] = []
        self.config = config
        self.verbose = verbose
        self.overview_max_list_len = overview_max_list_len
        self.run_dir: Optional[Path] = None
        self.steps_dir: Optional[Path] = None
        if log_dir is not None:
            self.run_dir = Path(log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.steps_dir = self.run_dir / "steps"
            self.steps_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def step(self, name: str) -> Iterator[Dict[str, Any]]:
        info: Dict[str, Any] = {}
        start = time.perf_counter()
        if self.verbose:
            print(f"[pipeline] {name}: starting...", flush=True)
        try:
            yield info
        finally:
            duration = time.perf_counter() - start
            self.steps.append({"name": name, "duration_seconds": duration, "info": info})
            if self.verbose:
                print(f"[pipeline] {name}: done in {duration:.1f}s", flush=True)
            self._flush(index=len(self.steps) - 1)

    @property
    def durations(self) -> Dict[str, float]:
        return {s["name"]: s["duration_seconds"] for s in self.steps}

    def _flush(self, index: int, result: Any = None) -> None:
        if self.run_dir is None or self.steps_dir is None:
            return

        step = self.steps[index]
        step_path = self.steps_dir / f"{index:02d}_{step['name']}.json"
        with open(step_path, "w") as f:
            json.dump(_to_jsonable(step), f, indent=2, default=str)

        total_duration = sum(s["duration_seconds"] for s in self.steps)
        steps_overview = []
        duration_ranking = []
        for s in self.steps:
            pct = (s["duration_seconds"] / total_duration * 100) if total_duration else 0.0
            duration_ranking.append({"name": s["name"], "duration_seconds": s["duration_seconds"], "pct_of_total": pct})
            steps_overview.append(
                {
                    "name": s["name"],
                    "duration_seconds": s["duration_seconds"],
                    "pct_of_total": pct,
                    "overview": _summarize(_to_jsonable(s["info"]), self.overview_max_list_len),
                }
            )

        # Two views of the same steps: `steps` in execution order with each
        # step's condensed content (what actually happened), and
        # `steps_by_duration_desc` ranked slowest-first with no content (just
        # "where did the time go" at a glance) -- plus `result`, the final
        # pipeline-wide comparison, all in the one file.
        log = {
            "config": _to_jsonable(self.config),
            "total_duration_seconds": total_duration,
            "steps": steps_overview,
            "steps_by_duration_desc": sorted(duration_ranking, key=lambda s: -s["duration_seconds"]),
            "result": _to_jsonable(result),
        }
        with open(self.run_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2, default=str)

    def finalize(self, result: Any) -> Optional[str]:
        """Rewrites log.json with the pipeline's final result folded in.
        Returns its path, or None if this RunLog was created without a
        log_dir (logging disabled)."""
        if self.run_dir is None:
            return None
        self._flush(index=len(self.steps) - 1, result=result)
        return str(self.run_dir / "log.json")
