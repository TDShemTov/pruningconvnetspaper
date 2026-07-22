"""Per-run logging: one JSON file per pipeline step (time + whatever
step-specific metrics that step attaches -- training epoch history,
activation-matrix shape, graph size, embedding-method config, pruning
results, ...) plus one top-level log.json tying them together and ranking
steps by duration.

The point is diagnosing *where a run's time actually goes* without
re-instrumenting anything ad hoc (as happened while chasing down the
node2vec/graph-construction bottleneck) -- "what's slow" is answered by
log.json's `steps_by_duration_desc` alone, and "what exactly happened in
step X" by that step's own small file, not a search through one big blob.
"""

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class RunLog:
    """Accumulates one entry per pipeline step, in the order they run.

    Usage: `with run_log.step("name") as info:` times the block; the block
    writes whatever metrics it wants into `info` (dicts, dataclasses, arrays
    -- anything `_to_jsonable` can walk) and they're captured alongside the
    duration.
    """

    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []

    @contextmanager
    def step(self, name: str) -> Iterator[Dict[str, Any]]:
        info: Dict[str, Any] = {}
        start = time.perf_counter()
        try:
            yield info
        finally:
            self.steps.append(
                {"name": name, "duration_seconds": time.perf_counter() - start, "info": info}
            )

    @property
    def durations(self) -> Dict[str, float]:
        return {s["name"]: s["duration_seconds"] for s in self.steps}


def write_run_log(config: Any, run_log: RunLog, result: Any, log_dir: str) -> str:
    """Writes `log_dir/<timestamp>/log.json` (config, total duration, steps
    ranked slowest-first, and the result) plus one
    `log_dir/<timestamp>/steps/<NN>_<name>.json` per step. Returns the
    top-level log.json path.
    """
    run_dir = Path(log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    steps_dir = run_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    total_duration = sum(s["duration_seconds"] for s in run_log.steps)

    step_summaries = []
    for i, step in enumerate(run_log.steps):
        step_summaries.append(
            {
                "name": step["name"],
                "duration_seconds": step["duration_seconds"],
                "pct_of_total": (step["duration_seconds"] / total_duration * 100) if total_duration else 0.0,
            }
        )
        step_path = steps_dir / f"{i:02d}_{step['name']}.json"
        with open(step_path, "w") as f:
            json.dump(_to_jsonable(step), f, indent=2, default=str)

    log = {
        "config": _to_jsonable(config),
        "total_duration_seconds": total_duration,
        "steps_by_duration_desc": sorted(step_summaries, key=lambda s: -s["duration_seconds"]),
        "step_order": [s["name"] for s in step_summaries],
        "result": _to_jsonable(result),
    }

    log_path = run_dir / "log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    return str(log_path)
