"""Per-run logging: how long each pipeline step took, and a JSON record of
the exact config + result for a run, so a slow run is self-diagnosing (no
need to re-instrument with ad-hoc timing prints, as happened while chasing
down the node2vec/graph-construction bottleneck) and every result stays
traceable back to exactly what config produced it.
"""

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


@contextmanager
def timed_step(durations: Dict[str, float], name: str):
    """Records `name`'s wall-clock duration (seconds) into `durations` on exit."""
    start = time.perf_counter()
    try:
        yield
    finally:
        durations[name] = time.perf_counter() - start


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def write_run_log(config: Any, durations: Dict[str, float], result: Any, log_dir: str) -> str:
    """Writes `log_dir/<timestamp>/log.json` with the config, per-step
    durations, and result for this run. Returns the written file's path."""
    run_dir = Path(log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "config": _to_jsonable(config),
        "step_durations_seconds": durations,
        "total_duration_seconds": sum(durations.values()),
        "result": _to_jsonable(result),
    }

    log_path = run_dir / "log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    return str(log_path)
