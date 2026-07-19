from .flops_timing import (
    FlopsParamsReport,
    InferenceBenchmark,
    TimingConfig,
    count_ops_and_params,
    measure_inference,
)
from .metrics import compute_metrics

# compare.py is deliberately not re-exported here: it depends on src.train
# (to run recalibration), and src.train depends on this package's own
# compute_metrics -- eagerly importing compare.py from this __init__ would
# make that a circular import. Import it directly: `from src.eval.compare import ...`.

__all__ = [
    "compute_metrics",
    "FlopsParamsReport",
    "InferenceBenchmark",
    "TimingConfig",
    "count_ops_and_params",
    "measure_inference",
]
