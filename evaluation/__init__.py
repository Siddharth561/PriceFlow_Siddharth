
from importlib import import_module
from typing import Any

__all__ = [
    "compare_policies",
    "export_metrics",
    "generate_scenarios",
    "plot_cumulative_revenue",
    "plot_regret",
    "run_ab_test",
]


def __getattr__(name: str) -> Any:
    if name in {"compare_policies", "generate_scenarios", "run_ab_test"}:
        return getattr(import_module(".run_ab_test", __name__), name)
    if name in {"export_metrics", "plot_cumulative_revenue", "plot_regret"}:
        return getattr(import_module(".plots", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
