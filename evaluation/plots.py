from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_cumulative_revenue(baseline_revenue: list[float], rl_revenue: list[float], output_path: str | Path) -> None:
    cumulative_baseline = pd.Series(baseline_revenue).cumsum()
    cumulative_rl = pd.Series(rl_revenue).cumsum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cumulative_baseline, label="Baseline", color="darkorange")
    ax.plot(cumulative_rl, label="RL", color="royalblue")
    ax.set_title("Cumulative revenue comparison")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Revenue (INR)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_regret(regret: list[float], output_path: str | Path) -> None:
    cumulative_regret = pd.Series(regret).cumsum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cumulative_regret, color="crimson")
    ax.set_title("Cumulative policy regret")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Regret (INR)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_metrics(metrics: dict[str, float | int | str], csv_path: str | Path, json_path: str | Path) -> None:
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
