from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from config import (
    DEFAULT_MODEL_PATH,
    LAT_MAX,
    LAT_MIN,
    LNG_MAX,
    LNG_MIN,
    MAX_DEMAND,
    MAX_SUPPLY,
    RESULTS_DIR,
    SURGE_LEVELS,
    WEATHER_INDEX,
    WEATHER_LABELS,
)
from environment.pricing_env import (
    DynamicPricingEnv,
    acceptance_probability_for_context,
    observation_from_context,
)

BASELINE_RAIN_MULTIPLIER = 1.5
BASELINE_OTHER_WEATHER_MULTIPLIER = 1.0
REVENUE_LIFT_TARGET_PERCENT = 10.0


def _weekend_peak_factor(hours: np.ndarray) -> np.ndarray:
    factors = np.zeros(hours.shape, dtype=np.float64)
    factors[(hours >= 8) & (hours <= 10)] = 1.2
    factors[(hours >= 17) & (hours <= 21)] = 1.5
    return factors


def generate_scenarios(scenario_count: int = 250, seed: int = 42) -> pd.DataFrame:
    if scenario_count <= 0:
        raise ValueError("scenario_count must be a positive integer.")

    rng = np.random.default_rng(seed)
    latitudes = rng.uniform(LAT_MIN, LAT_MAX, size=scenario_count)
    longitudes = rng.uniform(LNG_MIN, LNG_MAX, size=scenario_count)
    hours = rng.integers(0, 24, size=scenario_count)
    days = rng.integers(0, 7, size=scenario_count)
    weather_codes = rng.choice(3, size=scenario_count, p=[0.6, 0.25, 0.15])
    weather = np.asarray(WEATHER_LABELS)[weather_codes]

    demand = np.clip(
        60.0
        + rng.normal(22.0, 14.0, size=scenario_count)
        + _weekend_peak_factor(hours) * 26.0
        + np.where(weather_codes == WEATHER_INDEX["rain"], 20.0, 0.0),
        15.0,
        MAX_DEMAND,
    )
    supply = np.clip(
        38.0
        + rng.normal(10.0, 6.0, size=scenario_count)
        - np.where(weather_codes == WEATHER_INDEX["rain"], 13.0, 0.0)
        - np.where(hours < 6, 8.0, 0.0),
        6.0,
        MAX_SUPPLY,
    )
    base_price = np.clip(
        80.0 + rng.uniform(35.0, 170.0, size=scenario_count) + demand * 0.8,
        60.0,
        500.0,
    )

    return pd.DataFrame(
        {
            "scenario": np.arange(1, scenario_count + 1),
            "lat": latitudes,
            "lng": longitudes,
            "hour": hours,
            "day_of_week": days,
            "weather": weather,
            "demand": demand,
            "supply": supply,
            "base_price": base_price,
        }
    )


def baseline_policy(row: pd.Series) -> float:
    return (
        BASELINE_RAIN_MULTIPLIER
        if row["weather"] == "rain"
        else BASELINE_OTHER_WEATHER_MULTIPLIER
    )


def _acceptance_probability(row: pd.Series, multiplier: float) -> float:
    return acceptance_probability_for_context(
        demand=float(row["demand"]),
        supply=float(row["supply"]),
        weather=str(row["weather"]),
        surge_multiplier=multiplier,
    )


def _observation_for_scenario(row: pd.Series) -> np.ndarray:
    return observation_from_context(
        latitude=float(row["lat"]),
        longitude=float(row["lng"]),
        hour=int(row["hour"]),
        day_of_week=int(row["day_of_week"]),
        weather=str(row["weather"]),
        demand=float(row["demand"]),
        supply=float(row["supply"]),
        base_price=float(row["base_price"]),
    )


def _load_model(model_path: str | Path) -> tuple[DQN, DynamicPricingEnv]:
    saved_path = Path(model_path)
    if not saved_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {saved_path}. Train it first with "
            r".\venv\Scripts\python.exe -m training.train_agent --full-target --seed 42"
        )
    env = DynamicPricingEnv()
    try:
        model = DQN.load(str(saved_path), env=env, device="cpu")
    except Exception:
        env.close()
        raise
    if model.action_space.n != len(SURGE_LEVELS):
        env.close()
        raise ValueError(
            f"Model action count {model.action_space.n} does not match "
            f"the configured {len(SURGE_LEVELS)} surge levels."
        )
    if model.observation_space.shape != env.observation_space.shape:
        env.close()
        raise ValueError(
            "Model observation shape does not match DynamicPricingEnv: "
            f"{model.observation_space.shape} != {env.observation_space.shape}."
        )
    return model, env


def _policy_predictions(
    model: DQN,
    scenarios: pd.DataFrame,
    acceptance_draws: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    action_to_multiplier = np.asarray(SURGE_LEVELS, dtype=np.float64)
    for index, (_, row) in enumerate(scenarios.iterrows()):
        observation = _observation_for_scenario(row)
        action, _ = model.predict(observation, deterministic=True)
        action_index = int(action)
        if not 0 <= action_index < len(action_to_multiplier):
            raise ValueError(f"DQN returned an invalid action index: {action_index}.")
        multiplier = float(action_to_multiplier[action_index])
        probability = _acceptance_probability(row, multiplier)
        accepted = bool(acceptance_draws[index] < probability)
        records.append(
            {
                "surge_multiplier": multiplier,
                "acceptance_probability": probability,
                "accepted": accepted,
                "revenue": float(row["base_price"]) * multiplier if accepted else 0.0,
            }
        )
    return pd.DataFrame(records, index=scenarios.index)


def _policy_outcomes(
    scenarios: pd.DataFrame,
    multipliers: np.ndarray,
    acceptance_draws: np.ndarray,
) -> pd.DataFrame:
    """Simulate outcomes for a policy using the scenarios' shared random draws."""
    probabilities = np.asarray(
        [
            _acceptance_probability(row, float(multiplier))
            for (_, row), multiplier in zip(scenarios.iterrows(), multipliers)
        ],
        dtype=np.float64,
    )
    accepted = acceptance_draws < probabilities
    revenues = np.where(
        accepted,
        scenarios["base_price"].to_numpy(dtype=np.float64) * multipliers,
        0.0,
    )
    return pd.DataFrame(
        {
            "surge_multiplier": multipliers,
            "acceptance_probability": probabilities,
            "accepted": accepted,
            "revenue": revenues,
        },
        index=scenarios.index,
    )


def _expected_regret(
    scenarios: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> np.ndarray:

    benchmark_expected_revenue = np.zeros(len(scenarios), dtype=np.float64)
    for position, (_, row) in enumerate(scenarios.iterrows()):
        benchmark_expected_revenue[position] = max(
            _acceptance_probability(row, float(multiplier))
            * float(row["base_price"])
            * float(multiplier)
            for multiplier in SURGE_LEVELS
        )
    policy_expected_revenue = (
        outcomes["acceptance_probability"].to_numpy(dtype=np.float64)
        * scenarios["base_price"].to_numpy(dtype=np.float64)
        * outcomes["surge_multiplier"].to_numpy(dtype=np.float64)
    )
    return np.maximum(benchmark_expected_revenue - policy_expected_revenue, 0.0)


def _summarize_policy(
    outcomes: pd.DataFrame,
    regret: np.ndarray,
) -> dict[str, float]:
    """Summarize realized outcomes and expected per-context regret."""
    trip_count = len(outcomes)
    return {
        "total_revenue": float(outcomes["revenue"].sum()),
        "mean_revenue_per_trip": float(outcomes["revenue"].mean()) if trip_count else 0.0,
        "acceptance_rate": float(outcomes["accepted"].mean()) if trip_count else 0.0,
        "average_surge_multiplier": float(outcomes["surge_multiplier"].mean()) if trip_count else 0.0,
        "cumulative_policy_regret": float(regret.sum()),
        "mean_policy_regret_per_trip": float(regret.mean()) if trip_count else 0.0,
    }


def compare_policies(
    baseline_results: pd.DataFrame,
    rl_results: pd.DataFrame,
) -> dict[str, float | bool | None]:
    """Summarize two aligned outcome tables for existing evaluation callers."""
    if len(baseline_results) != len(rl_results):
        raise ValueError("Policy results must contain the same number of scenarios.")

    baseline_revenue = float(baseline_results["revenue"].sum())
    rl_revenue = float(rl_results["revenue"].sum())
    revenue_lift = (
        (rl_revenue - baseline_revenue) / baseline_revenue * 100.0
        if baseline_revenue > 0.0
        else None
    )
    baseline_multiplier_column = (
        "surge_multiplier" if "surge_multiplier" in baseline_results else "multiplier"
    )
    rl_multiplier_column = "surge_multiplier" if "surge_multiplier" in rl_results else "multiplier"

    return {
        "baseline_revenue": baseline_revenue,
        "rl_revenue": rl_revenue,
        "baseline_mean_revenue_per_trip": (
            float(baseline_results["revenue"].mean()) if len(baseline_results) else 0.0
        ),
        "rl_mean_revenue_per_trip": (
            float(rl_results["revenue"].mean()) if len(rl_results) else 0.0
        ),
        "baseline_acceptance_rate": (
            float(baseline_results["accepted"].mean()) if len(baseline_results) else 0.0
        ),
        "rl_acceptance_rate": (
            float(rl_results["accepted"].mean()) if len(rl_results) else 0.0
        ),
        "baseline_average_surge_multiplier": (
            float(baseline_results[baseline_multiplier_column].mean())
            if len(baseline_results)
            else 0.0
        ),
        "rl_average_surge_multiplier": (
            float(rl_results[rl_multiplier_column].mean()) if len(rl_results) else 0.0
        ),
        "revenue_lift_percent": revenue_lift,
        "revenue_lift_target_met": (
            revenue_lift is not None and revenue_lift >= REVENUE_LIFT_TARGET_PERCENT
        ),
    }


def _write_plots(
    baseline_outcomes: pd.DataFrame,
    rl_outcomes: pd.DataFrame,
    baseline_regret: np.ndarray,
    rl_regret: np.ndarray,
    output_dir: Path,
) -> dict[str, str]:
    scenario_numbers = np.arange(1, len(baseline_outcomes) + 1)
    paths = {
        "cumulative_revenue_plot": output_dir / "cumulative_revenue.png",
        "reward_plot": output_dir / "reward_comparison.png",
        "policy_regret_plot": output_dir / "policy_regret.png",
    }

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        scenario_numbers,
        baseline_outcomes["revenue"].cumsum(),
        label="Baseline",
        color="darkorange",
    )
    axis.plot(
        scenario_numbers,
        rl_outcomes["revenue"].cumsum(),
        label="DQN",
        color="royalblue",
    )
    axis.set_title("Cumulative realized revenue")
    axis.set_xlabel("Simulated trip")
    axis.set_ylabel("Revenue (INR)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["cumulative_revenue_plot"], dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    window = min(20, len(baseline_outcomes))
    if window:
        baseline_rewards = np.convolve(
            baseline_outcomes["revenue"].to_numpy(dtype=np.float64),
            np.ones(window, dtype=np.float64) / window,
            mode="valid",
        )
        rl_rewards = np.convolve(
            rl_outcomes["revenue"].to_numpy(dtype=np.float64),
            np.ones(window, dtype=np.float64) / window,
            mode="valid",
        )
        reward_scenarios = np.arange(window, len(scenario_numbers) + 1)
        axis.plot(
            reward_scenarios,
            baseline_rewards,
            label=f"Baseline ({window}-trip moving mean)",
            color="darkorange",
        )
        axis.plot(
            reward_scenarios,
            rl_rewards,
            label=f"DQN ({window}-trip moving mean)",
            color="royalblue",
        )
    axis.set_title("Realized reward per simulated trip")
    axis.set_xlabel("Simulated trip")
    axis.set_ylabel("Revenue reward (INR)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["reward_plot"], dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(scenario_numbers, np.cumsum(baseline_regret), label="Baseline", color="darkorange")
    axis.plot(scenario_numbers, np.cumsum(rl_regret), label="DQN", color="royalblue")
    axis.set_title("Cumulative contextual expected-revenue regret")
    axis.set_xlabel("Simulated trip")
    axis.set_ylabel("Regret (INR)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["policy_regret_plot"], dpi=150)
    plt.close(figure)
    return {key: str(path) for key, path in paths.items()}


def run_ab_test(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scenario_count: int = 250,
    seed: int = 42,
    output_dir: str | Path = RESULTS_DIR,
) -> dict[str, Any]:
    if scenario_count <= 0:
        raise ValueError("scenario_count must be a positive integer.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scenarios = generate_scenarios(scenario_count=scenario_count, seed=seed)
    acceptance_draws = np.random.default_rng(seed + 1).random(scenario_count)
    model, env = _load_model(model_path)
    try:
        baseline_multipliers = scenarios.apply(baseline_policy, axis=1).to_numpy(dtype=np.float64)
        baseline_outcomes = _policy_outcomes(scenarios, baseline_multipliers, acceptance_draws)
        rl_outcomes = _policy_predictions(model, scenarios, acceptance_draws)

        baseline_regret = _expected_regret(scenarios, baseline_outcomes)
        rl_regret = _expected_regret(scenarios, rl_outcomes)
    finally:
        env.close()

    baseline_metrics = _summarize_policy(baseline_outcomes, baseline_regret)
    rl_metrics = _summarize_policy(rl_outcomes, rl_regret)
    baseline_revenue = baseline_metrics["total_revenue"]
    rl_revenue = rl_metrics["total_revenue"]
    revenue_lift_percent = (
        (rl_revenue - baseline_revenue) / baseline_revenue * 100.0
        if baseline_revenue > 0.0
        else None
    )
    target_met = revenue_lift_percent is not None and revenue_lift_percent >= REVENUE_LIFT_TARGET_PERCENT

    metrics: dict[str, Any] = {
        "scenario_count": scenario_count,
        "seed": seed,
        "baseline_policy": "1.5x during rain; 1.0x during clear/cloudy weather",
        "regret_benchmark": (
            "Per-context maximum expected revenue among the nine configured surge actions "
            "under the simulation acceptance model; contextual reference only, not claimed "
            "to be a globally optimal policy."
        ),
        "revenue_lift_target_percent": REVENUE_LIFT_TARGET_PERCENT,
        "baseline": baseline_metrics,
        "dqn": rl_metrics,
        "revenue_lift_percent": revenue_lift_percent,
        "revenue_lift_target_met": target_met,
        "revenue_lift_target_status": (
            "DQN revenue improvement exceeds the 10% target."
            if target_met
            else "DQN revenue improvement does not exceed the 10% target."
            if revenue_lift_percent is not None
            else "Revenue lift is undefined because baseline revenue is zero; the 10% target was not met."
        ),
        "baseline_cumulative_revenue": baseline_outcomes["revenue"].cumsum().tolist(),
        "dqn_cumulative_revenue": rl_outcomes["revenue"].cumsum().tolist(),
        **_write_plots(baseline_outcomes, rl_outcomes, baseline_regret, rl_regret, output_path),
    }

    scenario_results = scenarios.copy()
    for policy_name, outcomes, regrets in (
        ("baseline", baseline_outcomes, baseline_regret),
        ("dqn", rl_outcomes, rl_regret),
    ):
        for metric_name in (
            "surge_multiplier",
            "acceptance_probability",
            "accepted",
            "revenue",
        ):
            scenario_results[f"{policy_name}_{metric_name}"] = outcomes[metric_name].to_numpy()
        scenario_results[f"{policy_name}_expected_regret"] = regrets
    scenario_results_path = output_path / "ab_test_scenarios.csv"
    metrics_json_path = output_path / "ab_test_metrics.json"
    metrics_csv_path = output_path / "ab_test_metrics.csv"
    metrics["scenario_results_path"] = str(scenario_results_path)
    metrics["metrics_json_path"] = str(metrics_json_path)
    metrics["metrics_csv_path"] = str(metrics_csv_path)
    scenario_results.to_csv(scenario_results_path, index=False)
    metrics_json_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.json_normalize(metrics).to_csv(metrics_csv_path, index=False)
    print(
        json.dumps(
            {
                "scenario_count": scenario_count,
                "seed": seed,
                "baseline": baseline_metrics,
                "dqn": rl_metrics,
                "revenue_lift_percent": revenue_lift_percent,
                "revenue_lift_target_met": target_met,
                "revenue_lift_target_status": metrics["revenue_lift_target_status"],
                "artifacts": {
                    "metrics_json": str(metrics_json_path),
                    "metrics_csv": str(metrics_csv_path),
                    "scenario_csv": str(scenario_results_path),
                    "cumulative_revenue_plot": metrics["cumulative_revenue_plot"],
                    "reward_plot": metrics["reward_plot"],
                    "policy_regret_plot": metrics["policy_regret_plot"],
                },
            },
            indent=2,
            allow_nan=False,
        )
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the trained PriceFlow DQN with a weather-rule baseline."
    )
    parser.add_argument(
        "--scenario-count",
        type=int,
        default=250,
        help="Number of identical contextual trips used for both policies.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for scenarios and matched rider outcomes.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Trained DQN zip path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory for machine-readable metrics and charts (default: {RESULTS_DIR}).",
    )
    args = parser.parse_args()
    run_ab_test(
        model_path=args.model,
        scenario_count=args.scenario_count,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
