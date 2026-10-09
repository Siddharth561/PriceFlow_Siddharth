from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, LAT_MAX, LAT_MIN, LNG_MAX, LNG_MIN, SURGE_LEVELS


def _weekend_peak_factor(hour: np.ndarray) -> np.ndarray:
    peak = np.zeros_like(hour, dtype=float)
    peak[(hour >= 8) & (hour <= 10)] = 1.2
    peak[(hour >= 17) & (hour <= 21)] = 1.5
    return peak


def generate_dataset(record_count: int = 500_000, output_path: str | Path = DATA_DIR / "rides_trips.csv", seed: int = 42) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Generate a reproducible synthetic ride dataset for Pune market simulation."""
    rng = np.random.default_rng(seed)
    n = int(record_count)

    latitudes = np.round(rng.uniform(LAT_MIN, LAT_MAX, size=n), 5)
    longitudes = np.round(rng.uniform(LNG_MIN, LNG_MAX, size=n), 5)
    hours = rng.integers(0, 24, size=n)
    day_of_week = rng.integers(0, 7, size=n)
    weather_codes = rng.choice(3, size=n, p=[0.6, 0.25, 0.15])
    weather_labels = np.array(["clear", "cloudy", "rain"])[weather_codes]
    demand = np.clip(
        60.0
        + rng.normal(22.0, 14.0, size=n)
        + _weekend_peak_factor(hours) * 26.0
        + np.where(weather_codes == 2, 20.0, 0.0),
        15.0,
        220.0,
    )
    supply = np.clip(
        38.0
        + rng.normal(10.0, 6.0, size=n)
        - np.where(weather_codes == 2, 13.0, 0.0)
        - np.where(hours < 6, 8.0, 0.0),
        6.0,
        110.0,
    )
    base_price = np.round(
        np.clip(80.0 + rng.uniform(35.0, 170.0, size=n) + demand * 0.8, 60.0, 500.0),
        2,
    )
    surge_multiplier = rng.choice(SURGE_LEVELS, size=n)
    acceptance_probability = np.clip(
        0.9
        * np.exp(-0.9 * (surge_multiplier - 1.0))
        * np.clip(demand / 140.0, 0.2, 1.2)
        * np.clip(1.25 / np.maximum(supply / 50.0, 0.3), 0.2, 1.2)
        * np.where(weather_labels == "rain", 0.82, 1.0),
        0.02,
        0.96,
    )
    accepted = rng.binomial(1, acceptance_probability)
    revenue = np.round(accepted * base_price * surge_multiplier, 2)

    frame = pd.DataFrame(
        {
            "trip_id": np.arange(1, n + 1),
            "latitude": latitudes,
            "longitude": longitudes,
            "hour": hours,
            "day_of_week": day_of_week,
            "weather": weather_labels,
            "demand": demand.round(2),
            "supply": supply.round(2),
            "base_price": base_price,
            "surge_multiplier": surge_multiplier.astype(float),
            "acceptance_probability": acceptance_probability.round(6),
            "accepted": accepted.astype(int),
            "revenue": revenue,
        }
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    summary = {
        "record_count": int(len(frame)),
        "acceptance_rate": float(frame["accepted"].mean()),
        "total_revenue": float(frame["revenue"].sum()),
        "output_path": str(output),
    }
    print(f"Generated {summary['record_count']} trips | acceptance rate: {summary['acceptance_rate']:.4f} | total revenue: {summary['total_revenue']:.2f} INR | output: {summary['output_path']}")
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Pune trip data for PriceFlow.")
    parser.add_argument("--count", type=int, default=500_000, help="Number of synthetic trip records to generate.")
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "rides_trips.csv"), help="Destination CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    generate_dataset(record_count=args.count, output_path=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
