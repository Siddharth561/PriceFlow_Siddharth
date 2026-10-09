
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import LAT_MAX, LAT_MIN, LNG_MAX, LNG_MIN, SURGE_LEVELS
from .simulator import accept_prob, demand_mean, supply_mean


def main() -> None:
    rng = np.random.default_rng(7)
    count = 500_000
    lat = np.round(rng.uniform(LAT_MIN, LAT_MAX, count), 5)
    lng = np.round(rng.uniform(LNG_MIN, LNG_MAX, count), 5)
    zone_x = np.clip(((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * 5).astype(int), 0, 4)
    zone_y = np.clip(((lng - LNG_MIN) / (LNG_MAX - LNG_MIN) * 5).astype(int), 0, 4)
    hour = rng.integers(0, 24, count)
    weather = rng.choice(3, count, p=[0.6, 0.2, 0.2])
    event = rng.binomial(1, 0.1, count)
    day = rng.integers(0, 60, count)
    minute = rng.integers(0, 60, count)
    timestamp = (
        np.datetime64("2026-01-01")
        + day.astype("timedelta64[D]")
        + hour.astype("timedelta64[h]")
        + minute.astype("timedelta64[m]")
    )
    demand_level = rng.poisson(demand_mean(zone_x, zone_y, hour, weather, event))
    supply_level = rng.poisson(supply_mean(zone_x, zone_y, hour, weather))
    trip_km = rng.uniform(1, 20, count)
    base_price = 50 + 10 * trip_km
    surge = rng.choice(SURGE_LEVELS, count)
    final_price = base_price * surge
    probability = accept_prob(surge, weather, hour, event)
    accepted = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "lat": lat,
            "lng": lng,
            "zone_x": zone_x,
            "zone_y": zone_y,
            "hour": hour,
            "weather": weather,
            "event": event,
            "demand_level": demand_level,
            "supply_level": supply_level,
            "trip_km": trip_km,
            "base_price": base_price,
            "surge_multiplier": surge,
            "final_price": final_price,
            "accepted": accepted,
        }
    )
    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    frame.to_csv("data/rides_trips.csv", index=False)
    print(f"Generated {frame.shape}; acceptance rate: {frame['accepted'].mean():.4f}")

    surges = np.asarray(SURGE_LEVELS)
    fig, ax = plt.subplots(figsize=(8, 5))
    for weather_id, label in enumerate(("Clear", "Cloudy", "Rain")):
        means = accept_prob(
            surges, weather_id, np.full(surges.shape, 12), np.zeros(surges.shape)
        )
        ax.plot(surges, means, marker="o", label=label)
    ax.set(xlabel="Surge multiplier", ylabel="Mean acceptance probability", title="Acceptance vs. surge by weather")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("results/acceptance_vs_surge.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
