
from typing import Any

import numpy as np

from .config import GRID, LAT_MAX, LAT_MIN, LNG_MAX, LNG_MIN, MAX_DEMAND, MAX_SUPPLY, PEAK_HOURS

_rng = np.random.default_rng(42)
ZONE_WEIGHT = _rng.uniform(0.6, 1.4, size=(GRID, GRID))
ZONE_WEIGHT[2, 2] = 2.0
ZONE_WEIGHT[0, 1] = 1.8
ZONE_WEIGHT[3, 3] = 1.7
_WEATHER_DEMAND = np.array([1.0, 1.1, 1.6])
_WEATHER_SUPPLY = np.array([1.0, 0.95, 0.6])
_WEATHER_SENSITIVITY = np.array([1.0, 0.85, 0.5])


def _return_scalar_if_scalar(value: np.ndarray, *inputs: Any) -> Any:
    if all(np.ndim(item) == 0 for item in inputs):
        return value.item()
    return value


def latlng_to_zone(lat: float, lng: float) -> tuple[int, int]:
    """Map Pune coordinates into a clipped grid cell."""
    zx = int(np.clip((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID, 0, GRID - 1))
    zy = int(np.clip((lng - LNG_MIN) / (LNG_MAX - LNG_MIN) * GRID, 0, GRID - 1))
    return zx, zy


def hour_factor(h: Any) -> Any:
    hours = np.asarray(h)
    result = 1 + 1.2 * np.exp(-((hours - 9) ** 2) / 4) + 1.5 * np.exp(-((hours - 18.5) ** 2) / 5)
    return _return_scalar_if_scalar(result, h)


def demand_mean(zx: Any, zy: Any, hour: Any, weather: Any, event: Any) -> Any:
    result = (
        20
        * ZONE_WEIGHT[np.asarray(zx), np.asarray(zy)]
        * hour_factor(hour)
        * _WEATHER_DEMAND[np.asarray(weather)]
        * (1 + 0.5 * np.asarray(event))
    )
    return _return_scalar_if_scalar(np.asarray(result), zx, zy, hour, weather, event)


def supply_mean(zx: Any, zy: Any, hour: Any, weather: Any) -> Any:
    del hour
    result = 25 * ZONE_WEIGHT[np.asarray(zx), np.asarray(zy)] * _WEATHER_SUPPLY[np.asarray(weather)]
    return _return_scalar_if_scalar(np.asarray(result), zx, zy, weather)


def accept_prob(surge: Any, weather: Any, hour: Any, event: Any) -> Any:
    surge_array = np.asarray(surge)
    weather_array = np.asarray(weather)
    hour_array = np.asarray(hour)
    event_array = np.asarray(event)
    sensitivity = 0.9 * _WEATHER_SENSITIVITY[weather_array]
    sensitivity = sensitivity * np.where(np.isin(hour_array, PEAK_HOURS), 0.75, 1.0)
    sensitivity = sensitivity * np.where(event_array == 1, 0.8, 1.0)
    result = np.clip(0.95 * np.exp(-sensitivity * (surge_array - 1)), 0.02, 0.95)
    return _return_scalar_if_scalar(result, surge, weather, hour, event)


def build_obs(
    zx: int,
    zy: int,
    hour: int,
    weather: int,
    event: bool,
    loyalty: float = 1.0,
) -> np.ndarray:
    theta = 2 * np.pi * hour / 24
    demand = demand_mean(zx, zy, hour, weather, event) * loyalty
    supply = supply_mean(zx, zy, hour, weather)
    return np.array(
        [
            zx / (GRID - 1),
            zy / (GRID - 1),
            np.sin(theta) * 0.5 + 0.5,
            np.cos(theta) * 0.5 + 0.5,
            weather / 2,
            float(event),
            min(demand / MAX_DEMAND, 1.0),
            min(supply / MAX_SUPPLY, 1.0),
            loyalty,
        ],
        dtype=np.float32,
    )
