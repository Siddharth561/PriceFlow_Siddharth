from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    BASE_PRICE,
    EPISODE_HORIZON,
    LAT_MAX,
    LAT_MIN,
    LNG_MAX,
    LNG_MIN,
    MAX_BASE_PRICE,
    MAX_DEMAND,
    MAX_SUPPLY,
    SURGE_LEVELS,
    WEATHER_LABELS,
)


def observation_from_context(
    latitude: float,
    longitude: float,
    hour: int,
    day_of_week: int,
    weather: str,
    demand: float,
    supply: float,
    base_price: float,
) -> np.ndarray:
    hour_angle = 2.0 * math.pi * hour / 24.0
    weather_index = WEATHER_LABELS.index(weather)
    return np.asarray(
        [
            (latitude - LAT_MIN) / (LAT_MAX - LAT_MIN),
            (longitude - LNG_MIN) / (LNG_MAX - LNG_MIN),
            (math.sin(hour_angle) + 1.0) / 2.0,
            (math.cos(hour_angle) + 1.0) / 2.0,
            day_of_week / 6.0,
            weather_index / (len(WEATHER_LABELS) - 1),
            demand / MAX_DEMAND,
            supply / MAX_SUPPLY,
            base_price / MAX_BASE_PRICE,
        ],
        dtype=np.float32,
    )


def acceptance_probability_for_context(
    demand: float,
    supply: float,
    weather: str,
    surge_multiplier: float,
) -> float:
    demand_factor = float(np.clip(demand / 140.0, 0.2, 1.2))
    supply_factor = float(np.clip(1.25 / max(supply / 50.0, 0.3), 0.2, 1.2))
    weather_factor = 0.82 if weather == "rain" else 1.0
    surge_factor = math.exp(-0.9 * (surge_multiplier - 1.0))
    probability = 0.9 * surge_factor * demand_factor * supply_factor * weather_factor
    return float(np.clip(probability, 0.02, 0.96))


class DynamicPricingEnv(gym.Env[np.ndarray, int]):

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_horizon: int = EPISODE_HORIZON,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(episode_horizon, bool) or int(episode_horizon) != episode_horizon:
            raise ValueError("episode_horizon must be a positive integer.")
        if episode_horizon <= 0:
            raise ValueError("episode_horizon must be a positive integer.")

        self.episode_horizon = int(episode_horizon)
        self.action_space = spaces.Discrete(len(SURGE_LEVELS))
        # Two location values, two cyclic hour values, and day/weather/demand/
        # supply/base-price values. Every feature is normalized to [0, 1].
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.latitude = LAT_MIN
        self.longitude = LNG_MIN
        self.hour = 0
        self.day_of_week = 0
        self.weather_idx = 0
        self.weather = WEATHER_LABELS[0]
        self.demand = 0.0
        self.supply = 0.0
        self.base_price = float(BASE_PRICE)
        self._episode_over = False
        self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            raise ValueError("Reset options are not supported.")

        self.current_step = 0
        self._episode_over = False
        self._sample_trip()
        return self._make_observation(), self._state_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_over:
            raise RuntimeError("Episode is finished; call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the valid action space {self.action_space}.")

        surge_multiplier = float(SURGE_LEVELS[int(action)])
        acceptance_probability = self._acceptance_probability(surge_multiplier)
        accepted = bool(self.np_random.random() < acceptance_probability)
        revenue = float(self.base_price * surge_multiplier if accepted else 0.0)
        reward = revenue

        self.current_step += 1
        terminated = self.current_step >= self.episode_horizon
        truncated = False
        self._episode_over = terminated

        trip_info = {
            "surge_multiplier": surge_multiplier,
            "acceptance_probability": acceptance_probability,
            "accepted": accepted,
            "demand": float(self.demand),
            "supply": float(self.supply),
            "revenue": revenue,
            "current_step": self.current_step,
            "step": self.current_step,
            "weather": self.weather,
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "latitude": float(self.latitude),
            "longitude": float(self.longitude),
            "base_price": float(self.base_price),
        }

        if not terminated:
            self._sample_trip()
        observation = self._make_observation()
        return observation, reward, terminated, truncated, trip_info

    def _sample_trip(self) -> None:
        self.latitude = float(self.np_random.uniform(LAT_MIN, LAT_MAX))
        self.longitude = float(self.np_random.uniform(LNG_MIN, LNG_MAX))
        self.hour = int(self.np_random.integers(0, 24))
        self.day_of_week = int(self.np_random.integers(0, 7))
        self.weather_idx = int(self.np_random.choice(3, p=[0.6, 0.25, 0.15]))
        self.weather = WEATHER_LABELS[self.weather_idx]

        peak_factor = 0.0
        if 8 <= self.hour <= 10:
            peak_factor = 1.2
        elif 17 <= self.hour <= 21:
            peak_factor = 1.5

        is_rain = self.weather == "rain"
        self.demand = float(
            np.clip(
                60.0
                + self.np_random.normal(22.0, 14.0)
                + peak_factor * 26.0
                + (20.0 if is_rain else 0.0),
                15.0,
                MAX_DEMAND,
            )
        )
        self.supply = float(
            np.clip(
                38.0
                + self.np_random.normal(10.0, 6.0)
                - (13.0 if is_rain else 0.0)
                - (8.0 if self.hour < 6 else 0.0),
                6.0,
                MAX_SUPPLY,
            )
        )
        self.base_price = float(
            np.clip(
                80.0 + self.np_random.uniform(35.0, 170.0) + self.demand * 0.8,
                60.0,
                MAX_BASE_PRICE,
            )
        )

    def _make_observation(self) -> np.ndarray:
        return observation_from_context(
            latitude=self.latitude,
            longitude=self.longitude,
            hour=self.hour,
            day_of_week=self.day_of_week,
            weather=self.weather,
            demand=self.demand,
            supply=self.supply,
            base_price=self.base_price,
        )

    def _state_info(self) -> dict[str, int | float | str]:
        return {
            "latitude": float(self.latitude),
            "longitude": float(self.longitude),
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "weather": self.weather,
            "demand": float(self.demand),
            "supply": float(self.supply),
            "base_price": float(self.base_price),
            "current_step": self.current_step,
        }

    def _acceptance_probability(self, surge_multiplier: float) -> float:
        return acceptance_probability_for_context(
            demand=self.demand,
            supply=self.supply,
            weather=self.weather,
            surge_multiplier=surge_multiplier,
        )


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    check_env(DynamicPricingEnv())
    print("Environment validation passed.")
