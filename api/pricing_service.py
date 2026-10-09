from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from api.schemas import PriceRequest, PriceResponse
from config import DEFAULT_MODEL_PATH, SURGE_LEVELS
from environment.pricing_env import DynamicPricingEnv, observation_from_context


class PricingService:

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.model: DQN | None = None

    def load_model(self) -> DQN:
        """Load and validate the model, retaining its compatible environment."""
        if self.model is not None:
            return self.model
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Trained pricing model not found at {self.model_path}.")

        env = DynamicPricingEnv()
        try:
            model = DQN.load(str(self.model_path), env=env, device="cpu")
            if model.observation_space.shape != env.observation_space.shape:
                raise ValueError(
                    "Saved model observation shape does not match the pricing environment."
                )
            if model.action_space.n != len(SURGE_LEVELS):
                raise ValueError(
                    "Saved model action count does not match the configured surge levels."
                )
        except Exception:
            env.close()
            raise
        self.model = model
        return model

    def close(self) -> None:
        if self.model is not None:
            model_env = self.model.get_env()
            if model_env is not None:
                model_env.close()
            self.model = None

    def _observation_for_request(self, request: PriceRequest) -> np.ndarray:
        """Use exactly the normalized feature layout defined by DynamicPricingEnv."""
        return observation_from_context(
            latitude=request.lat,
            longitude=request.lng,
            hour=request.hour,
            day_of_week=request.day_of_week,
            weather=request.weather,
            demand=request.demand,
            supply=request.supply,
            base_price=request.base_price,
        )

    @staticmethod
    def _acceptance_probability(request: PriceRequest, multiplier: float) -> float:
        """Estimate acceptance with the environment's configured response curve."""
        demand_factor = float(np.clip(request.demand / 140.0, 0.2, 1.2))
        supply_factor = float(np.clip(1.25 / max(request.supply / 50.0, 0.3), 0.2, 1.2))
        weather_factor = 0.82 if request.weather == "rain" else 1.0
        surge_factor = math.exp(-0.9 * (multiplier - 1.0))
        probability = 0.9 * surge_factor * demand_factor * supply_factor * weather_factor
        return float(np.clip(probability, 0.02, 0.96))

    def predict(self, request: PriceRequest) -> PriceResponse:
        if self.model is None:
            raise RuntimeError("Pricing model is not loaded.")

        observation = self._observation_for_request(request)
        started_at = time.perf_counter()
        action, _ = self.model.predict(observation, deterministic=True)
        inference_latency_ms = (time.perf_counter() - started_at) * 1000.0

        action_index = int(action)
        if not 0 <= action_index < len(SURGE_LEVELS):
            raise RuntimeError(f"Loaded DQN returned invalid action index {action_index}.")
        multiplier = float(SURGE_LEVELS[action_index])
        return PriceResponse(
            surge_multiplier=multiplier,
            base_price=float(request.base_price),
            final_price=float(request.base_price * multiplier),
            estimated_acceptance_probability=self._acceptance_probability(request, multiplier),
            model_identifier=self.model_path.name,
            metadata={
                "weather": request.weather,
                "hour": request.hour,
                "day_of_week": request.day_of_week,
                "demand": request.demand,
                "supply": request.supply,
            },
            inference_latency_ms=float(inference_latency_ms),
        )


pricing_service = PricingService()


def get_pricing_service() -> PricingService:
    """Return the process-wide pricing service instance."""
    return pricing_service
