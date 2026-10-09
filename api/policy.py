

import torch
from stable_baselines3 import DQN

from src.config import SURGE_LEVELS, WEATHER
from src.simulator import build_obs, latlng_to_zone

torch.set_num_threads(1)
_model: DQN | None = None


def load(path: str = "models/dqn_pricing") -> None:
    global _model
    if _model is None:
        _model = DQN.load(path, device="cpu")


def get_surge(lat: float, lng: float, hour: int, weather: str, event: bool) -> float:
    if _model is None:
        raise RuntimeError("Pricing model has not been loaded")
    zx, zy = latlng_to_zone(lat, lng)
    observation = build_obs(zx, zy, hour, WEATHER[weather], event)
    action, _ = _model.predict(observation, deterministic=True)
    return float(SURGE_LEVELS[int(action)])
