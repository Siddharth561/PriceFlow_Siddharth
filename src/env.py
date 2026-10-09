
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

from .config import BASE_PRICE, GRID, SURGE_LEVELS
from .simulator import accept_prob, build_obs, demand_mean, supply_mean


class PricingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(len(SURGE_LEVELS))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32)
        self.hour = 0
        self.weather = 0
        self.event = False
        self.zx = 0
        self.zy = 0
        self.loyalty = 1.0

    def _observation(self) -> np.ndarray:
        return build_obs(self.zx, self.zy, self.hour % 24, self.weather, self.event, self.loyalty)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Start a deterministic, seeded zone-day episode."""
        super().reset(seed=seed)
        del options
        self.hour = 0
        self.weather = int(self.np_random.choice(3, p=[0.6, 0.2, 0.2]))
        self.event = bool(self.np_random.random() < 0.1)
        self.zx = int(self.np_random.integers(0, GRID))
        self.zy = int(self.np_random.integers(0, GRID))
        self.loyalty = 1.0
        return self._observation(), {}

    def _lam_cap(self, surge: float) -> tuple[float, float]:
        lam = demand_mean(self.zx, self.zy, self.hour, self.weather, self.event) * self.loyalty
        cap = supply_mean(self.zx, self.zy, self.hour, self.weather) * (1 + 0.4 * (surge - 1))
        return float(lam), float(cap)

    def expected_revenue(self, action: int) -> float:
        surge = float(SURGE_LEVELS[action])
        lam, cap = self._lam_cap(surge)
        probability = accept_prob(surge, self.weather, self.hour, self.event)
        return float(min(lam * probability, cap) * BASE_PRICE * surge)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one hourly surge decision and return the Gymnasium transition."""
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be in [0, {len(SURGE_LEVELS) - 1}], got {action}")
        surge = float(SURGE_LEVELS[action])
        lam, cap = self._lam_cap(surge)
        requests = int(self.np_random.poisson(lam))
        probability = float(accept_prob(surge, self.weather, self.hour, self.event))
        accepted = min(int(self.np_random.binomial(requests, probability)), int(cap))
        revenue = accepted * BASE_PRICE * surge
        reject_rate = 1 - accepted / requests if requests else 0.0
        self.loyalty = float(np.clip(self.loyalty - 0.05 * reject_rate + 0.02, 0.5, 1.0))
        self.hour += 1
        terminated = self.hour >= 24
        return self._observation(), revenue / 1000, terminated, False, {
            "revenue": revenue,
            "accepted": accepted,
        }


if __name__ == "__main__":
    check_env(PricingEnv())
    print("env OK")
