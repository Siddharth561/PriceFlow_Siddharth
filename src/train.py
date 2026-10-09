
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from .env import PricingEnv


def main() -> None:
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    env = Monitor(PricingEnv(), filename="results/train_monitor")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.95,
        train_freq=4,
        target_update_interval=500,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,
        seed=42,
        device="cpu",
        verbose=1,
    )
    model.learn(total_timesteps=10_000 * 24)
    model.save("models/dqn_pricing")
    env.close()

    monitor = pd.read_csv("results/train_monitor.monitor.csv", skiprows=1)
    rolling = monitor["r"].rolling(100, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monitor["r"], color="steelblue", alpha=0.25, label="Episode reward")
    ax.plot(rolling, color="navy", linewidth=2, label="100-episode rolling mean")
    ax.set(xlabel="Episode", ylabel="Reward (thousands of INR)", title="DQN training reward")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("results/reward_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved model and reward curve from {len(monitor)} episodes.")


if __name__ == "__main__":
    main()
