from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from config import DEFAULT_MODEL_PATH, RESULTS_DIR, TRAINING_EPISODES_TARGET
from environment.pricing_env import DynamicPricingEnv

LEARNING_RATE = 5e-4
BUFFER_SIZE = 50_000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_STARTS = 2_000
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05
EXPLORATION_FRACTION = 0.3
TARGET_UPDATE_INTERVAL = 1_000
TRAIN_FREQUENCY = 4
DEFAULT_EVALUATION_EPISODES = 100


class EpisodeRewardCallback(BaseCallback):

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        for episode_info in self.locals["infos"]:
            episode = episode_info.get("episode")
            if episode is not None:
                reward = float(episode["r"])
                if not np.isfinite(reward):
                    raise ValueError("Training produced a non-finite episode reward.")
                self.episode_rewards.append(reward)
        return True


def _evaluate_policy(
    model: DQN,
    eval_env: DynamicPricingEnv,
    episode_count: int,
    seed: int,
) -> list[float]:
    episode_rewards: list[float] = []
    for episode_index in range(episode_count):
        observation, _ = eval_env.reset(seed=seed + episode_index)
        total_reward = 0.0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            action_index = int(action)
            if not eval_env.action_space.contains(action_index):
                raise ValueError(f"Loaded DQN produced invalid action {action_index}.")
            observation, reward, terminated, truncated, _ = eval_env.step(action_index)
            total_reward += float(reward)
            if terminated or truncated:
                break
        if not np.isfinite(total_reward):
            raise ValueError("Evaluation produced a non-finite episode reward.")
        episode_rewards.append(total_reward)
    return episode_rewards


def load_trained_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    env: DynamicPricingEnv | None = None,
) -> DQN:
    saved_model_path = Path(model_path)
    if not saved_model_path.is_file():
        raise FileNotFoundError(f"No trained DQN model found at {saved_model_path}.")
    return DQN.load(
        str(saved_model_path),
        env=env if env is not None else DynamicPricingEnv(),
        device="cpu",
    )


def _write_episode_rewards(path: Path, rewards: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["episode", "total_reward"])
        writer.writerows(enumerate(rewards, start=1))


def _plot_rewards(training_rewards: list[float], evaluation_rewards: list[float]) -> tuple[Path, Path]:
    reward_plot_path = RESULTS_DIR / "reward_curve.png"
    evaluation_plot_path = RESULTS_DIR / "evaluation_rewards.png"

    figure, axis = plt.subplots(figsize=(10, 5))
    if training_rewards:
        episode_numbers = np.arange(1, len(training_rewards) + 1)
        axis.plot(episode_numbers, training_rewards, alpha=0.35, color="steelblue", label="Episode reward")
        window = min(100, len(training_rewards))
        moving_average = np.convolve(
            np.asarray(training_rewards, dtype=np.float64),
            np.ones(window, dtype=np.float64) / window,
            mode="valid",
        )
        axis.plot(
            np.arange(window, len(training_rewards) + 1),
            moving_average,
            color="navy",
            linewidth=2,
            label=f"{window}-episode moving average",
        )
    axis.set_title("PriceFlow DQN training rewards")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Total revenue reward (INR)")
    axis.grid(alpha=0.2)
    if training_rewards:
        axis.legend()
    figure.tight_layout()
    figure.savefig(reward_plot_path, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        np.arange(1, len(evaluation_rewards) + 1),
        evaluation_rewards,
        marker="o",
        linestyle="-",
        color="darkgreen",
    )
    axis.set_title("PriceFlow DQN held-out evaluation rewards")
    axis.set_xlabel("Evaluation episode")
    axis.set_ylabel("Total revenue reward (INR)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(evaluation_plot_path, dpi=150)
    plt.close(figure)
    return reward_plot_path, evaluation_plot_path


def run_training(
    episodes: int = 25,
    seed: int = 42,
    total_timesteps: int | None = None,
    evaluation_episodes: int = DEFAULT_EVALUATION_EPISODES,
    learning_rate: float = LEARNING_RATE,
    buffer_size: int = BUFFER_SIZE,
    batch_size: int = BATCH_SIZE,
    gamma: float = GAMMA,
    learning_starts: int = LEARNING_STARTS,
    exploration_initial_eps: float = EXPLORATION_INITIAL_EPS,
    exploration_final_eps: float = EXPLORATION_FINAL_EPS,
    exploration_fraction: float = EXPLORATION_FRACTION,
    target_update_interval: int = TARGET_UPDATE_INTERVAL,
) -> dict[str, Any]:

    if episodes <= 0:
        raise ValueError("episodes must be a positive integer.")
    if evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be a positive integer.")
    if total_timesteps is not None and total_timesteps <= 0:
        raise ValueError("total_timesteps must be a positive integer.")
    if buffer_size <= 0 or batch_size <= 0 or learning_starts < 0:
        raise ValueError("buffer_size and batch_size must be positive; learning_starts cannot be negative.")
    if target_update_interval <= 0:
        raise ValueError("target_update_interval must be a positive integer.")
    if not 0.0 < learning_rate or not 0.0 <= gamma <= 1.0:
        raise ValueError("learning_rate must be positive and gamma must be in [0, 1].")
    if not 0.0 <= exploration_final_eps <= exploration_initial_eps <= 1.0:
        raise ValueError("Exploration epsilon values must satisfy 0 <= final <= initial <= 1.")
    if not 0.0 < exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must be in (0, 1].")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    training_env = Monitor(DynamicPricingEnv(seed=seed))
    episode_horizon = training_env.env.episode_horizon
    if total_timesteps is None:
        total_timesteps = episodes * episode_horizon

    callback = EpisodeRewardCallback()
    model = DQN(
        "MlpPolicy",
        training_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=TRAIN_FREQUENCY,
        target_update_interval=target_update_interval,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exploration_fraction,
        seed=seed,
        device="cpu",
        verbose=0,
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            reset_num_timesteps=True,
        )
        model.save(DEFAULT_MODEL_PATH)
    finally:
        training_env.close()

    evaluation_env = DynamicPricingEnv(episode_horizon=episode_horizon)
    evaluation_model = load_trained_model(DEFAULT_MODEL_PATH, env=evaluation_env)
    try:
        evaluation_rewards = _evaluate_policy(
            model=evaluation_model,
            eval_env=evaluation_env,
            episode_count=evaluation_episodes,
            seed=seed + 1_000_000,
        )
    finally:
        evaluation_model.get_env().close()

    training_rewards_path = RESULTS_DIR / "training_episode_rewards.csv"
    evaluation_rewards_path = RESULTS_DIR / "evaluation_episode_rewards.csv"
    _write_episode_rewards(training_rewards_path, callback.episode_rewards)
    _write_episode_rewards(evaluation_rewards_path, evaluation_rewards)
    reward_plot_path, evaluation_plot_path = _plot_rewards(
        callback.episode_rewards,
        evaluation_rewards,
    )

    training_mean = float(np.mean(callback.episode_rewards)) if callback.episode_rewards else 0.0
    training_std = float(np.std(callback.episode_rewards)) if callback.episode_rewards else 0.0
    summary: dict[str, Any] = {
        "requested_episodes": int(episodes),
        "completed_training_episodes": len(callback.episode_rewards),
        "timesteps": int(model.num_timesteps),
        "episode_horizon": episode_horizon,
        "seed": int(seed),
        "training_mean_reward": training_mean,
        "training_std_reward": training_std,
        "evaluation_episodes": len(evaluation_rewards),
        "evaluation_mean_reward": float(np.mean(evaluation_rewards)),
        "evaluation_std_reward": float(np.std(evaluation_rewards)),
        "model_path": str(DEFAULT_MODEL_PATH),
        "training_episode_rewards_path": str(training_rewards_path),
        "evaluation_episode_rewards_path": str(evaluation_rewards_path),
        "reward_curve_path": str(reward_plot_path),
        "evaluation_plot_path": str(evaluation_plot_path),
        "hyperparameters": {
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_starts": learning_starts,
            "exploration_initial_eps": exploration_initial_eps,
            "exploration_final_eps": exploration_final_eps,
            "exploration_fraction": exploration_fraction,
            "target_update_interval": target_update_interval,
            "train_frequency": TRAIN_FREQUENCY,
        },
    }

    summary_path = RESULTS_DIR / "training_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the PriceFlow DQN policy.")
    parser.add_argument("--episodes", type=int, default=25, help="Training episode target for a short or custom run.")
    parser.add_argument("--full-target", action="store_true", help="Use the configured 10,000-episode training target.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Optional short-run override; otherwise episodes * environment horizon.")
    parser.add_argument("--evaluation-episodes", type=int, default=DEFAULT_EVALUATION_EPISODES, help="Number of separate held-out episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training and reproducible evaluation.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
    parser.add_argument("--exploration-initial-eps", type=float, default=EXPLORATION_INITIAL_EPS)
    parser.add_argument("--exploration-final-eps", type=float, default=EXPLORATION_FINAL_EPS)
    parser.add_argument("--exploration-fraction", type=float, default=EXPLORATION_FRACTION)
    parser.add_argument("--target-update-interval", type=int, default=TARGET_UPDATE_INTERVAL)
    args = parser.parse_args()

    episodes = TRAINING_EPISODES_TARGET if args.full_target else args.episodes
    run_training(
        episodes=episodes,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        evaluation_episodes=args.evaluation_episodes,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_starts=args.learning_starts,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        exploration_fraction=args.exploration_fraction,
        target_update_interval=args.target_update_interval,
    )


if __name__ == "__main__":
    main()
