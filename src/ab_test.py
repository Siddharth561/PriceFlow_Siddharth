
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from stable_baselines3 import DQN

from .config import SURGE_LEVELS
from .env import PricingEnv

EPISODES = 2_000


def evaluate(model: DQN | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = PricingEnv()
    episode_revenues = np.zeros(EPISODES)
    cumulative = np.zeros(EPISODES * 24)
    regrets: list[float] = []
    cursor = 0
    for episode in range(EPISODES):
        observation, _ = env.reset(seed=10_000 + episode)
        done = False
        while not done:
            if model is None:
                action = int(np.flatnonzero(SURGE_LEVELS == (1.5 if env.weather == 2 else 1.0))[0])
            else:
                chosen, _ = model.predict(observation, deterministic=True)
                action = int(chosen)
            chosen_revenue = env.expected_revenue(action)
            regrets.append(max(env.expected_revenue(a) for a in range(len(SURGE_LEVELS))) - chosen_revenue)
            observation, _, done, _, info = env.step(action)
            episode_revenues[episode] += info["revenue"]
            cumulative[cursor] = info["revenue"] + (cumulative[cursor - 1] if cursor else 0)
            cursor += 1
    return episode_revenues, cumulative, np.asarray(regrets)


def main() -> None:
    Path("results").mkdir(exist_ok=True)
    model = DQN.load("models/dqn_pricing", device="cpu")
    naive_revenue, naive_cumulative, naive_regret = evaluate(None)
    rl_revenue, rl_cumulative, rl_regret = evaluate(model)
    lift = (rl_revenue.mean() / naive_revenue.mean() - 1) * 100
    test = ttest_ind(rl_revenue, naive_revenue, equal_var=False)

    print(f"{'Policy':<12} {'Mean daily revenue (INR)':>25}")
    print("-" * 39)
    print(f"{'Naive':<12} {naive_revenue.mean():>25,.2f}")
    print(f"{'RL (DQN)':<12} {rl_revenue.mean():>25,.2f}")
    print(f"Revenue lift: {lift:.2f}%")
    print(f"Welch t-test p-value: {test.pvalue:.6g}")

    fig, ax = plt.subplots(figsize=(10, 5))
    steps = np.arange(1, len(naive_cumulative) + 1)
    ax.plot(steps, naive_cumulative, label="Naive", alpha=0.8)
    ax.plot(steps, rl_cumulative, label="RL (DQN)", alpha=0.8)
    ax.set(xlabel="Hourly decision (matched episodes)", ylabel="Cumulative revenue (INR)", title="Cumulative revenue by policy")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("results/cumulative_revenue.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(len(naive_regret)), naive_regret, label="Naive", alpha=0.5)
    ax.plot(np.arange(len(rl_regret)), rl_regret, label="RL (DQN)", alpha=0.5)
    ax.set(xlabel="Hourly decision", ylabel="One-step regret (INR)", title="One-step greedy-oracle regret")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("results/regret_plot.png", dpi=150)
    plt.close(fig)

    results = {
        "episodes_per_policy": EPISODES,
        "naive_mean_daily_revenue": float(naive_revenue.mean()),
        "rl_mean_daily_revenue": float(rl_revenue.mean()),
        "revenue_lift_percent": float(lift),
        "welch_t_statistic": float(test.statistic),
        "welch_p_value": float(test.pvalue),
        "naive_mean_one_step_regret": float(naive_regret.mean()),
        "rl_mean_one_step_regret": float(rl_regret.mean()),
        "naive_episode_revenues": naive_revenue.tolist(),
        "rl_episode_revenues": rl_revenue.tolist(),
    }
    with open("results/ab_results.json", "w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=2)


if __name__ == "__main__":
    main()
