"""
Frozen Lake 8x8 — Enhanced Q-Learning Solution
================================================
Environment : FrozenLake-enhanced  (custom Gymnasium registration)
Algorithm   : Q-Learning with Epsilon-Greedy exploration
Author      : Your Name
Date        : 2026

Description
-----------
This file is an enhanced version of frozen_lake_q.py. It registers and uses
a custom FrozenLake environment (frozen_lake_enhanced.py) that overlays live
Q-values on every cell of the map during rendering, making it easy to watch
the agent's policy evolve in real time.

The Q-Learning logic is identical to the base version — the only difference
is that the Q-table and current episode index are passed back to the
environment each step so the overlay can stay in sync.

Environment Registration
------------------------
The enhanced environment must be registered before use. The entry_point
points to frozen_lake_enhanced.py which must exist in the same directory.

Observation Space
-----------------
  64 discrete states (0 = top-left, 63 = bottom-right on an 8x8 grid)

Action Space
------------
  0 → left
  1 → down
  2 → right
  3 → up
"""

import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Register the enhanced Frozen Lake environment
# (entry_point = "<filename_without_.py>:<ClassName>")
# ---------------------------------------------------------------------------
gym.register(
    id="FrozenLake-enhanced",
    entry_point="frozen_lake_enhanced:FrozenLakeEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,   # theoretical optimum ≈ 0.91
)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE        = 0.9       # Alpha  – step size for Q-value updates
DISCOUNT_FACTOR      = 0.9       # Gamma  – weight placed on future rewards
EPSILON_START        = 1.0       # Start fully random (100 % exploration)
EPSILON_DECAY        = 0.0001    # Subtracted from ε each episode (1 / 0.0001 = 10 000 steps)
LEARNING_RATE_FINAL  = 0.0001    # Learning rate used once exploration ends
Q_TABLE_FILE         = "frozen_lake8x8.pkl"
PLOT_FILE            = "frozen_lake8x8.png"


# ---------------------------------------------------------------------------
# Helper — plot and save training curve
# ---------------------------------------------------------------------------
def plot_rewards(rewards_per_episode: np.ndarray, window: int = 100):
    """
    Plot the rolling sum of rewards over the last `window` episodes and
    save the figure to disk.
    """
    episodes = len(rewards_per_episode)
    sum_rewards = np.array([
        np.sum(rewards_per_episode[max(0, t - window): t + 1])
        for t in range(episodes)
    ])

    plt.figure(figsize=(10, 5))
    plt.plot(sum_rewards, color="steelblue", linewidth=1.2)
    plt.xlabel("Episode")
    plt.ylabel(f"Successes in last {window} episodes")
    plt.title("FrozenLake 8x8 (Enhanced) — Q-Learning Training Curve")
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()
    print(f"Training curve saved → {PLOT_FILE}")


# ---------------------------------------------------------------------------
# Core training / evaluation loop
# ---------------------------------------------------------------------------
def run(episodes: int, is_training: bool = True, render: bool = False):
    """
    Train or evaluate the Q-Learning agent on the enhanced FrozenLake 8x8.

    Parameters
    ----------
    episodes    : Number of episodes to run.
    is_training : If True, update the Q-table and save it afterwards.
                  If False, load a saved Q-table and run in greedy mode.
    render      : If True, open the visual environment window with Q-overlay.
    """
    render_mode = "human" if render else None
    env = gym.make(
        "FrozenLake-enhanced",
        desc=None,
        map_name="8x8",
        is_slippery=True,
        render_mode=render_mode,
    )

    # ── Q-table ─────────────────────────────────────────────────────────────
    if is_training:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))  # 64 × 4
        print(f"New Q-table initialised  {q_table.shape}")
    else:
        with open(Q_TABLE_FILE, "rb") as f:
            q_table = pickle.load(f)
        print(f"Q-table loaded from '{Q_TABLE_FILE}'  {q_table.shape}")

    # ── Exploration schedule ─────────────────────────────────────────────────
    epsilon       = EPSILON_START
    learning_rate = LEARNING_RATE
    rng           = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    # ── Episode loop ─────────────────────────────────────────────────────────
    for episode in range(episodes):
        state      = env.reset()[0]   # int in [0, 63]
        terminated = False            # True when hole reached or goal reached
        truncated  = False            # True when step count exceeds 200

        while not terminated and not truncated:

            # ε-greedy action selection
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()       # explore
            else:
                action = np.argmax(q_table[state, :])    # exploit

            next_state, reward, terminated, truncated, _ = env.step(action)

            # Bellman update
            if is_training:
                current_q   = q_table[state, action]
                best_next_q = np.max(q_table[next_state, :])
                q_table[state, action] = current_q + learning_rate * (
                    reward + DISCOUNT_FACTOR * best_next_q - current_q
                )

            # Pass Q-table and episode index to enhanced renderer
            if env.render_mode == "human":
                env.set_q(q_table)
                env.set_episode(episode)

            state = next_state

        # Decay epsilon (floor at 0); freeze learning rate once fully greedy
        epsilon = max(epsilon - EPSILON_DECAY, 0.0)
        if epsilon == 0.0:
            learning_rate = LEARNING_RATE_FINAL

        # Record success (reward == 1 means the goal was reached)
        if reward == 1:
            rewards_per_episode[episode] = 1

        if (episode + 1) % 1_000 == 0:
            success_rate = rewards_per_episode[max(0, episode - 999): episode + 1].mean()
            print(f"  Episode {episode + 1:>6} / {episodes}  |  "
                  f"ε = {epsilon:.4f}  |  "
                  f"success rate (last 1k): {success_rate:.1%}")

    env.close()

    # ── Save Q-table ─────────────────────────────────────────────────────────
    if is_training:
        with open(Q_TABLE_FILE, "wb") as f:
            pickle.dump(q_table, f)
        print(f"Q-table saved → '{Q_TABLE_FILE}'")

    plot_rewards(rewards_per_episode)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run(episodes=15_000, is_training=True, render=True)