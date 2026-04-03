"""
Mountain Car - Q-Learning Solution
===================================
Environment : MountainCar-v0  (Gymnasium)
Algorithm   : Q-Learning with Epsilon-Greedy exploration
Author      : Your Name
Date        : 2026

Description
-----------
A Q-Learning agent that learns to drive an underpowered car up a steep hill.
Because the car cannot climb directly, it must learn to build momentum by
rocking back and forth. The continuous observation space (position + velocity)
is discretised into a 20x20 grid so a standard Q-table can be used.

Observation Space
-----------------
  position : -1.2  →  0.6   (goal ≥ 0.45)
  velocity : -0.07 →  0.07

Action Space
------------
  0 → push left
  1 → no push (neutral)
  2 → push right
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE    = 0.9   # Alpha  – how strongly new info overrides old
DISCOUNT_FACTOR  = 0.9   # Gamma  – importance of future rewards
EPSILON_START    = 1.0   # Start fully random (100 % exploration)
NUM_BINS         = 20    # Discretisation bins per observation dimension
Q_TABLE_FILE     = "mountain_car.pkl"


# ---------------------------------------------------------------------------
# Helper – build discretised observation spaces
# ---------------------------------------------------------------------------
def build_spaces(env: gym.Env, num_bins: int = NUM_BINS):
    """Return evenly-spaced bin edges for position and velocity."""
    pos_space = np.linspace(
        env.observation_space.low[0],
        env.observation_space.high[0],
        num_bins,
    )
    vel_space = np.linspace(
        env.observation_space.low[1],
        env.observation_space.high[1],
        num_bins,
    )
    return pos_space, vel_space


def discretise(state, pos_space, vel_space):
    """Map a continuous (position, velocity) state to bin indices."""
    pos_idx = np.digitize(state[0], pos_space)
    vel_idx = np.digitize(state[1], vel_space)
    return pos_idx, vel_idx


# ---------------------------------------------------------------------------
# Helper – plot and save training curve
# ---------------------------------------------------------------------------
def plot_rewards(rewards_per_episode: np.ndarray, window: int = 100):
    """Plot the rolling mean reward and save to mountain_car.png."""
    episodes = len(rewards_per_episode)
    mean_rewards = np.array([
        np.mean(rewards_per_episode[max(0, t - window): t + 1])
        for t in range(episodes)
    ])

    plt.figure(figsize=(10, 5))
    plt.plot(mean_rewards, color="steelblue", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel(f"Mean Reward (last {window} eps)")
    plt.title("MountainCar-v0 — Q-Learning Training Curve")
    plt.tight_layout()
    plt.savefig("mountain_car.png")
    plt.close()
    print("Training curve saved → mountain_car.png")


# ---------------------------------------------------------------------------
# Core training / evaluation loop
# ---------------------------------------------------------------------------
def run(episodes: int, is_training: bool = True, render: bool = False):
    """
    Train or evaluate the Q-Learning agent on MountainCar-v0.

    Parameters
    ----------
    episodes    : Number of episodes to run.
    is_training : If True, update the Q-table and save it afterwards.
                  If False, load a saved Q-table and run in greedy mode.
    render      : If True, open the visual environment window.
    """
    render_mode = "human" if render else None
    env = gym.make("MountainCar-v0", render_mode=render_mode)

    pos_space, vel_space = build_spaces(env)

    # ── Q-table ────────────────────────────────────────────────────────────
    if is_training:
        # Shape: (bins × bins × actions) = 20 × 20 × 3
        q_table = np.zeros((NUM_BINS, NUM_BINS, env.action_space.n))
        print(f"New Q-table initialised  {q_table.shape}")
    else:
        with open(Q_TABLE_FILE, "rb") as f:
            q_table = pickle.load(f)
        print(f"Q-table loaded from '{Q_TABLE_FILE}'  {q_table.shape}")

    # ── Exploration schedule ────────────────────────────────────────────────
    epsilon       = EPSILON_START
    epsilon_decay = 2.0 / episodes   # Reaches ~0 by the halfway point
    rng           = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    # ── Episode loop ────────────────────────────────────────────────────────
    for episode in range(episodes):
        raw_state      = env.reset()[0]
        pos_idx, vel_idx = discretise(raw_state, pos_space, vel_space)

        terminated     = False
        total_reward   = 0.0

        while not terminated and total_reward > -1_000:

            # ε-greedy action selection
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()          # explore
            else:
                action = np.argmax(q_table[pos_idx, vel_idx, :])  # exploit

            next_state, reward, terminated, _, _ = env.step(action)
            next_pos_idx, next_vel_idx = discretise(next_state, pos_space, vel_space)

            # Bellman update
            if is_training:
                current_q  = q_table[pos_idx, vel_idx, action]
                best_next_q = np.max(q_table[next_pos_idx, next_vel_idx, :])
                q_table[pos_idx, vel_idx, action] = current_q + LEARNING_RATE * (
                    reward + DISCOUNT_FACTOR * best_next_q - current_q
                )

            pos_idx, vel_idx = next_pos_idx, next_vel_idx
            total_reward    += reward

        # Decay epsilon (floor at 0)
        epsilon = max(epsilon - epsilon_decay, 0.0)
        rewards_per_episode[episode] = total_reward

        if (episode + 1) % 500 == 0 or episode == 0:
            print(f"  Episode {episode + 1:>5} / {episodes}  |  "
                  f"reward: {total_reward:>8.1f}  |  ε = {epsilon:.4f}")

    env.close()

    # ── Persist Q-table ─────────────────────────────────────────────────────
    if is_training:
        with open(Q_TABLE_FILE, "wb") as f:
            pickle.dump(q_table, f)
        print(f"Q-table saved → '{Q_TABLE_FILE}'")

    plot_rewards(rewards_per_episode)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ── Training (uncomment to train from scratch) ──────────────────────────
    # run(episodes=5000, is_training=True, render=False)

    # ── Evaluation (runs 10 greedy episodes with rendering) ─────────────────
    run(episodes=10, is_training=False, render=True)