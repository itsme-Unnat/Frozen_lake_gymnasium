"""
Mountain Car — Deep Q-Learning (DQN) Solution
===============================================
Environment : MountainCar-v0  (Gymnasium)
Algorithm   : Deep Q-Learning with Experience Replay & Target Network
Author      : Your Name
Date        : 2026

Description
-----------
A Deep Q-Learning agent that learns to drive an underpowered car up a steep
hill. Two neural networks are used:
  - Policy DQN  : updated every step via gradient descent
  - Target DQN  : a frozen copy of the policy network, synced periodically
                  to stabilise training

The continuous (position, velocity) observation is discretised into 20 bins
per dimension before being fed into the networks.

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

import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Neural Network — Deep Q-Network
# ---------------------------------------------------------------------------
class DQN(nn.Module):
    """
    A simple two-layer fully-connected Q-Network.

    Parameters
    ----------
    in_states   : Number of input features (observation dimensions).
    h1_nodes    : Number of hidden-layer neurons.
    out_actions : Number of output Q-values (one per action).
    """

    def __init__(self, in_states: int, h1_nodes: int, out_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)    # hidden layer
        self.out = nn.Linear(h1_nodes, out_actions)  # output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))  # ReLU activation on hidden layer
        return self.out(x)       # raw Q-values (no activation on output)


# ---------------------------------------------------------------------------
# Experience Replay Memory
# ---------------------------------------------------------------------------
class ReplayMemory:
    """
    Circular buffer that stores (state, action, next_state, reward, done)
    transitions and supports uniform random sampling for mini-batch training.
    """

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition: tuple):
        """Store a single transition."""
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list:
        """Return a random batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


# ---------------------------------------------------------------------------
# Deep Q-Learning Agent
# ---------------------------------------------------------------------------
class MountainCarDQL:
    """
    Deep Q-Learning agent for MountainCar-v0.

    Hyperparameters
    ---------------
    All class-level constants can be tuned without touching the training logic.
    """

    # ── Hyperparameters ─────────────────────────────────────────────────────
    LEARNING_RATE      = 0.01     # Adam optimiser learning rate (α)
    DISCOUNT_FACTOR    = 0.9      # Future-reward discount (γ)
    NETWORK_SYNC_STEPS = 50_000   # Steps between policy → target network sync
    REPLAY_MEMORY_SIZE = 100_000  # Maximum transitions stored in replay buffer
    MINI_BATCH_SIZE    = 32       # Transitions sampled per optimisation step
    NUM_BINS           = 20       # Discretisation bins per observation dimension
    MODEL_SAVE_PREFIX  = "mountaincar_dql"

    # ── Loss & optimiser (initialised in train) ──────────────────────────────
    loss_fn   = nn.MSELoss()
    optimizer = None

    # ── Observation bin edges (set in train / test) ──────────────────────────
    pos_space = None
    vel_space = None

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------
    def _build_spaces(self, env: gym.Env):
        """Construct discretised bin edges from the environment's obs limits."""
        self.pos_space = np.linspace(
            env.observation_space.low[0],
            env.observation_space.high[0],
            self.NUM_BINS,
        )
        self.vel_space = np.linspace(
            env.observation_space.low[1],
            env.observation_space.high[1],
            self.NUM_BINS,
        )

    def state_to_tensor(self, state) -> torch.Tensor:
        """
        Discretise a continuous (position, velocity) state and return it as a
        FloatTensor suitable for the DQN.

        Example
        -------
        Input  → (0.3, -0.03)
        Output → tensor([16., 6.])
        """
        pos_idx = np.digitize(state[0], self.pos_space)
        vel_idx = np.digitize(state[1], self.vel_space)
        return torch.FloatTensor([pos_idx, vel_idx])

    # -----------------------------------------------------------------------
    # Optimisation step (mini-batch Bellman update)
    # -----------------------------------------------------------------------
    def _optimise(self, mini_batch: list, policy_dqn: DQN, target_dqn: DQN):
        """
        Perform one gradient-descent step on the policy network using a
        mini-batch of stored transitions.
        """
        current_q_list = []
        target_q_list  = []

        for state, action, next_state, reward, terminated in mini_batch:

            if terminated:
                # Terminal state: target is just the immediate reward
                target = torch.FloatTensor([reward])
            else:
                # Bellman equation: r + γ * max Q(s', a'; θ⁻)
                with torch.no_grad():
                    best_next_q = target_dqn(self.state_to_tensor(next_state)).max()
                    target = torch.FloatTensor([reward + self.DISCOUNT_FACTOR * best_next_q])

            # Current Q-values from policy network
            current_q = policy_dqn(self.state_to_tensor(state))
            current_q_list.append(current_q)

            # Build target Q-vector (only update the taken action)
            target_q = target_dqn(self.state_to_tensor(state)).detach().clone()
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute MSE loss and backpropagate
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # -----------------------------------------------------------------------
    # Training progress plot
    # -----------------------------------------------------------------------
    def _plot_progress(self, rewards_per_episode: list, epsilon_history: list):
        """Save a two-panel training-progress figure to disk."""
        plt.figure(1, figsize=(12, 5))
        plt.clf()

        # Panel 1 — reward per episode
        plt.subplot(1, 2, 1)
        plt.plot(rewards_per_episode, color="steelblue", linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode")

        # Panel 2 — epsilon decay
        plt.subplot(1, 2, 2)
        plt.plot(epsilon_history, color="darkorange", linewidth=1.2)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon Decay")

        plt.tight_layout()
        plt.savefig(f"{self.MODEL_SAVE_PREFIX}.png")
        print(f"Training plot saved → {self.MODEL_SAVE_PREFIX}.png")

    # -----------------------------------------------------------------------
    # Public: train
    # -----------------------------------------------------------------------
    def train(self, episodes: int, render: bool = False):
        """
        Train the DQN agent on MountainCar-v0.

        Parameters
        ----------
        episodes : Total number of training episodes.
        render   : If True, open the visual environment window.
        """
        env = gym.make("MountainCar-v0", render_mode="human" if render else None)
        num_states  = env.observation_space.shape[0]   # 2: position & velocity
        num_actions = env.action_space.n               # 3: left / neutral / right

        self._build_spaces(env)

        # ── Networks ────────────────────────────────────────────────────────
        policy_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())  # start in sync

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.LEARNING_RATE)

        # ── Replay buffer & tracking ─────────────────────────────────────────
        memory             = ReplayMemory(self.REPLAY_MEMORY_SIZE)
        rewards_per_episode: list = []
        epsilon_history:     list = []

        epsilon      = 1.0                      # start fully random
        step_count   = 0
        goal_reached = False
        best_reward  = -200.0

        print(f"Starting training for {episodes} episodes …\n")

        # ── Episode loop ─────────────────────────────────────────────────────
        for episode in range(episodes):
            state      = env.reset()[0]
            terminated = False
            total_reward = 0.0

            while not terminated and total_reward > -1_000:

                # ε-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_tensor(state)).argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                memory.push((state, action, next_state, reward, terminated))
                state       = next_state
                step_count += 1

            rewards_per_episode.append(total_reward)

            if terminated:
                goal_reached = True

            # Save best model checkpoint
            if total_reward > best_reward:
                best_reward = total_reward
                save_path   = f"{self.MODEL_SAVE_PREFIX}_{episode}.pt"
                torch.save(policy_dqn.state_dict(), save_path)
                print(f"  ★  New best reward {best_reward:.1f} at episode {episode} → {save_path}")

            # Periodic progress report & plot
            if episode != 0 and episode % 1_000 == 0:
                print(f"  Episode {episode:>6} / {episodes}  |  ε = {epsilon:.4f}")
                self._plot_progress(rewards_per_episode, epsilon_history)

            # Optimise once enough experience is available and goal seen once
            if len(memory) > self.MINI_BATCH_SIZE and goal_reached:
                mini_batch = memory.sample(self.MINI_BATCH_SIZE)
                self._optimise(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1.0 / episodes, 0.0)
                epsilon_history.append(epsilon)

                # Sync target network periodically
                if step_count >= self.NETWORK_SYNC_STEPS:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
                    print(f"  ↻  Target network synced at episode {episode}")

        env.close()
        self._plot_progress(rewards_per_episode, epsilon_history)
        print("\nTraining complete.")

    # -----------------------------------------------------------------------
    # Public: test
    # -----------------------------------------------------------------------
    def test(self, episodes: int, model_filepath: str):
        """
        Evaluate a trained policy on MountainCar-v0 with rendering.

        Parameters
        ----------
        episodes       : Number of evaluation episodes to run.
        model_filepath : Path to a saved policy-network .pt file.
        """
        env = gym.make("MountainCar-v0", render_mode="human")
        num_states  = env.observation_space.shape[0]
        num_actions = env.action_space.n

        self._build_spaces(env)

        # Load saved policy weights
        policy_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load(model_filepath))
        policy_dqn.eval()
        print(f"Loaded model from '{model_filepath}'")

        for episode in range(episodes):
            state      = env.reset()[0]
            terminated = False
            truncated  = False
            total_reward = 0.0

            while not terminated and not truncated:
                with torch.no_grad():
                    action = policy_dqn(self.state_to_tensor(state)).argmax().item()

                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            print(f"  Episode {episode + 1:>3} / {episodes}  |  reward: {total_reward:.1f}")

        env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = MountainCarDQL()

    # ── Train (uncomment to run) ─────────────────────────────────────────────
    # agent.train(episodes=20_000, render=False)

    # ── Evaluate a saved model ───────────────────────────────────────────────
    agent.test(episodes=10, model_filepath="mountaincar_dql_17000.pt")