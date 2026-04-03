import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Neural Network Model
# =========================
class DeepQNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =========================
# Replay Memory
# =========================
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# =========================
# DQN Agent
# =========================
class FrozenLakeDQNAgent:

    def __init__(self):
        # Hyperparameters
        self.lr = 0.001
        self.gamma = 0.9
        self.batch_size = 32
        self.memory_size = 1000
        self.sync_rate = 10

        self.loss_fn = nn.MSELoss()

    def train(self, episodes=1000, render=False, slippery=False):

        env = gym.make(
            "FrozenLake-v1",
            map_name="4x4",
            is_slippery=slippery,
            render_mode="human" if render else None
        )

        state_size = env.observation_space.n
        action_size = env.action_space.n

        # Networks
        policy_net = DeepQNetwork(state_size, state_size, action_size)
        target_net = DeepQNetwork(state_size, state_size, action_size)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.lr)
        memory = ExperienceBuffer(self.memory_size)

        epsilon = 1.0
        rewards = []
        epsilon_history = []
        step_counter = 0

        print("🚀 Training Started...")

        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:

                # Epsilon-Greedy
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_net(self._one_hot(state, state_size)).argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                memory.store((state, action, next_state, reward, terminated))
                state = next_state
                total_reward += reward
                step_counter += 1

                # Train only if enough memory
                if memory.size() > self.batch_size:
                    batch = memory.sample(self.batch_size)
                    self._learn(batch, policy_net, target_net, optimizer, state_size)

                # Sync networks
                if step_counter > self.sync_rate:
                    target_net.load_state_dict(policy_net.state_dict())
                    step_counter = 0

            rewards.append(total_reward)

            # Epsilon decay
            epsilon = max(epsilon - 1/episodes, 0)
            epsilon_history.append(epsilon)

        env.close()

        torch.save(policy_net.state_dict(), "dqn_frozenlake.pt")

        self._plot(rewards, epsilon_history)
        print("✅ Training Completed")

    # =========================
    # Learning Step
    # =========================
    def _learn(self, batch, policy_net, target_net, optimizer, state_size):

        current_qs = []
        target_qs = []

        for state, action, next_state, reward, done in batch:

            if done:
                target = reward
            else:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(
                        target_net(self._one_hot(next_state, state_size))
                    )

            current_q = policy_net(self._one_hot(state, state_size))
            target_q = target_net(self._one_hot(state, state_size)).clone()

            target_q[action] = target

            current_qs.append(current_q)
            target_qs.append(target_q)

        loss = self.loss_fn(torch.stack(current_qs), torch.stack(target_qs))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # =========================
    # State Encoding
    # =========================
    def _one_hot(self, state, size):
        vec = torch.zeros(size)
        vec[state] = 1
        return vec

    # =========================
    # Testing
    # =========================
    def test(self, episodes=10, slippery=False):

        env = gym.make(
            "FrozenLake-v1",
            map_name="4x4",
            is_slippery=slippery,
            render_mode="human"
        )

        state_size = env.observation_space.n
        action_size = env.action_space.n

        model = DeepQNetwork(state_size, state_size, action_size)
        model.load_state_dict(torch.load("dqn_frozenlake.pt"))
        model.eval()

        success = 0

        print("🧪 Testing Agent...")

        for _ in range(episodes):
            state, _ = env.reset()
            done = False

            while not done:
                with torch.no_grad():
                    action = model(self._one_hot(state, state_size)).argmax().item()

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            success += reward

        env.close()
        print(f"✅ Success Rate: {(success/episodes)*100:.2f}%")

    # =========================
    # Plot Results
    # =========================
    def _plot(self, rewards, epsilon_history):

        plt.figure(figsize=(10,4))

        # Rewards
        plt.subplot(1,2,1)
        moving_avg = [sum(rewards[max(0,i-100):(i+1)]) for i in range(len(rewards))]
        plt.plot(moving_avg)
        plt.title("Rewards (Last 100 Episodes)")

        # Epsilon
        plt.subplot(1,2,2)
        plt.plot(epsilon_history)
        plt.title("Epsilon Decay")

        plt.savefig("dqn_training.png")
        plt.close()


# =========================
# Run Program
# =========================
if __name__ == "__main__":

    agent = FrozenLakeDQNAgent()

    # Train
    agent.train(episodes=1000, slippery=False)

    # Test
    agent.test(episodes=20, slippery=False)