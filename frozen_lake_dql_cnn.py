import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# CNN Model for DQN
# =========================
class CNN_DQN(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 1 * 1, action_size)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# =========================
# CNN-DQN Agent
# =========================
class FrozenLakeCNNAgent:

    def __init__(self):
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

        action_size = env.action_space.n

        policy_net = CNN_DQN(action_size)
        target_net = CNN_DQN(action_size)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.lr)
        memory = ReplayBuffer(self.memory_size)

        epsilon = 1.0
        rewards = []
        epsilon_decay = []
        step_count = 0

        print("🚀 CNN-DQN Training Started")

        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:

                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_net(self._state_to_tensor(state)).argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                memory.push((state, action, next_state, reward, terminated))
                state = next_state
                total_reward += reward
                step_count += 1

                if len(memory) > self.batch_size:
                    batch = memory.sample(self.batch_size)
                    self._update(batch, policy_net, target_net, optimizer)

                if step_count > self.sync_rate:
                    target_net.load_state_dict(policy_net.state_dict())
                    step_count = 0

            rewards.append(total_reward)

            epsilon = max(epsilon - 1/episodes, 0)
            epsilon_decay.append(epsilon)

        env.close()
        torch.save(policy_net.state_dict(), "cnn_dqn_model.pt")

        self._plot_results(rewards, epsilon_decay)
        print("✅ Training Completed")

    # =========================
    # Learning Step
    # =========================
    def _update(self, batch, policy_net, target_net, optimizer):

        current_qs = []
        target_qs = []

        for state, action, next_state, reward, done in batch:

            if done:
                target = reward
            else:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(
                        target_net(self._state_to_tensor(next_state))
                    )

            current_q = policy_net(self._state_to_tensor(state))
            target_q = target_net(self._state_to_tensor(state)).clone()

            target_q[0][action] = target

            current_qs.append(current_q)
            target_qs.append(target_q)

        loss = self.loss_fn(torch.stack(current_qs), torch.stack(target_qs))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # =========================
    # State → Image Encoding
    # =========================
    def _state_to_tensor(self, state):
        tensor = torch.zeros(1, 3, 4, 4)

        r = state // 4
        c = state % 4

        # Custom color encoding (unique idea 💡)
        tensor[0][0][r][c] = 1.0     # Red
        tensor[0][1][r][c] = 0.3     # Green
        tensor[0][2][r][c] = 0.5     # Blue

        return tensor

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

        action_size = env.action_space.n

        model = CNN_DQN(action_size)
        model.load_state_dict(torch.load("cnn_dqn_model.pt"))
        model.eval()

        success = 0

        print("🧪 Testing CNN Agent")

        for _ in range(episodes):
            state, _ = env.reset()
            done = False

            while not done:
                with torch.no_grad():
                    action = model(self._state_to_tensor(state)).argmax().item()

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            success += reward

        env.close()
        print(f"✅ Success Rate: {(success/episodes)*100:.2f}%")

    # =========================
    # Plotting
    # =========================
    def _plot_results(self, rewards, epsilon):

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        moving = [sum(rewards[max(0,i-100):(i+1)]) for i in range(len(rewards))]
        plt.plot(moving)
        plt.title("Reward Trend")

        plt.subplot(1,2,2)
        plt.plot(epsilon)
        plt.title("Epsilon Decay")

        plt.savefig("cnn_dqn_results.png")
        plt.close()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    agent = FrozenLakeCNNAgent()

    # Train
    agent.train(episodes=1000, slippery=False)

    # Test
    agent.test(episodes=20, slippery=False)