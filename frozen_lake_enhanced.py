import numpy as np
import matplotlib.pyplot as plt
import pickle
from frozen_lake import FrozenLakeEnv   # your custom env file


class QLearningFrozenLake:
    def __init__(self, map_size="8x8", slippery=True, render=True):
        self.env = FrozenLakeEnv(
            map_name=map_size,
            is_slippery=slippery,
            render_mode="human" if render else None
        )

        self.state_space = self.env.observation_space.n
        self.action_space = self.env.action_space.n

        self.q_table = np.zeros((self.state_space, self.action_space))

    def train(self, episodes=5000):
        alpha = 0.9
        gamma = 0.9
        epsilon = 1.0
        decay = 0.0001

        rewards = []

        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            # 👇 Pass Q-table + episode to environment (for visualization)
            self.env.set_q(self.q_table)
            self.env.set_episode(ep)

            while not done:
                # Exploration vs Exploitation
                if np.random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Q-learning update rule
                self.q_table[state, action] += alpha * (
                    reward + gamma * np.max(self.q_table[next_state]) - self.q_table[state, action]
                )

                state = next_state
                total_reward += reward

            # Epsilon decay
            epsilon = max(epsilon - decay, 0)

            if epsilon == 0:
                alpha = 0.0001

            rewards.append(total_reward)

        self.env.close()
        self.save_model()
        self.plot_rewards(rewards)

    def test(self, episodes=500):
        self.load_model()
        success = 0

        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False

            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

            success += reward

        print(f"\n✅ Success Rate: {(success/episodes)*100:.2f}%")

    def save_model(self):
        with open("q_model.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def load_model(self):
        with open("q_model.pkl", "rb") as f:
            self.q_table = pickle.load(f)

    def plot_rewards(self, rewards):
        moving_sum = []

        for i in range(len(rewards)):
            moving_sum.append(sum(rewards[max(0, i-100):(i+1)]))

        plt.plot(moving_sum)
        plt.title("Learning Curve (Last 100 Episodes)")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.savefig("learning_curve.png")
        plt.close()


if __name__ == "__main__":
    agent = QLearningFrozenLake(render=True)

    print("🚀 Training Started...")
    agent.train(episodes=2000)

    print("🧪 Testing Agent...")
    agent.test(episodes=500)