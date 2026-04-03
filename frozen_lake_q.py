import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


class FrozenLakeAgent:
    def __init__(self, map_size="8x8", slippery=True):
        self.env = gym.make(
            "FrozenLake-v1",
            map_name=map_size,
            is_slippery=slippery
        )
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

        # Initialize Q-table
        self.q_table = np.zeros((self.state_size, self.action_size))

    def train(self, episodes=10000, render=False):
        if render:
            self.env = gym.make(
                "FrozenLake-v1",
                map_name="8x8",
                is_slippery=True,
                render_mode="human"
            )

        alpha = 0.9       # Learning rate
        gamma = 0.9       # Discount factor
        epsilon = 1.0     # Exploration rate
        decay = 0.0001

        rng = np.random.default_rng()
        rewards = np.zeros(episodes)

        for ep in range(episodes):
            state = self.env.reset()[0]
            done = False

            while not done:
                # Epsilon-greedy policy
                if rng.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Q-learning update
                self.q_table[state, action] += alpha * (
                    reward + gamma * np.max(self.q_table[next_state]) - self.q_table[state, action]
                )

                state = next_state

            # Decay epsilon
            epsilon = max(epsilon - decay, 0)

            # Reduce learning rate after exploration ends
            if epsilon == 0:
                alpha = 0.0001

            rewards[ep] = reward

        self.env.close()
        self._save_model()
        self._plot_rewards(rewards)

    def test(self, episodes=1000):
        self._load_model()

        success_count = 0

        for _ in range(episodes):
            state = self.env.reset()[0]
            done = False

            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

            success_count += reward

        print(f"Success Rate: {success_count / episodes * 100:.2f}%")

    def _save_model(self):
        with open("q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def _load_model(self):
        with open("q_table.pkl", "rb") as f:
            self.q_table = pickle.load(f)

    def _plot_rewards(self, rewards):
        moving_avg = np.zeros(len(rewards))

        for i in range(len(rewards)):
            moving_avg[i] = np.sum(rewards[max(0, i - 100):(i + 1)])

        plt.plot(moving_avg)
        plt.title("Training Performance (Moving Sum of Rewards)")
        plt.xlabel("Episodes")
        plt.ylabel("Reward (Last 100 Episodes)")
        plt.savefig("training_plot.png")
        plt.close()


if __name__ == "__main__":
    agent = FrozenLakeAgent()

    # Train the agent
    agent.train(episodes=1000, render=True)

    # Test performance
    agent.test(episodes=500)