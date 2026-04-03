import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


class CartPoleQLearning:

    def __init__(self, render=False):
        self.env = gym.make(
            "CartPole-v1",
            render_mode="human" if render else None
        )

        # Discretization bins
        self.pos_bins = np.linspace(-2.4, 2.4, 10)
        self.vel_bins = np.linspace(-4, 4, 10)
        self.angle_bins = np.linspace(-0.2095, 0.2095, 10)
        self.ang_vel_bins = np.linspace(-4, 4, 10)

        # Q-table
        self.q_table = np.zeros((
            len(self.pos_bins) + 1,
            len(self.vel_bins) + 1,
            len(self.angle_bins) + 1,
            len(self.ang_vel_bins) + 1,
            self.env.action_space.n
        ))

        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.00001

        self.rng = np.random.default_rng()

    # =========================
    # State Discretization
    # =========================
    def _discretize(self, state):
        return (
            np.digitize(state[0], self.pos_bins),
            np.digitize(state[1], self.vel_bins),
            np.digitize(state[2], self.angle_bins),
            np.digitize(state[3], self.ang_vel_bins)
        )

    # =========================
    # Training
    # =========================
    def train(self):
        rewards_history = []
        episode = 0

        print("🚀 Training Started...")

        while True:
            state, _ = self.env.reset()
            state_d = self._discretize(state)

            total_reward = 0
            done = False

            while not done and total_reward < 10000:

                # Epsilon-greedy
                if self.rng.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state_d])

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state_d = self._discretize(next_state)

                # Q-learning update
                self.q_table[state_d][action] += self.alpha * (
                    reward +
                    self.gamma * np.max(self.q_table[next_state_d]) -
                    self.q_table[state_d][action]
                )

                state_d = next_state_d
                total_reward += reward

            rewards_history.append(total_reward)

            # Logging
            mean_reward = np.mean(rewards_history[-100:])

            if episode % 100 == 0:
                print(f"Episode: {episode} | Reward: {total_reward:.0f} | Mean(100): {mean_reward:.1f} | Epsilon: {self.epsilon:.2f}")

            # Stop condition
            if mean_reward > 1000:
                print("✅ Environment Solved!")
                break

            # Decay epsilon
            self.epsilon = max(self.epsilon - self.epsilon_decay, 0)

            episode += 1

        self.env.close()
        self._save_model()
        self._plot(rewards_history)

    # =========================
    # Testing
    # =========================
    def test(self, episodes=5):
        self._load_model()

        for ep in range(episodes):
            state, _ = self.env.reset()
            state_d = self._discretize(state)

            total_reward = 0
            done = False

            while not done:
                action = np.argmax(self.q_table[state_d])
                state, reward, terminated, truncated, _ = self.env.step(action)
                state_d = self._discretize(state)
                done = terminated or truncated
                total_reward += reward

            print(f"Test Episode {ep+1}: Reward = {total_reward}")

        self.env.close()

    # =========================
    # Save / Load
    # =========================
    def _save_model(self):
        with open("cartpole_q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def _load_model(self):
        with open("cartpole_q_table.pkl", "rb") as f:
            self.q_table = pickle.load(f)

    # =========================
    # Plotting
    # =========================
    def _plot(self, rewards):
        moving_avg = [
            np.mean(rewards[max(0, i-100):(i+1)])
            for i in range(len(rewards))
        ]

        plt.plot(moving_avg)
        plt.title("CartPole Q-Learning Performance")
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward (Last 100)")
        plt.savefig("cartpole_training.png")
        plt.close()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    agent = CartPoleQLearning(render=False)

    # Train
    agent.train()

    # Test
    agent = CartPoleQLearning(render=True)
    agent.test(episodes=3)