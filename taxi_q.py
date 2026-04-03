import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    # 1. Setup Environment
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if(is_training):
        # Taxi-v3 has 500 states and 6 possible actions
        q = np.zeros((env.observation_space.n, env.action_space.n)) 
    else:
        # Load the trained "brain" (Q-Table)
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)

    # 2. Hyperparameters
    learning_rate_a = 0.9   # Alpha: how much we learn from new info
    discount_factor_g = 0.9 # Gamma: how much we value future rewards
    epsilon = 1             # Exploration rate (1 = 100% random)
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    # 3. Training/Testing Loop
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False      
        truncated = False       
        total_rewards = 0

        while(not terminated and not truncated):
            # Choose Action (Epsilon-Greedy)
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() 
            else:
                action = np.argmax(q[state, :])

            # Perform Action
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Update Q-Table (The Learning Part)
            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state
            total_rewards += reward

        # Decay epsilon so we explore less over time
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Track performance
        rewards_per_episode[i] = total_rewards

        if (i + 1) % 1000 == 0:
            print(f"Episode: {i + 1} - Epsilon: {epsilon:.2f}")

    env.close()

    # 4. Save the Q-Table if training
    if is_training:
        with open('taxi.pkl', 'wb') as f:
            pickle.dump(q, f)

    # 5. Plot Results
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    
    plt.plot(sum_rewards)
    plt.title('Cumulative Rewards (Moving Average)')
    plt.savefig('taxi_rewards.png')
    plt.show()

if __name__ == '__main__':
    # Train for 15,000 episodes
    print("Starting Training...")
    run(15000, is_training=True, render=False)

    # Watch the trained agent play 10 times
    print("Training Complete. Watching Trained Agent...")
    run(10, is_training=False, render=True)