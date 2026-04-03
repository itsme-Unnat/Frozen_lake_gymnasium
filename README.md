<a name="readme-top"></a>

<h1 align="center">Gymnasium Reinforcement Learning Implementations</h1>

<p align="center">
  <strong>A collection of AI agents trained to solve Gymnasium environments using Q-Learning and Deep Q-Networks (DQN).</strong>
</p>

---

## 🚀 Project Overview
This repository contains my personal implementations of various Reinforcement Learning (RL) algorithms. The project explores the transition from **Tabular RL** (Q-Learning) to **Deep RL** (Neural Networks) using the [Gymnasium Library](https://gymnasium.farama.org/).

The goal is to demonstrate how an agent can learn optimal strategies through trial and error across different types of observation spaces (Discrete and Continuous).

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **RL Framework:** Gymnasium
* **Deep Learning:** PyTorch
* **Data & Viz:** NumPy, Matplotlib, Pygame

---

## 📂 Implementation Modules

### 1. Beginner: Tabular Q-Learning
These scripts use a Q-Table and the Epsilon-Greedy algorithm to solve discrete environments.
* **FrozenLake-v1 (8x8):** Navigating an agent across ice to a goal without falling into holes.
  * `python frozen_lake_q.py`
* **Taxi-v3:** Picking up and dropping off passengers in a grid-based world.
  * `python taxi_q.py`
* **Classic Control:** Solvers for **MountainCar-v0**, **CartPole-v1**, and **Acrobot-v1**.

### 2. Intermediate: Deep Q-Learning (DQN)
Implementations using Neural Networks to approximate Q-values, essential for environments with continuous states.
* **Frozen Lake (DQN):** Solving the 4x4 grid using PyTorch-based Deep Q-Networks.
* **Mountain Car (DQN):** Using Experience Replay to solve the classic underpowered car problem.
* **CNN Integration:** Experiments with Convolutional Neural Networks for image-based state processing.

### 3. Advanced: Stable Baselines3 (SB3)
Benchmarking performance using professional-grade RL frameworks.
* **Humanoid-v4:** Training complex walking motions.
* **BipedalWalker-v3:** Implementation of automated training checkpoints.

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/itsme-Unnat/Frozen_lake_gymnasium.git](https://github.com/itsme-Unnat/Frozen_lake_gymnasium.git)
   cd Frozen_lake_gymnasium
