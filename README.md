<a name="readme-top"></a>

<h1 align="center">🧊 Gymnasium (Deep) Reinforcement Learning Tutorials</h1>

<p align="center">
  A personal collection of Python solutions and tutorials for training RL agents using the <a href="https://gymnasium.farama.org/">Gymnasium Library</a> (formerly OpenAI Gym).
</p>

<p align="center">
  <img src="https://gymnasium.farama.org/_images/frozen_lake.gif" alt="Frozen Lake Demo" width="300"/>
</p>

---

## 👋 About This Repo

This repository documents my journey learning **Reinforcement Learning** from the ground up — starting with simple grid-world environments like **Frozen Lake** and progressing to deep RL on complex continuous spaces. Each solution includes clean, well-commented Python code.

---

## 🛠️ Installation

The [Gymnasium Library](https://gymnasium.farama.org/) is fully supported on **Linux** and **macOS**. Windows users may encounter issues with the Box2D package (Bipedal Walker, Car Racing, Lunar Lander):

- `ERROR: Failed building wheels for box2d-py`
- `ERROR: Command swig.exe failed`
- `ERROR: Microsoft Visual C++ 14.0 or greater is required`

**Recommended fix:** Use Windows Subsystem for Linux (WSL) to install Gymnasium Box2D environments on Windows.

---

## 🐣 Beginner RL Tutorials

### 🧊 Q-Learning — Frozen Lake 8x8

> **Best starting point for beginners!**

Solves the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 8x8 map using Q-Learning and the **Epsilon-Greedy** algorithm for exploration and exploitation. Focused purely on practical application — no heavy theory required.

**📄 Code:** [`frozen_lake_q.py`](frozen_lake_q.py)

---

### 🔍 Q-Learning — Frozen Lake 8x8 Enhanced

An "enhanced" version of FrozenLake that overlays **live Q-values** on the grid during training. The map fills the entire screen for readability, with shortcut keys to control animation speed.

**📄 Code:**
- [`frozen_lake_enhanced.py`](frozen_lake_enhanced.py) — Modified environment with Q-value overlay
- [`frozen_lake_qe.py`](frozen_lake_qe.py) — Training script using the enhanced environment

---

### 🚕 Q-Learning — Taxi (Multidimensional Discrete Space)

The [Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment: the agent learns to pick up and drop off passengers. Similar to Frozen Lake but with a more complex observation space.

**📄 Code:** [`taxi_q.py`](taxi_q.py)

---

### 🚗 Q-Learning — Mountain Car (Continuous Observation Space)

Solves [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) — an environment with a **continuous** observation space (no discrete grid cells). The car traverses a mountain road with no clear state boundaries.

**📄 Code:** [`mountain_car_q.py`](mountain_car_q.py)

---

### 🎯 Q-Learning — Cart Pole (Multiple Continuous Spaces)

Solves [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) — a more complex continuous environment tracking cart position, cart velocity, pole angle, and angular velocity simultaneously.

**📄 Code:** [`cartpole_q.py`](cartpole_q.py)

---

## 🧠 Deep Reinforcement Learning Tutorials

### 🔢 Getting Started with Neural Networks

A hands-on end-to-end walkthrough of Loss and Gradient Descent on the simplest possible neural network — great foundation before diving into Deep RL.

**📁 Repo:** [Basic Neural Network](https://github.com/johnnycode8/basic_neural_network)

---

### 🤖 Deep Q-Learning (DQN) Explained

Explains how **Deep Q-Learning (DQL)** uses two neural networks — a **Policy DQN** and a **Target DQN** — to train [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 4x4. Also covers Epsilon-Greedy and **Experience Replay**. Built with **PyTorch**.

**Topics covered:** Q-Table vs DQN, how the DQN learns, Experience Replay, full code walkthrough.

**📄 Code:** [`frozen_lake_dql.py`](frozen_lake_dql.py)  
**🔧 Dependency:** [PyTorch](https://pytorch.org/)

---

### 🏔️ DQN — Mountain Car

Applies Deep Q-Learning to [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) (previously solved with Q-Learning) — great for seeing how to adapt the DQN code to different environments.

**📄 Code:** [`mountain_car_dql.py`](mountain_car_dql.py)  
**🔧 Dependency:** [PyTorch](https://pytorch.org/)

---

### 🖼️ Convolutional Neural Networks (CNN) for RL

Introduces **convolutional layers** in DQNs for environments where the agent learns from **visual input** (e.g., Atari). Demonstrated on FrozenLake-v1 with image-style inputs.

**📄 Code:** [`frozen_lake_dql_cnn.py`](frozen_lake_dql_cnn.py)  
**🔧 Dependency:** [PyTorch](https://pytorch.org/)

---

## ⚡ Stable Baselines3 Tutorials

### 🚀 Getting Started — Train MuJoCo Humanoid-v4

Introduction to [Stable Baselines3](https://stable-baselines3.readthedocs.io/) using SAC, TD3, and A2C algorithms on [Humanoid-v4](https://gymnasium.farama.org/environments/mujoco/humanoid/). Includes TensorBoard monitoring.

**📄 Code:** [`sb3.py`](sb3.py)

---

### 🧭 Choosing RL Algorithms in SB3

A guide to selecting the right RL algorithm from SB3's library as a beginner.

---

### 🔄 Dynamic Algorithm Loading — Train Pendulum-v1

Makes algorithm loading dynamic and trains [Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/) with SAC and TD3 simultaneously, monitoring progress in TensorBoard.

**📄 Code:** [`sb3v2.py`](sb3v2.py)

---

### 🏆 Auto-Stop Training When Best Model Found

Walks through code that automatically stops training upon finding the best model — demonstrated on [BipedalWalker-v3](https://gymnasium.farama.org/environments/box2d/bipedal_walker/) using SAC.

**📄 Code:** [`sb3v3.py`](sb3v3.py)

---

## 📦 Dependencies Summary

| Library | Purpose |
|---|---|
| [Gymnasium](https://gymnasium.farama.org/) | RL environments |
| [PyTorch](https://pytorch.org/) | Deep Q-Networks (DQN, CNN) |
| [Stable Baselines3](https://stable-baselines3.readthedocs.io/) | Pre-built RL algorithms |

---

<p align="right">(<a href="#readme-top">back to top</a>)</p>