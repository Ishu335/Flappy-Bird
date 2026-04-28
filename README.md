# 🐦 Flappy Bird AI (Deep Q-Learning)

![Flappy Bird AI Demo](https://github.com/Ishu335/Flappy-Bird/blob/1a0bb7b9c6b2ce62a7a7ee40d8965da1aba4c03a/img/flappy_bird_rl.gif)

An AI agent that learns to play Flappy Bird using **Reinforcement Learning (Deep Q-Network - DQN)**.  
The agent improves over time by interacting with the environment and learning from rewards and penalties.

---

## 🚀 Project Overview

This project implements a **Deep Q-Learning algorithm** where the agent learns optimal actions (flap or no flap) to maximize its score.

- Starts with random actions  
- Learns from mistakes  
- Gradually improves performance  

---

## 🧠 Key Features

- ✅ Deep Q-Network (DQN) using PyTorch  
- ✅ Experience Replay for stable learning  
- ✅ Target Network Synchronization  
- ✅ Epsilon-Greedy Strategy (Exploration vs Exploitation)  
- ✅ Epsilon Decay for improved learning over time  
- ✅ Mini-batch training  
- ✅ Works on CPU, CUDA, and Apple MPS  

---

## ⚙️ Tech Stack

- Python  
- PyTorch  
- Gymnasium  
- flappy-bird-gymnasium  

---

## 📂 Project Structure
```
game  |
      ├── dqn.py
      ├── experience_replay.py
      ├── parameters.yaml
      ├── agent.py
      ├── flappy_bird_game.py
      ├── runs/
        ├── *.log
        └── *.pt   
```
