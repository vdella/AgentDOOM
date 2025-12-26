# AgentDOOM

**AgentDOOM** is a reinforcement learning research project focused on training PPO (Proximal Policy Optimization) agents to play **Atari Breakout**, comparing different visual encoders such as **CNNs** and **Vision Transformers (ViTs)**.

---

## Overview

- Actor-Critic architecture using PPO
- Visual state encoding with CNN and ViT backbones
- Experimental setup for encoder comparison in Atari environments

---

## Repository Structure

```text
AgentDOOM/
├── agents/        # CNN and ViT agent definitions
├── training/      # Training and evaluation scripts
├── notebooks/     # Experimental notebooks
├── models/        # Saved checkpoints
├── logs/          # Training logs and metrics
├── gifs/          # Gameplay visualizations
├── requirements.txt
├── pyproject.toml
└── README.md
