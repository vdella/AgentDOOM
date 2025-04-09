import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation

from models.cnn_encoder import CNNEncoder
from models.actor_critic import ActorCritic
import ale_py

import csv
import os

from agents.ppo_agent import PPOAgent


def make_tetris_env():
    env = gym.make("ALE/Tetris-v5", render_mode="human")

    env = AtariPreprocessing(
        env,
        screen_size=84,
        frame_skip=1,
        grayscale_obs=True,
        terminal_on_life_loss=True
    )
    env = FrameStackObservation(env, stack_size=4)

    return env


def preprocess(obs):
    """Convert LazyFrames to a normalized batched tensor."""

    obs = np.array(obs)  # LazyFrames → ndarray
    obs = torch.tensor(obs, dtype=torch.float32) / 255.0
    return obs.unsqueeze(0)  # [1, 4, 84, 84]


def run(steps=1000):
    env = make_tetris_env()
    obs_space = env.observation_space.shape
    num_actions = env.action_space.n

    print(f"Obs space: {obs_space}, Actions: {num_actions}")

    encoder = CNNEncoder()
    model = ActorCritic(encoder, feature_dim=512, num_actions=num_actions)

    agent = PPOAgent(
        model=model,
        env=env,
        lr=2.5e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        k_epochs=4,
        batch_size=64,
        total_timesteps=10_000
    )

    agent.train(preprocess)

    env.close()


if __name__ == "__main__":
    run()
