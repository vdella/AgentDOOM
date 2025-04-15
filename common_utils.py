import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
import ale_py


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_atari_env(env_name="ALE/Breakout-v5"):
    env = gym.make(env_name, render_mode=None)

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
    obs = np.array(obs)
    obs = torch.tensor(obs, dtype=torch.float32) / 255.0
    return obs.unsqueeze(0).to(device)