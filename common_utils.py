import torch
import numpy as np
import os
import gymnasium as gym
import random
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
import ale_py


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # ensures deterministic conv ops
    torch.backends.cudnn.benchmark = False     # disables autotuner


def make_atari_env(env_name="ALE/Breakout-v5", render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)

    env = AtariPreprocessing(
        env,
        screen_size=84,
        frame_skip=1,
        grayscale_obs=True,
        terminal_on_life_loss=False
    )
    env = FrameStackObservation(env, stack_size=4)

    return env


def preprocess(obs):
    obs = np.array(obs)
    obs = torch.tensor(obs, dtype=torch.float32) / 255.0
    return obs.unsqueeze(0).to(device)