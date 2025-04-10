import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
import ale_py

from models.vit_encoder import ViTEncoder
from models.actor_critic import ActorCritic
from agents.ppo_agent import PPOAgent


def make_tetris_env():
    env = gym.make("ALE/Tetris-v5", render_mode=None)

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
    return obs.unsqueeze(0)


def run(log_path="../logs/ppo_vit.csv"):
    env = make_tetris_env()
    num_actions = env.action_space.n

    encoder = ViTEncoder(
        img_size=84,
        patch_size=6,
        in_channels=4,
        emb_dim=256,
        depth=4,
        num_heads=4,
        dropout=0.1
    )
    model = ActorCritic(encoder, feature_dim=256, num_actions=num_actions)

    agent = PPOAgent(
        model=model,
        env=env,
        total_timesteps=10000  # test run
    )

    agent.train(preprocess, log_path=log_path, save_every=10000)

    env.close()

if __name__ == "__main__":
    run()
