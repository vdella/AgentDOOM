import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation

from models.cnn_encoder import CNNEncoder
from models.actor_critic import ActorCritic
import ale_py


def make_tetris_env():
    env = gym.make("ALE/Tetris-v5", render_mode="human")

    # Apply grayscale, resize, frame skip, and frame stacking
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
    """
    Convert LazyFrames to a normalized batched tensor.
    """
    obs = np.array(obs)  # LazyFrames → ndarray
    obs = torch.tensor(obs, dtype=torch.float32) / 255.0
    return obs.unsqueeze(0)  # [1, 4, 84, 84]


def run(steps=1000):
    env = make_tetris_env()
    obs, _ = env.reset()

    # Setup CNN + ActorCritic model
    encoder = CNNEncoder()
    num_actions = env.action_space.n
    model = ActorCritic(encoder, feature_dim=512, num_actions=num_actions)
    model.eval()

    for step in range(steps):
        obs_tensor = preprocess(obs)

        with torch.no_grad():
            logits, value = model(obs_tensor)

        print(f"Step {step}")
        print(" - Action logits shape:", logits.shape)
        print(" - Value estimate shape:", value.shape)

        # Pick action greedily (max logit)
        action = torch.argmax(logits, dim=-1).item()

        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    run()
