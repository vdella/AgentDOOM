import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation

def make_env(env_id, render=False):
    mode = "rgb_array" if render else None
    env = gym.make(env_id, render_mode=mode)

    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=True)
    env = FrameStackObservation(env, stack_size=4)

    return env
