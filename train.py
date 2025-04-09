import gymnasium
import ale_py


env = gymnasium.make("ALE/Tetris-v5", render_mode="human")
epochs = 10000


def raw_exec():
    """Shows the execution of the rawest version
    of the Tetris game -- without PPO-CNNs
    and PPO-ViTs."""

    env.reset()

    for _ in range(epochs):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    raw_exec()