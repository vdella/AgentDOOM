import gymnasium
import ale_py


def main():
    env = gymnasium.make("ALE/Pong-v5", render_mode="human")
    env.reset()

    for _ in range(100):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()