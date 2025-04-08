from agents.vit_ppo_agent import ViTPPOAgent
from envs.make_env import make_env

def main():
    env_id = "PongNoFrameskip-v4"
    env = make_env(env_id, render=True)

    agent = ViTPPOAgent(env, total_timesteps=10_000)  # Short test run
    print("Environment and agent initialized. Starting training...")
    agent.train(render=True)

if __name__ == "__main__":
    main()

