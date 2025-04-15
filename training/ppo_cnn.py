from agents.ppo_agent import PPOAgent
from models.actor_critic import ActorCritic
from models.cnn_encoder import CNNEncoder


from common_utils import *


def run(atari_env="ALE/Breakout-v5",
        steps=1000000,
        log_path="../logs/ppo_cnn.csv",
        checkpoint_path="../checkpoints/ppo_cnn.pt"):

    env = make_atari_env(atari_env)
    obs_space = env.observation_space.shape
    num_actions = env.action_space.n

    print(f"Obs space: {obs_space}, Actions: {num_actions}")

    encoder = CNNEncoder()
    model = ActorCritic(encoder, feature_dim=512, num_actions=num_actions)
    model.to(device)

    agent = PPOAgent(
        model=model,
        env=env,
        lr = 1e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.3,
        k_epochs=4,
        batch_size=64,
        total_timesteps=steps
    )

    agent.train(preprocess, log_path=log_path, checkpoint_path=checkpoint_path)

    env.close()


if __name__ == "__main__":
    run(atari_env='ALE/Pong-v5',
        steps=500000,
        log_path='../logs/ppo_cnn_pong_500k_iterations.csv',
        checkpoint_path='../checkpoints/ppo_cnn/pong/500k/')

    run(atari_env='ALE/Pong-v5',
        steps=1000000,
        log_path='../logs/ppo_cnn_pong_1mi_iterations.csv',
        checkpoint_path='../checkpoints/ppo_cnn/pong/1mi/')

    run(atari_env='ALE/Pong-v5',
        steps=2000000,
        log_path='../logs/ppo_cnn_pong_2mi_iterations.csv',
        checkpoint_path='../checkpoints/ppo_cnn/pong/2mi/')

    run(atari_env='ALE/Breakout-v5',
        steps=500000,
        log_path='../logs/ppo_cnn_breakout_500k_iterations.csv',
        checkpoint_path='../checkpoints/ppo_cnn/breakout/500k/')

    run(atari_env='ALE/Breakout-v5',
        steps=1000000,
        log_path='../logs/ppo_cnn_breakout_1mi_iterations.csv',
        checkpoint_path='../checkpoints/ppo_cnn/breakout/1mi/')

    run(atari_env='ALE/Breakout-v5',
        steps=2000000,
        log_path='../logs/ppo_cnn_breakout_2mi_iterations.csv',
        checkpoint_path='../checkpoints/ppo_cnn/breakout/2mi/')

