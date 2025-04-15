from agents.ppo_agent import PPOAgent
from models.actor_critic import ActorCritic
from models.vit_encoder import ViTEncoder
from common_utils import *


def run(atari_env,
        steps=1000000,
        log_path="../logs/ppo_vit_breakout.csv",
        checkpoint_path="../checkpoints/ppo_vit.pt"):

    env = make_atari_env(atari_env)
    obs_space = env.observation_space.shape
    num_actions = env.action_space.n

    print(f"Obs space: {obs_space}, Actions: {num_actions}")

    encoder = ViTEncoder(
        img_size=84,
        patch_size=6,
        in_channels=4,
        emb_dim=512,
        depth=4,
        num_heads=4,
        dropout=0.1
    )
    model = ActorCritic(encoder, feature_dim=512, num_actions=num_actions)
    model.to(device)

    agent = PPOAgent(
        model=model,
        env=env,
        lr=2.5e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        k_epochs=4,
        batch_size=64,
        total_timesteps=steps
    )

    agent.train(preprocess, log_path=log_path)

    env.close()

if __name__ == "__main__":
    run(steps=500000, log_path='../logs/ppo_vit_breakout_500k_iterations.csv')
