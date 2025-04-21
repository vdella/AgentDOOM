from agents.ppo_agent import PPOAgent
from models.actor_critic import ActorCritic
from models.cnn_encoder import CNNEncoder
from PIL import Image

from common_utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_gif(checkpoint_path, gif_name="../gifs/ppo_breakout.gif"):
    encoder = CNNEncoder()
    model = ActorCritic(encoder, feature_dim=512, num_actions=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Environment
    env = make_atari_env(render_mode="rgb_array")
    obs, _ = env.reset()
    done = False
    frames = []

    while not done:
        obs_tensor = preprocess(obs)

        with torch.no_grad():
            logits, _ = model(obs_tensor)
            action = torch.argmax(logits, dim=-1).item()

        obs, reward, done, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(Image.fromarray(frame))

        if done or truncated:
            break

    frames[0].save(
        gif_name,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0
    )
    print(f"GIF saved as {gif_name}")


if __name__ == "__main__":
    generate_gif("../checkpoints/ppo_cnn/breakout/lr_1e-4_clip_01/2mi/final.pt")
