import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from models.vit_encoder import ViTEncoder
import cv2


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear(self):
        self.__init__()


def compute_gae(rewards, values, is_terminals, gamma, lam):
    advantages = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - is_terminals[t]) - values[t]
        gae = delta + gamma * lam * (1 - is_terminals[t]) * gae
        advantages.insert(0, gae)
    return advantages


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, num_actions, encoder_cfg):
        super(ActorCritic, self).__init__()
        self.encoder = ViTEncoder(**encoder_cfg)
        self.policy_head = nn.Linear(encoder_cfg['emb_dim'], num_actions)
        self.value_head = nn.Linear(encoder_cfg['emb_dim'], 1)

    def forward(self, x):
        features = self.encoder(x)
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        return action_logits, value


class ViTPPOAgent:
    def __init__(self, env, encoder_cfg=None, lr=2.5e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, k_epochs=4, batch_size=64, total_timesteps=1e6):
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.total_timesteps = int(total_timesteps)

        if encoder_cfg is None:
            encoder_cfg = {
                'img_size': 84,
                'patch_size': 8,
                'in_channels': 4,
                'emb_dim': 256,
                'depth': 6,
                'num_heads': 4,
                'dropout': 0.1
            }

        self.model = ActorCritic(self.obs_shape, self.num_actions, encoder_cfg)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size


    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0) / 255.0  # Normalize pixels
        with torch.no_grad():
            logits, _ = self.model(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1)
        return action.item(), probs[0, action.item()].item()

    def train(self, render=False):
        buffer = RolloutBuffer()
        state, _ = self.env.reset()
        ep_rewards = deque(maxlen=100)
        episode_reward = 0

        for t in range(self.total_timesteps):
            action, log_prob, value = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            if render:
                frame = self.env.render()
                if frame is not None:
                    cv2.imshow("Agent View", frame)
                    cv2.waitKey(1)  # ~60 FPS

            buffer.states.append(state)
            buffer.actions.append(action)
            buffer.logprobs.append(log_prob)
            buffer.values.append(value)
            buffer.rewards.append(reward)
            buffer.is_terminals.append(done)

            state = next_state
            episode_reward += reward

            if done:
                state, _ = self.env.reset()
                if render:
                    cv2.waitKey(500)  # Pause briefly between episodes
                ep_rewards.append(episode_reward)
                episode_reward = 0

            if (t + 1) % 2048 == 0:
                advantages = compute_gae(buffer.rewards, buffer.values, buffer.is_terminals, self.gamma, self.lam)
                returns = [adv + val for adv, val in zip(advantages, buffer.values)]

                states = torch.FloatTensor(np.array(buffer.states)) / 255.0
                actions = torch.LongTensor(buffer.actions)
                old_logprobs = torch.FloatTensor(buffer.logprobs)
                returns = torch.FloatTensor(returns)
                advantages = torch.FloatTensor(advantages)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                for _ in range(self.k_epochs):
                    for i in range(0, len(states), self.batch_size):
                        end = i + self.batch_size
                        batch_states = states[i:end]
                        batch_actions = actions[i:end]
                        batch_old_logprobs = old_logprobs[i:end]
                        batch_returns = returns[i:end]
                        batch_advantages = advantages[i:end]

                        logits, values = self.model(batch_states)
                        dist = torch.distributions.Categorical(logits=logits)
                        entropy = dist.entropy().mean()
                        new_logprobs = dist.log_prob(batch_actions)

                        ratio = torch.exp(new_logprobs - batch_old_logprobs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = F.mse_loss(values.squeeze(), batch_returns)
                        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                buffer.clear()
                avg_reward = np.mean(ep_rewards) if ep_rewards else 0
                print(f"Step: {t+1}, Average Reward (100 ep): {avg_reward:.2f}")
