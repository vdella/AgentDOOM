import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
from collections import Counter
import csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def compute_gae(rewards, values, is_terminals, gamma=0.99, lam=0.95):
    """Used in PPO, the Generalized Advantage Estimation (GAE) presents
    a bias-invariance tradeoff in policy gradients. works
    as the advantage estimation formula."""

    advantages = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - is_terminals[t]) - values[t]
        gae = delta + gamma * lam * (1 - is_terminals[t]) * gae
        advantages.insert(0, gae)
    return advantages

class PPOAgent:
    def __init__(self, model, env, lr=2.5e-4, gamma=0.99, lam=0.95, clip_eps=0.2, k_epochs=4, batch_size=64, total_timesteps=1e6):
        self.model = model
        self.env = env
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.total_timesteps = int(total_timesteps)

    def select_action(self, state):
        """Returns an action index, the log probability of that said action and the
        estimated state-value associated with it. Runs policy without gradients."""

        with torch.no_grad():
            logits, value = self.model(state)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    def train(self, preprocess, log_path, save_every=204800, checkpoint_path="../checkpoints/"):
        buffer = RolloutBuffer()
        state, _ = self.env.reset()
        ep_rewards = deque(maxlen=100)  # Keeps the last 100 episode rewards.
        episode_reward = 0

        for t in range(self.total_timesteps):
            # Preprocess, step-in and checking.
            state_tensor = preprocess(state)
            action, log_prob, value = self.select_action(state_tensor)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Buffer storing and episode tracking.
            buffer.states.append(state)
            buffer.actions.append(action)
            buffer.logprobs.append(log_prob)
            buffer.values.append(value)
            buffer.rewards.append(reward)
            buffer.is_terminals.append(done)

            state = next_state
            episode_reward += reward

            if done:
                # Records reward from the episode.
                state, _ = self.env.reset()
                ep_rewards.append(episode_reward)
                episode_reward = 0

            # Training phase.
            if (t + 1) % 2048 == 0:  # 2048 is standard for low-computing envs, such as Atari (OpenAI baseline).

                advantages = compute_gae(  # Computes returns and advantages.
                    buffer.rewards, buffer.values, buffer.is_terminals, self.gamma, self.lam
                )
                returns = [adv + val for adv, val in zip(advantages, buffer.values)]

                # Converts buffers to tensors and normalizes advantages.
                states = torch.stack([preprocess(s).squeeze(0) for s in buffer.states]).to(device)
                actions = torch.LongTensor(buffer.actions).to(device)
                old_logprobs = torch.FloatTensor(buffer.logprobs).to(device)
                returns = torch.FloatTensor(returns).to(device)
                advantages = torch.FloatTensor(advantages).to(device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                for _ in range(self.k_epochs):  # k_epochs over mini-batches.

                    # Computes PPO clipped objective and entropy bonus.
                    for i in range(0, len(states), self.batch_size):

                        # Gets batch's variables.
                        end = i + self.batch_size
                        batch_states = states[i:end]
                        batch_actions = actions[i:end]
                        batch_old_logprobs = old_logprobs[i:end]
                        batch_returns = returns[i:end]
                        batch_advantages = advantages[i:end]

                        logits, values = self.model(batch_states)

                        # Creates a categorical distribution over discrete actions using logits, and...
                        # ... Is equivalent to softmax(logits).
                        dist = torch.distributions.Categorical(logits=logits)
                        new_logprobs = dist.log_prob(batch_actions)

                        # An attempt to encourage exploration. Overconfident policies are penalized.
                        entropy = dist.entropy().mean()

                        # From PPO's main paper, gets the ratio for the objective functions.
                        ratio = torch.exp(new_logprobs - batch_old_logprobs)
                        unclipped_objective = ratio * batch_advantages
                        clipped_surrogate = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages

                        # Takes the negative of the clipped objective, as we want to maximize it.
                        policy_loss = -torch.min(unclipped_objective, clipped_surrogate).mean()

                        # Applies MSE between values and returns.
                        value_loss = F.mse_loss(values.squeeze(), batch_returns)

                        # Calculate the total loss for the episode batch.
                        loss = policy_loss + 0.5 * value_loss - 0.05 * entropy

                        # Clears gradients, backpropagates the loss, ...
                        self.optimizer.zero_grad()
                        loss.backward()

                        # ... Stores locals(), ...
                        last_entropy = entropy.item()
                        last_value_loss = value_loss.item()

                        # ... And steps.
                        self.optimizer.step()

                buffer.clear()  # Needs to clear buffer for the next round.
                avg_reward = np.mean(ep_rewards) if ep_rewards else 0

                # Checks locals() for printing purposes.
                if 'last_entropy' in locals() and 'last_value_loss' in locals():
                    print(f"Step: {t+1}, AvgReward: {avg_reward:.4f}, Entropy: {last_entropy:.4f}, ValueLoss: {last_value_loss:.4f}")
                else:
                    print(f"Step: {t+1}, AvgReward: {avg_reward:.4f}")

                # Shows action distribution.
                action_counts = Counter(buffer.actions)
                total_actions = sum(action_counts.values())
                for act in sorted(action_counts):
                    pct = 100.0 * action_counts[act] / total_actions
                    print(f"  Action {act}: {pct:.1f}%")

                # Log reward to CSV
                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([t + 1, avg_reward])

                # Save checkpoint
                if (t + 1) % save_every == 0:
                    ckpt_path = checkpoint_path + f"ckpt_{t + 1}.pt"
                    torch.save(self.model.state_dict(), ckpt_path)

        # Save final model
        ckpt_path = checkpoint_path + f"final.pt"
        torch.save(self.model.state_dict(), ckpt_path)


if __name__ == '__main__':
    print("Using device:", device)
