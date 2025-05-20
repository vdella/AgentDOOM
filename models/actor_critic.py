import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_actions: int):
        super().__init__()

        # The input is of shape [B, 4, 84, 84]. Take B as the batch_size.

        self.encoder = encoder
        self.policy_head = nn.Linear(feature_dim, num_actions)
        self.value_head = nn.Linear(feature_dim, 1)

    def forward(self, x):
        features = self.encoder(x)  # [B, C, H, W]
        action_logits = self.policy_head(features)  # [B, C, H, W] -> [B, num_actions]
        state_values = self.value_head(features)  # [B, C, H, W] -> [B, 1].
        return action_logits, state_values  # [B, num_actions], [B, 1].
