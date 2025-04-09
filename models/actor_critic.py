import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_actions: int):
        """
        Args:
            encoder: a feature extractor (CNN or ViT)
            feature_dim: output dimension of the encoder (e.g. 512)
            num_actions: size of the discrete action space
        """
        super().__init__()
        self.encoder = encoder
        self.policy_head = nn.Linear(feature_dim, num_actions)
        self.value_head = nn.Linear(feature_dim, 1)

    def forward(self, x):
        """
        Args:
            x: input image tensor [B, C, H, W]
        Returns:
            action_logits: [B, num_actions]
            state_values: [B, 1]
        """
        features = self.encoder(x)
        action_logits = self.policy_head(features)
        state_values = self.value_head(features)
        return action_logits, state_values
