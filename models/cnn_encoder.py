import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()

        # The initial tensor is of shape [B, 4, 84, 84]. Take B as the batch_size.
        # The general formulae is: out_size = (in_size + 2*padding - kernel_size) / stride + 1.
        # If not given, the padding is assumed to be 0.

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=8,
                      stride=4),  # (84 in - 8 kernel) / 4 stride = 19 -> 19 + 1 = 20 -> [32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),  # (20 in - 4 kernel) / 2 stride = 8 -> 8 + 1 = 9 -> [64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),  # (9 in - 3 kernel) / 1 stride = 6 -> 6 + 1 = 7 -> [64, 7, 7]
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()  # Takes [B, n1, n2, n3, ..., nt] and flattens it to [B, n1*n2*...*nt]
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),  # Maps 3136 to a 512 dimension vector.
            nn.ReLU()
        )

    def forward(self, x):
        """Gets a tensor of shape [batch_size, 4, 84, 84]
        Returns a tensor of shape [batch_size, 512]."""

        x = self.conv_layers(x)  # [B, 4, 84, 84] -> [B, 64, 7, 7].
        x = self.flatten(x)  # [B, 64, 7, 7] -> [B, 64*7*7].
        x = self.fc(x)  # [B, 64*7*7] -> [B, 512].
        return x


if __name__ == "__main__":
    encoder = CNNEncoder()
    dummy_input = torch.randn(2, 4, 84, 84)  # batch of 2
    out = encoder(dummy_input)
    print(out.shape)  # Should be [2, 512].
