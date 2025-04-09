import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),  # -> [32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> [64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> [64, 7, 7]
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),  # 3136 → 512
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, 4, 84, 84]
        """
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x  # shape: [batch_size, 512]


if __name__ == "__main__":
    encoder = CNNEncoder()
    dummy_input = torch.randn(2, 4, 84, 84)  # batch of 2
    out = encoder(dummy_input)
    print(out.shape)  # should be [2, 512]
