import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Embeds images patches into vectors."""

    def __init__(self, in_channels=4, patch_size=6, emb_dim=256, img_size=84):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] -> [B, emb_dim, H', W'].
        x = x.flatten(2)  # [B, emb_dim, H', W'] -> [B, emb_dim, N].
        x = x.transpose(1, 2)  # [B, emb_dim, N] -> [B, N, emb_dim].
        return x


class ViTEncoder(nn.Module):
    def __init__(self, img_size=84, patch_size=6, in_channels=4, emb_dim=256, depth=6, num_heads=4, dropout=0.1):
        super().__init__()

        # Starts with [B, 4, 84, 84]. Take B as the batch_size.
        # The math here works around the num_patches. These have the following form;
        # num_patches = (image_height / patch_size) * (image_width / patch_size).

        self.patch_embed = PatchEmbed(in_channels, patch_size, emb_dim, img_size)
        num_patches = self.patch_embed.n_patches  # [B, 4, 84, 84] -> (84 / 6) = 14 -> 14*14 = 196 -> [B, 196, 256].

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))  # Class token for global info.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))  # For all tokens, including [CLS].
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, 4, 84, 84] -> [B, 196, 256].

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 256].
        x = torch.cat((cls_tokens, x), dim=1)  # Adds [CLS] at the beggining of the seq. -> [B, 196 + 1, 256].
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.dropout(x)

        x = self.transformer(x)  # [B, 196+1, 256].
        x = self.norm(x)

        return x[:, 0]


if __name__ == "__main__":
    # Example usage
    vit = ViTEncoder()
    x = torch.randn(2, 4, 84, 84)  # Batch of 2 images
    out = vit(x)
    print(out.shape)  # Should be [2, 256]