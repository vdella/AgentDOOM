import torch
import torch.nn as nn
from einops import rearrange


class ViTEncoder(nn.Module):
    def __init__(self, img_size=84, patch_size=8, in_channels=4, emb_dim=256, depth=6, num_heads=4, dropout=0.1):
        super(ViTEncoder, self).__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size

        self.patch_embed = nn.Linear(self.patch_dim, emb_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embed(x)  # [B, num_patches, emb_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, emb_dim]
        x += self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        return x[:, 0]  # Return the CLS token as global feature
