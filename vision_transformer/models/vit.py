import torch
import torch.nn as nn

from .encoder_block import EncoderBlock
from .input_layer import InputLayer


class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        patch_size: int,
        image_size: int,
        num_blocks: int,
        nb_head: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_layer = InputLayer(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            image_size=image_size,
        )
        self.encoder = nn.Sequential(
            *[
                EncoderBlock(
                    embed_dim=embed_dim,
                    nb_head=nb_head,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, W, H) -> (B, N, D)
        out = self.input_layer(x)
        # (B, N, D) -> (B, N, D)
        out = self.encoder(out)
        # extract only class token
        # (B, N, D) -> (B, D)
        cls_token = out[:, 0]
        # (B, D) -> (B, M)
        pred = self.mlp_head(cls_token)
        return pred
