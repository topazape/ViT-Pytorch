import torch
import torch.nn as nn


class InputLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        image_size: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size

        # number of patches
        self.nb_patch = (self.image_size // self.patch_size) ** 2
        # split into patches
        self.patch_embed_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        # cls token
        self.cls_token = nn.parameter.Parameter(data=torch.randn(1, 1, self.embed_dim))
        # positional embedding
        self.positional_embedding = nn.parameter.Parameter(
            data=torch.randn(1, self.nb_patch + 1, self.embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch, Channel, Height, Width) -> (B, D, H/P, W/P)
        out = self.patch_embed_layer(x)
        # (B, D, H/P, W/P) -> (B, D, Np)
        # flatten from H/P(2) to W/P(3)
        out = torch.flatten(out, start_dim=2, end_dim=3)
        # (B, D, Np) -> (B, Np, D)
        out = out.transpose(1, 2)
        # concat class token
        # cat (B, 1, D), (B, Np, D) -> (B, Np + 1, D)
        out = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), out], dim=1)
        # add positional embedding
        out += self.positional_embedding
        return out
