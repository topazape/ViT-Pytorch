import torch
import torch.nn as nn

from .multihead_attention import MultiHeadSelfAttention


class EncoderBlock(nn.Module):
    def __init__(
        self, embed_dim: int, nb_head: int, hidden_dim: int, dropout: float
    ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.msa = MultiHeadSelfAttention(
            embed_dim=embed_dim, nb_head=nb_head, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add skip-connect
        out = self.msa(self.ln1(x)) + x
        # add skip-connect
        out = self.mlp(self.ln2(out)) + out
        return out
