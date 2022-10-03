import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, nb_head: int, dropout: float) -> None:
        super().__init__()
        self.nb_head = nb_head
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // nb_head

        self.w_q = nn.Linear(
            in_features=self.embed_dim, out_features=self.embed_dim, bias=False
        )
        self.w_k = nn.Linear(
            in_features=self.embed_dim, out_features=self.embed_dim, bias=False
        )
        self.w_v = nn.Linear(
            in_features=self.embed_dim, out_features=self.embed_dim, bias=False
        )
        self.w_o = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, nb_patch, _ = x.size()
        # (B, N, D) -> (B, nb_head, N, D//nb_head)
        # query
        q = self.w_q(x)
        q = q.view(batch_size, self.nb_head, nb_patch, self.head_dim)
        # key
        k = self.w_k(x)
        k = k.view(batch_size, self.nb_head, nb_patch, self.head_dim)
        # value
        v = self.w_v(x)
        v = v.view(batch_size, self.nb_head, nb_patch, self.head_dim)

        # inner product
        # (B, nb_head, N, D//nb_head) × (B, nb_head, D//nb_head, N) -> (B, nb_head, N, N)
        dots = (q @ k.transpose(2, 3)) / self.head_dim**0.5
        # softmax by columns
        # dim=3 eq dim=-1. dim=-1 applies softmax to the last dimension
        attn = F.softmax(dots, dim=3)
        # weighted
        # (B, nb_head, N, N) × (B, nb_head, N, D//nb_head) -> (B, nb_head, N, D//nb_head)
        out = attn @ v
        # (B, nb_head, N, D//nb_head) -> (B, N, nb_head, D//nb_head) -> (B, N, D)
        out = out.transpose(1, 2).reshape(batch_size, nb_patch, self.embed_dim)
        out = self.w_o(out)
        return out
