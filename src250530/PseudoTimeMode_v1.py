# Causal Masked Set Transformer for PseudoTime Inference (ANLN-supervised)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

        # Learnable causal mask [heads, features, features]
        self.causal_mask = nn.Parameter(torch.randn(num_heads, dim, dim))

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv_proj(x)  # (B, N, 3*D)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, N, N)

        # Apply causal mask (learnable, sigmoid scaled)
        mask = torch.sigmoid(self.causal_mask).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, D, D)
        attn_scores = attn_scores * mask[:, :, :N, :N]  # Apply only to valid size

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class SAB(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = CausalAttention(dim, num_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.attn = CausalAttention(dim, num_heads)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.size(0)
        seed = self.seed_vectors.expand(B, -1, -1)
        x = self.attn(self.ln(seed), x)
        return x


class PseudoTimeModel(nn.Module):
    def __init__(self, input_dim, dim=128, num_heads=4):
        super().__init__()
        self.embed = nn.Linear(input_dim, dim)
        self.encoder = nn.Sequential(
            SAB(dim, num_heads),
            SAB(dim, num_heads)
        )
        self.pma = PMA(dim, num_heads, num_seeds=1)
        self.regressor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = self.pma(x)  # Output shape: (B, 1, D)
        t = self.regressor(x.squeeze(1))  # Output: (B, 1) => (B)
        return t.squeeze(-1)
