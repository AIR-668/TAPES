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

        self.causal_mask = nn.Parameter(torch.randn(num_heads, dim, dim))

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.sigmoid(self.causal_mask).unsqueeze(0).expand(B, -1, -1, -1)
        attn_scores = attn_scores * mask[:, :, :N, :N]

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
        x = self.pma(x)
        t = self.regressor(x.squeeze(1))
        return t.squeeze(-1)


# Loss functions

def causal_pairwise_rank_loss(y_true, y_pred):
    diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    indicator = (y_true.unsqueeze(1) < y_true.unsqueeze(0)).float()
    loss_matrix = indicator * F.softplus(-diff)
    return loss_matrix.sum() / (indicator.sum() + 1e-8)


def spearman_loss(y_true, y_pred):
    rank_true = y_true.argsort().argsort().float()
    rank_pred = y_pred.argsort().argsort().float()
    corr = torch.corrcoef(torch.stack([rank_true, rank_pred]))[0, 1]
    return 1 - corr


def mse_loss(y_true, y_pred):
    return F.mse_loss(y_pred, y_true)


def total_loss(y_true, y_pred, lambdas=(1.0, 1.0, 0.1, 0.5)):
    l_rank = causal_pairwise_rank_loss(y_true, y_pred)
    l_spear = spearman_loss(y_true, y_pred)
    l_perm = torch.tensor(0.0)  # Optional: Add permutation regularization later
    l_mse = mse_loss(y_true, y_pred)
    return lambdas[0]*l_rank + lambdas[1]*l_spear + lambdas[2]*l_perm + lambdas[3]*l_mse

