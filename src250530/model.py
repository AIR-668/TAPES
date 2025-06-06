from modules import *
class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
        ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
        nn.Dropout(0.1),
        ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        nn.Dropout(0.1),
        ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


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


class CausalSetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False, use_causal=False):
        super(CausalSetTransformer, self).__init__()
        self.use_causal = use_causal
        
        # Encoder stack
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            nn.Dropout(0.1),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            nn.Dropout(0.1),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        
        # Optional causal attention module (acts on sample dimension)
        self.causal_attn = CausalAttention(dim_hidden, num_heads) if use_causal else None
        
        # Decoder stack
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, X):
        X = self.enc(X)  # shape: (B, N, D)
        if self.use_causal:
            X = self.causal_attn(X)  # apply causal attention across samples
        return self.dec(X)

class ExpressionKnowledgeFusion(nn.Module):
    def __init__(self, dim_input, dim_k):
        super().__init__()
        self.dim_input = dim_input
        self.dim_k = dim_k

        # 1. Project expression to key/value
        self.kv_proj = nn.Linear(1, dim_k * 2)

        # 2. FiLM MLP
        self.film_mlp = nn.Sequential(
            nn.Linear(dim_k, dim_k),
            nn.ReLU(),
            nn.Linear(dim_k, dim_k * 2)  # gamma, beta
        )

        # 3. Project raw expression for gating
        self.expr_proj = nn.Linear(dim_input, dim_k)

        # 4. Gating MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim_k * 2, dim_k),
            nn.ReLU(),
            nn.Linear(dim_k, dim_k),
            nn.Sigmoid()
        )

    def forward(self, x_expr, k_embed):
        """
        x_expr: (B, G) expression vector
        k_embed: (G, d_k) knowledge embedding
        Returns:
            h_final: (B, G, d_k) fused representation
        """
        B, G = x_expr.shape
        d_k = self.dim_k

        # Step 1: Cross Attention
        x_proj = x_expr.unsqueeze(-1)        # (B, G, 1)
        kv = self.kv_proj(x_proj)            # ✅ (B, G, 2*d_k)
        k, v = kv.chunk(2, dim=-1)           # (B, G, d_k)
        q = k_embed.unsqueeze(0).expand(B, -1, -1)  # (B, G, d_k)


        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  # (B, G, G)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, G, G)
        h_i = torch.matmul(attn_weights, v)  # (B, G, d_k)

        # Step 2: FiLM
        gamma_beta = self.film_mlp(k_embed)  # (G, 2*d_k)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(0)  # (1, G, d_k)
        beta = beta.unsqueeze(0)
        h_tilde = gamma * h_i + beta  # (B, G, d_k)

        # Step 3: Gating
        x_proj_flat = self.expr_proj(x_expr)  # (B, d_k)
        x_proj = x_proj_flat.unsqueeze(1).expand(-1, G, -1)  # (B, G, d_k)
        concat = torch.cat([h_tilde, x_proj], dim=-1)  # (B, G, 2*d_k)
        gate = self.gate_mlp(concat)  # (B, G, d_k)

        # Final output: gated fusion
        h_final = gate * h_tilde + (1 - gate) * x_proj  # (B, G, d_k)
        return h_final
