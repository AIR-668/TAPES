# Trian带有因果模块&因果loss的模型
# 支持 Apptainer 部署、wandb 可视化、加入真实 LLM encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import wandb
import os

# === WANDB Init ===
wandb.init(project="GeneLLM2Time", name="pseudotime-train")

# === Dummy dataset for pseudotime ===
class PseudoTimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Rank loss (pairwise) ===
def rank_loss(y_pred, y_true):
    loss = 0.0
    n = y_pred.size(0)
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] < y_true[j]:
                loss += torch.log(1 + torch.exp(-(y_pred[j] - y_pred[i])))
    return loss / (n * (n - 1))

# === Spearman correlation loss ===
def spearman_loss(y_pred, y_true):
    pred_rank = y_pred.argsort().argsort().float()
    true_rank = y_true.argsort().argsort().float()
    pred_rank = (pred_rank - pred_rank.mean()) / pred_rank.std()
    true_rank = (true_rank - true_rank.mean()) / true_rank.std()
    return 1 - torch.cosine_similarity(pred_rank, true_rank, dim=0)

# === Sinkhorn regularization (placeholder) ===
def sinkhorn_regularization(P):
    return torch.norm(P @ P.T - torch.eye(P.size(0)).to(P.device))

# === Real BiomedBERT encoder wrapper ===
class BiomedEncoder(nn.Module):
    def __init__(self, model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, texts):
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        output = self.model(**encoded)
        return output.last_hidden_state[:, 0, :]  # CLS token

# === Full model ===
class CausalSetToSequence(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.permutation_net = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=2
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.encoder(x)
        P = torch.softmax(self.permutation_net(h), dim=-1)
        h_perm = torch.matmul(P, h)
        h_seq = self.decoder(h_perm.unsqueeze(1)).squeeze(1)
        out = self.head(h_seq).squeeze(-1)
        return out, P

# === Training loop ===
def train_model():
    # Dummy data
    X = torch.randn(100, 16)
    y = torch.sort(torch.rand(100))[0] * 10

    dataset = PseudoTimeDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = CausalSetToSequence(input_dim=16, hidden_dim=32).to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(model.encoder.weight.device), y_batch.to(model.encoder.weight.device)
            y_pred, P = model(x_batch)

            loss_rank = rank_loss(y_pred, y_batch)
            loss_spearman = spearman_loss(y_pred, y_batch).mean()
            loss_mse = F.mse_loss(y_pred, y_batch)
            loss_perm = sinkhorn_regularization(P)

            loss = 1.0 * loss_rank + 1.0 * loss_spearman + 0.5 * loss_mse + 0.1 * loss_perm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            wandb.log({"rank_loss": loss_rank.item(), "spearman_loss": loss_spearman.item(),
                       "mse_loss": loss_mse.item(), "perm_loss": loss_perm.item(), "total_loss": loss.item()})

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    os.environ['WANDB_MODE'] = 'online'  # or 'offline' if needed
    train_model()
