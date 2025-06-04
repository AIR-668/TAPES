import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import wandb
import os

# === WANDB Init ===
wandb.init(project="GeneLLM2Time", name="pseudotime-train")

# === Dataset with column names + numerical features ===
class TableDataset(Dataset):
    def __init__(self, col_texts, numeric_features, pseudotime_labels):
        self.col_texts = col_texts  # list of column descriptions (strings)
        self.numeric_features = numeric_features  # tensor of shape [N, D]
        self.labels = pseudotime_labels  # tensor of shape [N]

    def __len__(self):
        return len(self.numeric_features)

    def __getitem__(self, idx):
        return self.col_texts[idx], self.numeric_features[idx], self.labels[idx]

# === Losses ===
def rank_loss(y_pred, y_true):
    loss = 0.0
    n = y_pred.size(0)
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] < y_true[j]:
                loss += torch.log(1 + torch.exp(-(y_pred[j] - y_pred[i])))
    return loss / (n * (n - 1))

def spearman_loss(y_pred, y_true):
    pred_rank = y_pred.argsort().argsort().float()
    true_rank = y_true.argsort().argsort().float()
    pred_rank = (pred_rank - pred_rank.mean()) / pred_rank.std()
    true_rank = (true_rank - true_rank.mean()) / true_rank.std()
    return 1 - torch.cosine_similarity(pred_rank, true_rank, dim=0)

def sinkhorn_regularization(P):
    return torch.norm(P @ P.T - torch.eye(P.size(0)).to(P.device))

# === Real BiomedBERT encoder ===
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
    def __init__(self, text_dim, numeric_dim, hidden_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.numeric_proj = nn.Linear(numeric_dim, hidden_dim)
        self.permutation_net = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=2
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, text_embed, numeric_input):
        t = self.text_proj(text_embed)
        n = self.numeric_proj(numeric_input)
        h = t + n
        P = torch.softmax(self.permutation_net(h), dim=-1)
        h_perm = torch.matmul(P, h)
        h_seq = self.decoder(h_perm.unsqueeze(1)).squeeze(1)
        out = self.head(h_seq).squeeze(-1)
        return out, P

# === Training loop ===
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dummy data
    dummy_texts = ["patient age gender", "gene TP53 mutation", "lung adenocarcinoma"] * 34
    numeric_tensor = torch.randn(102, 10)
    pseudotime = torch.sort(torch.rand(102))[0] * 10

    dataset = TableDataset(dummy_texts, numeric_tensor, pseudotime)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    encoder = BiomedEncoder().to(device)
    model = CausalSetToSequence(text_dim=768, numeric_dim=10, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=2e-4)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for text_batch, x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            text_embed = encoder(text_batch)

            y_pred, P = model(text_embed, x_batch)

            loss_rank = rank_loss(y_pred, y_batch)
            loss_spearman = spearman_loss(y_pred, y_batch).mean()
            loss_mse = F.mse_loss(y_pred, y_batch)
            loss_perm = sinkhorn_regularization(P)

            loss = 1.0 * loss_rank + 1.0 * loss_spearman + 0.5 * loss_mse + 0.1 * loss_perm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"rank_loss": loss_rank.item(), "spearman_loss": loss_spearman.item(),
                       "mse_loss": loss_mse.item(), "perm_loss": loss_perm.item(), "total_loss": loss.item()})
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    os.environ['WANDB_MODE'] = 'online'  # or 'offline'
    train_model()
