import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
import os
import argparse
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from model import *
from preprocess import*

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--num_bench', type=int, default=100)
parser.add_argument('--net', type=str, default='set_transformer')
parser.add_argument('--K', type=int, default=16)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--run_name', type=str, default='trial')
parser.add_argument('--num_steps', type=int, default=50000)
parser.add_argument('--test_freq', type=int, default=200)
parser.add_argument('--save_freq', type=int, default=400)
parser.add_argument('--num_epochs', type=int, default=10) 

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


def permutation_loss(y_pred):
    # Sinkhorn soft permutation loss
    from torch.nn.functional import kl_div
    B = y_pred.shape[0]
    scores = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)  # (B, B)
    P = torch.softmax(scores, dim=-1)
    log_P = torch.log(P + 1e-8)
    uniform = torch.full_like(P, 1.0 / B)
    loss = kl_div(log_P, uniform, reduction='batchmean')
    return loss


def total_loss(y_true, y_pred, lambdas=(1.0, 1.0, 0.1, 0.5)):
    l_rank = causal_pairwise_rank_loss(y_true, y_pred)
    l_spear = spearman_loss(y_true, y_pred)
    l_perm = permutation_loss(y_pred)
    l_mse = mse_loss(y_true, y_pred)
    return lambdas[0]*l_rank + lambdas[1]*l_spear + lambdas[2]*l_perm + lambdas[3]*l_mse


def train(train_dataloader, val_dataloader,net, patience=10, delta=1e-4):
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = total_loss  # 明确使用哪个 loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.num_steps // 2, gamma=0.1)

    best_loss = float("inf")
    patience_counter = 0
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(args.run_name)
    logger.addHandler(logging.FileHandler(
        os.path.join(save_dir, 'train_' + time.strftime('%Y%m%d-%H%M') + '.log'),
        mode='w'))
    logger.info(str(args) + '\n')
    for epoch in range(args.num_epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0

        # ---------- TRAIN ----------
        net.train()
        for fused_X, y in train_dataloader:
            fused_X, y = fused_X.to(device), y.to(device)

            optimizer.zero_grad()
            output = net(fused_X)          # (B, K, 1)
            output = output.mean(dim=1)    # 聚合成 (B, 1)
            loss = criterion(y, output)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()

        # ---------- VAL ----------
        net.eval()
        with torch.no_grad():
            for fused_X, y in val_dataloader:
                fused_X, y = fused_X.to(device), y.to(device)
                output = net(fused_X)
                output = output.mean(dim=1)
                loss = criterion(y, output)
                epoch_val_loss += loss.item()

        epoch_train_loss /= len(train_dataloader)
        epoch_val_loss /= len(val_dataloader)

        logger.info(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_loss - delta:
            best_loss = epoch_val_loss
            patience_counter = 0
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            print(f"Early stopping at epoch {epoch+1}")
            break

    logger.info("Final model saved.")
    print("Final model saved.")


def fuse_in_batches(fusion_module, X_all, K_all, batch_size=128):
    fusion_module.eval()
    fused_list = []
    K_all = K_all.to(device)  # 移动一次，不要每次都重复

    for i in range(0, X_all.size(0), batch_size):
        x_batch = X_all[i:i+batch_size].to(device)
        with torch.no_grad():
            fused = fusion_module(x_batch, K_all).cpu()  # ❗注意 K_all 不切 batch
        fused_list.append(fused)
    return torch.cat(fused_list, dim=0)
           
def reload_data():
    # Step 1: Load data
    X_train, y_train, X_test, y_test, train_idx_tensor, test_idx_tensor = torch.load("/beacon/data01/zhen.lu001/TAPES/src250530/preprocessed_data2.pt")
    embeddings = load_embeddings()

    # Step 2: Normalize y
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y_train = torch.tensor(scaler_y.fit_transform(y_train.numpy().reshape(-1, 1)), dtype=torch.float32)
    y_test = torch.tensor(scaler_y.transform(y_test.numpy().reshape(-1, 1)), dtype=torch.float32)

    # Step 3: Build aligned full X and K
    X_all = torch.cat([X_train, X_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)
    K_all = embeddings

    # Step 4: Move to device and fuse (no gradient)
    fusion_module = ExpressionKnowledgeFusion(
        dim_input=X_all.size(1),
        dim_k=K_all.size(1)
        ).to(device)
    fusion_module.eval()
    with torch.no_grad():
        fused_all = fuse_in_batches(fusion_module, X_all, K_all, batch_size=32)
        print(f"✅ fused_all shape: {fused_all.shape}")
    # Step 5: Split back
    fused_train = fused_all[train_idx_tensor]
    fused_test = fused_all[test_idx_tensor]
    y_train = y_all[train_idx_tensor]
    y_test = y_all[test_idx_tensor]

    fused_train, fused_val, y_train, y_val = train_test_split(
        fused_train, y_train, test_size=0.2, random_state=42
    )

    # Step 6: Dataset & Dataloader
    class FusedDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = FusedDataset(fused_train, y_train)
    val_dataset = FusedDataset(fused_val, y_val)
    test_dataset = FusedDataset(fused_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    global D
    D = fused_train.shape[2]

    return train_dataloader, val_dataloader, test_dataloader, scaler_y



def set_seed(seed):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
seed = 42  
set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = os.path.join('results', args.net, args.run_name)
base_filename = "model"
ext = ".tar"
counter = 1
K = args.K
dim_output = 1
model_path = os.path.join(save_dir, f"{base_filename}{ext}")

while os.path.exists(model_path):
    model_path = os.path.join(save_dir, f"{base_filename}_{counter}{ext}")
    counter += 1

if __name__ == '__main__':
    if args.mode == 'train':
        train_dataloader, val_dataloader, test_dataloader, scaler_y = reload_data()

        net = CausalSetTransformer(D, K, dim_output).to(device)
        train(train_dataloader,val_dataloader,net)