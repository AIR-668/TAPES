import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from torch.utils.data import Dataset, DataLoader


def load_data(top_genes=500, test_size=0.2, random_state=42):
    # Step 1: Load and transpose
    df = pd.read_csv('/beacon/data01/zhen.lu001/harvard_project/set_transformer/TcgaTargetGtex_rsem_gene_tpm.txt', delimiter='\t')
    df_transposed = df.T
    df_transposed.reset_index(drop=False, inplace=True)
    df_transposed.columns = df_transposed.iloc[0]
    df_transposed = df_transposed.drop(0)
    df = df_transposed

    # Step 2: Filter GTEx and TCGA samples
    df_filtered = df[df['sample'].str.contains('gtex|tcga', case=False, na=False)].copy()
    df_filtered['label'] = df_filtered['sample'].str.contains('gtex', case=False).replace({True: 'gtex', False: 'tcga'})

    # Step 3: Convert expression columns to numeric
    expr_data = df_filtered.drop(columns=['sample', 'label']).apply(pd.to_numeric, errors='coerce')

    # Step 4: Select top-K high variance genes
    if isinstance(top_genes, int):
        variances = expr_data.var(axis=0)
        variances = variances.drop("ENSG00000011426.10", errors="ignore")  # Drop ANLN
        selected_genes = variances.sort_values(ascending=False).head(top_genes).index.tolist()
    elif isinstance(top_genes, list):
        selected_genes = [g for g in top_genes if g in expr_data.columns]
    else:
        raise ValueError("top_genes must be an int or list of gene names.")

    # Step 5: Subset X and get label
    X = expr_data[selected_genes]
    anln = df_filtered['ENSG00000011426.10'].apply(pd.to_numeric, errors='coerce')
    y = anln

    # Step 6: Scale X
    X = X.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # Step 7: Split indices
    indices = np.arange(len(X_tensor))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

    train_idx_tensor = torch.tensor(train_idx, dtype=torch.long)
    test_idx_tensor = torch.tensor(test_idx, dtype=torch.long)

    X_train, X_test = X_tensor[train_idx_tensor], X_tensor[test_idx_tensor]
    y_train, y_test = y_tensor[train_idx_tensor], y_tensor[test_idx_tensor]

    # Step 8: Save
    torch.save((X_train, y_train, X_test, y_test, train_idx_tensor, test_idx_tensor), "preprocessed_data2.pt")
    anln.to_pickle("anln_data.pkl")
    print(f"✅ Saved to preprocessed_data2.pt with {len(selected_genes)} genes.")

    # return selected_genes


def load_embeddings(path="/beacon/data01/zhen.lu001/harvard_project/BioBert/biomedbert_gene_embeddings_top500.npy"):
    embeddings = np.load(path)
    return torch.tensor(embeddings, dtype=torch.float32)

class GeneExpressionDataset(Dataset):
    def __init__(self, X, embeddings, y):
        self.X = X
        self.embeddings = embeddings
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx].unsqueeze(0)      # (1, G)
        k_embed = self.embeddings[idx].unsqueeze(0)  # (1, d_k)
        label = self.y[idx]
        return x_sample, k_embed, label




if __name__ == '__main__':
    load_data(top_genes=500)  # You can specify top_genes if needed