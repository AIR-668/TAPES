import argparse
import time
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import os
from tqdm import tqdm
import json

# ========== Parse Arguments ==========
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU device index (0-7)')
args = parser.parse_args()

# ========== Config ==========
csv_path = "/workspace/data/TAPES_Data_Harvard_random50.csv"
pt_output_path = "/workspace/data/biomedbert_embeddings.pt"
json_output_path = "/workspace/data/embedding_meta.json"
csv_output_path = "/workspace/data/embedding_preview.csv"
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# ========== Setup Device ==========
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ========== Load CSV ==========
print("[INFO] Loading CSV...")
start_csv = time.time()
df = pd.read_csv(csv_path)
print(f"[INFO] Loaded CSV with shape {df.shape} in {time.time() - start_csv:.2f} sec")

# Extract row names and column names
row_names = df.iloc[:, 0].tolist()
col_names = df.columns.tolist()
print(f"[INFO] First column (row names) preview: {row_names[:5]}")
print(f"[INFO] Column names: {col_names}")

# ========== Build input texts ==========
first_row = df.iloc[0]
input_texts = [f"{col}: {str(first_row[col])}" for col in df.columns]

# ========== Load Model ==========
print(f"[INFO] Loading model: {model_name}")
start_model = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()
print(f"[INFO] Model loaded in {time.time() - start_model:.2f} sec")

# ========== Encode ==========
print("[INFO] Encoding column descriptions with BiomedBERT...")
start_encode = time.time()
embeddings = []
preview_data = []

with torch.no_grad():
    for col_name, text in tqdm(zip(col_names, input_texts), total=len(input_texts), desc="Embedding", ncols=80):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        emb_vector = cls_embedding.squeeze(0).cpu()
        embeddings.append(emb_vector)

        # print preview of first 5 dimensions
        preview_dims = emb_vector[:5].tolist()
        print(f"[EMBEDDING] {col_name:20s} → {preview_dims}")
        preview_data.append({
            "column_name": col_name,
            "input_text": text,
            **{f"dim_{i}": val for i, val in enumerate(preview_dims)}
        })

embeddings_tensor = torch.stack(embeddings)
print(f"[INFO] Embedding completed in {time.time() - start_encode:.2f} sec")
print(f"[INFO] Embedding tensor shape: {embeddings_tensor.shape}")

# ========== Save .pt ==========
os.makedirs(os.path.dirname(pt_output_path), exist_ok=True)
torch.save(embeddings_tensor, pt_output_path)
print(f"[INFO] Saved embeddings to {pt_output_path}")

# ========== Save metadata ==========
meta = {
    "row_names": row_names,
    "col_names": col_names,
    "embedding_shape": list(embeddings_tensor.shape),
}
with open(json_output_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"[INFO] Saved metadata to {json_output_path}")

# ========== Save preview CSV ==========
preview_df = pd.DataFrame(preview_data)
preview_df.to_csv(csv_output_path, index=False)
print(f"[INFO] Saved preview CSV to {csv_output_path}")
