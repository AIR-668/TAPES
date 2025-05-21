import argparse
import time
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import os
from tqdm import tqdm

# ========== Parse Arguments ==========
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU device index (0-7)')
args = parser.parse_args()

# ========== Config ==========
csv_path = "/workspace/data/TAPES_Data_Harvard_rabdom50.csv"
output_path = "/workspace/data/biomedbert_embeddings.pt"
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# ========== Setup Device ==========
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ========== Load CSV ==========
print("[INFO] Loading CSV...")
start_csv = time.time()
df = pd.read_csv(csv_path)
first_row = df.iloc[0]
input_texts = [f"{col}: {str(first_row[col])}" for col in df.columns]
print(f"[INFO] Loaded {len(input_texts)} column descriptions in {time.time() - start_csv:.2f} sec")

# ========== Load Model ==========
print(f"[INFO] Loading model: {model_name}")
start_model = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()
print(f"[INFO] Model loaded in {time.time() - start_model:.2f} sec")

# ========== Encode with TQDM ==========
print("[INFO] Encoding columns with BiomedBERT...")
start_encode = time.time()
embeddings = []

with torch.no_grad():
    for text in tqdm(input_texts, desc="Embedding", ncols=80):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embedding.squeeze(0).cpu())

embeddings_tensor = torch.stack(embeddings)
print(f"[INFO] Embedding completed in {time.time() - start_encode:.2f} sec")
print(f"[INFO] Embedding shape: {embeddings_tensor.shape}")

# ========== Save ==========
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save(embeddings_tensor, output_path)
print(f"[INFO] Saved embeddings to {output_path}")