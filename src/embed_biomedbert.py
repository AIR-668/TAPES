from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import os

# ====== Config ======
csv_path = "/workspace/data/TAPES_Data_Harvard_rabdom50.csv"
output_path = "/workspace/data/biomedbert_embeddings.pt"
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# ====== Load CSV ======
df = pd.read_csv(csv_path)
first_row = df.iloc[0]
input_texts = [f"{col}: {str(first_row[col])}" for col in df.columns]

# ====== Load BiomedBERT ======
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# ====== Encode and extract embedding ======
model.eval()
with torch.no_grad():
    embeddings = []
    for text in input_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # shape: [1, seq_len, hidden_dim]
        cls_embedding = last_hidden_state[:, 0, :]      # use [CLS] token representation
        embeddings.append(cls_embedding.squeeze(0))

    embeddings_tensor = torch.stack(embeddings)  # shape: [num_columns, hidden_dim]

# ====== Save ======
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save(embeddings_tensor, output_path)
print(f"Saved {embeddings_tensor.shape} to {output_path}")