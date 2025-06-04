import pandas as pd
import torch
import argparse
from sklearn.preprocessing import StandardScaler

# === 功能 ===
# 将 CSV 表格处理为：
# 1. 每行的列名组成的文本（用于 BiomedBERT 编码）
# 2. 每行的数值特征张量
# 3. 输出 pseudotime 标签（若存在）

def preprocess_csv(csv_path, output_prefix, pseudotime_col=None):
    df = pd.read_csv(csv_path)

    if pseudotime_col:
        y = torch.tensor(df[pseudotime_col].values, dtype=torch.float32)
        df = df.drop(columns=[pseudotime_col])
    else:
        y = None

    # 提取列名作为表头文本，组合成每行共享的描述
    column_names = df.columns.tolist()
    column_text = ' '.join(column_names)
    col_texts = [column_text] * len(df)

    # 提取并标准化数值数据
    numeric_data = df.values.astype(float)
    numeric_tensor = torch.tensor(StandardScaler().fit_transform(numeric_data), dtype=torch.float32)

    # 保存
    torch.save(col_texts, f"{output_prefix}_col_texts.pt")
    torch.save(numeric_tensor, f"{output_prefix}_features.pt")
    if y is not None:
        torch.save(y, f"{output_prefix}_pseudotime.pt")

    print("✅ 数据预处理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="路径：输入 CSV 文件")
    parser.add_argument("--output", type=str, required=True, help="输出前缀")
    parser.add_argument("--pseudotime", type=str, default=None, help="真实 pseudotime 列名")
    args = parser.parse_args()

    preprocess_csv(args.csv, args.output, args.pseudotime)