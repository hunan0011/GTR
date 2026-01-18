import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# ======================= 配置 =======================
BGE_LOCAL_PATH = "/hdd02/jiangyutao/Code/bge-base-en-v1.5"
TSV_PATH = "/hdd02/jiangyutao/github/DRhard/data/doc/dataset/msmarco-docs.tsv"

OUTPUT_MEMMAP_PATH = "msmarco-bge.memmap"
MAPPING_PATH = "msmarco-bge_id2offset.tsv"

BATCH_SIZE = 512           # FP32 下安全值：RTX 3090/4090 可尝试 768~1024
MAX_LENGTH = 512
USE_FP16 = False           # 关闭 FP16，使用全精度
CHUNK_SIZE = 100_000       # 每次读取 10 万文档，可根据内存调整（5万~20万）
# ====================================================

print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(BGE_LOCAL_PATH)
model = AutoModel.from_pretrained(BGE_LOCAL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"使用设备: {device}, FP16: {USE_FP16}")

embedding_dim = model.config.hidden_size  # 自动获取，通常是 768

# ==================== 高效统计总文档数 ====================
print("正在统计总文档数...")
# 方法：只读第一列（docid），速度最快
total_docs = len(pd.read_csv(TSV_PATH, sep='\t', header=None, usecols=[0], dtype=str))
print(f"总文档数: {total_docs:,}")

# ==================== 创建 memmap 和映射文件 ====================
print(f"创建 embedding memmap 文件: {OUTPUT_MEMMAP_PATH} ({total_docs} x {embedding_dim})")
embeddings_memmap = np.memmap(
    OUTPUT_MEMMAP_PATH,
    dtype='float32',
    mode='w+',
    shape=(total_docs, embedding_dim)
)

mapping_file = open(MAPPING_PATH, 'w', encoding='utf-8')
mapping_file.write("docid\toffset\n")

# ==================== 开始分块处理 ====================
print("开始分块编码文档（title + body，不包含 URL）...")
global_idx = 0

with torch.no_grad():
    for chunk_df in tqdm(
        pd.read_csv(
            TSV_PATH,
            sep="\t",
            header=None,
            names=["docid", "url", "title", "body"],
            usecols=[0, 1, 2, 3],
            dtype=str,
            on_bad_lines='skip',
            chunksize=CHUNK_SIZE
        ),
        desc="Processing chunks",
        total=(total_docs + CHUNK_SIZE - 1) // CHUNK_SIZE
    ):
        # 清洗字段
        chunk_df["title"] = chunk_df["title"].fillna("").str.strip()
        chunk_df["body"] = chunk_df["body"].fillna("").str.strip()
        chunk_df["docid"] = chunk_df["docid"].astype(str).str.strip()

        # 智能拼接：有 title 就用 title + " " + body，否则只用 body
        texts = []
        for title, body in zip(chunk_df["title"], chunk_df["body"]):
            if title:
                texts.append(title + " " + body)
            else:
                texts.append(body)

        docids = chunk_df["docid"].tolist()
        chunk_size = len(texts)

        # 分 batch 编码
        for i in range(0, chunk_size, BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(device)

            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS embedding

            # BGE 必须：L2 归一化
            cls_emb = F.normalize(cls_emb, p=2, dim=1)

            # 转 numpy 并写入 memmap
            batch_emb = cls_emb.cpu().numpy()
            start_idx = global_idx + i
            end_idx = start_idx + len(batch_texts)
            embeddings_memmap[start_idx:end_idx] = batch_emb

            # 可选：释放显存
            del inputs, outputs, cls_emb, batch_emb
            torch.cuda.empty_cache() if device.type == 'cuda' else None

        # 写入当前 chunk 的 docid → offset 映射
        for docid in docids:
            mapping_file.write(f"{docid}\t{global_idx}\n")
            global_idx += 1

        # 每 chunk 结束刷新一次磁盘
        embeddings_memmap.flush()

# ==================== 完成收尾 ====================
mapping_file.close()
print("\n全部完成！")
print(f"实际处理文档数: {global_idx:,}")
print(f"Embeddings 保存至: {OUTPUT_MEMMAP_PATH} ({total_docs} x {embedding_dim})")
print(f"docid → offset 映射保存至: {MAPPING_PATH}")
print("提示：嵌入已进行 L2 归一化，可直接用于余弦相似度 / 内积检索")
print("输入格式：title + \" \" + body（无 title 时仅 body，无 URL）")