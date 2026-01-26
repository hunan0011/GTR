import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

# ================= 配置区域 (请手动修改这里) =================
# 1. 验证集路径
QRELS_PATH = "/hdd02/jiangyutao/Code/passages/qrels.dev.small.tsv"
QUERY_PATH = "/hdd02/jiangyutao/Code/passages/queries.dev.small.tsv"

# 2. 向量路径 (BGE 编码好的文档向量 Memmap)
MEMMAP_PATH = "/hdd02/jiangyutao/Code/msmarco-passages-bge.memmap"

# 3. ID 映射路径 (改为使用 id2offset)
# 原来是: collection.tsv
# 现在改用你的映射文件:
DOCID_TO_INDEX_PATH = "/hdd02/jiangyutao/Code/msmarco-passages-bge_id2offset.tsv"

# 4. 模型路径 (BGE)
MODEL_NAME = "/hdd02/jiangyutao/Code/bge-base-en-v1.5"

# 5. 其他参数
BATCH_SIZE = 128
DOC_CHUNK_SIZE = 500000 
TOP_K = 100
EMBEDDING_DIM = 768     # 注意检查是否匹配 BGE-large
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ================= 模型定义 =================
class BGEEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        emb = out.last_hidden_state[:, 0]
        return F.normalize(emb, dim=1)

# ================= 数据集定义 =================
class QueryDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"Loading queries from {path}...")
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    qid = parts[0]
                    text = parts[1]
                    full_text = QUERY_INSTRUCTION + text
                    self.data.append((qid, full_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qid, text = self.data[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return qid, enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

def collate_fn(batch):
    qids, ids, masks = zip(*batch)
    return list(qids), torch.stack(ids), torch.stack(masks)

# ================= 工具函数 (修改版) =================
def load_pid_map(path):
    """
    自适应加载 ID 映射
    目标: 返回一个列表或字典, index -> pid
    """
    print(f"Loading PID mapping from {path}...")
    try:
        # 先读取前几行探测格式
        sample = pd.read_csv(path, sep="\t", nrows=5)
        
        # 情况 1: 包含 'offset' 和 'docid'/'pid' 列头 (标准 id2offset)
        if "offset" in sample.columns and ("docid" in sample.columns or "pid" in sample.columns):
            print("  -> Detected format: header with 'offset' column")
            df = pd.read_csv(path, sep="\t")
            id_col = "docid" if "docid" in df.columns else "pid"
            
            # 构建列表: list[offset] = pid
            max_idx = df["offset"].max()
            index2pid = [None] * (max_idx + 1)
            for _, row in df.iterrows():
                index2pid[row["offset"]] = str(row[id_col])
            return index2pid

        # 情况 2: 只有两列数据, 默认第一列是 pid, 第二列是 offset (或者反过来)
        # 这里我们假设如果读两列，通常第一列是 ID
        elif len(sample.columns) >= 2:
            print("  -> Detected format: multi-column (assuming col[0]=pid, col[1]=offset or vice versa)")
            # 这是一个简单的假设，如果你的文件是 offset \t pid，这里可能需要反转
            # 通常 id2offset 是: id \t offset
            # 但既然你有 collection.tsv 的经验，这里我们直接按 offset 排序读取更稳妥
            # 最稳妥的方式：如果文件没有表头，且你确定它是 id2offset
            # 我们直接把整个文件读进来，用字典映射
            df = pd.read_csv(path, sep="\t", header=None)
            
            # 尝试判断哪一列是 int (offset)
            col0_is_int = pd.api.types.is_integer_dtype(df[0])
            col1_is_int = pd.api.types.is_integer_dtype(df[1])

            index2pid = {}
            if col1_is_int and not col0_is_int:
                # col 0 is PID, col 1 is Offset
                for pid, off in zip(df[0], df[1]):
                    index2pid[off] = str(pid)
            elif col0_is_int and not col1_is_int:
                # col 0 is Offset, col 1 is PID
                for off, pid in zip(df[0], df[1]):
                    index2pid[off] = str(pid)
            else:
                # 都在或者都不在，默认行号就是 offset
                print("  -> Warning: Could not infer offset column. Using Row Index as Offset.")
                return df[0].astype(str).tolist()
                
            # 转为 list (如果 offset 是连续的)
            max_off = max(index2pid.keys())
            final_list = [None] * (max_off + 1)
            for off, pid in index2pid.items():
                final_list[off] = pid
            return final_list

        # 情况 3: 只有一列 (pid)，行号即 offset
        else:
            print("  -> Detected format: single column (Row Index = Offset)")
            df = pd.read_csv(path, sep="\t", header=None)
            return df[0].astype(str).tolist()

    except Exception as e:
        print(f"Error loading PID map: {e}")
        exit()

def load_qrels(path):
    print(f"Loading Qrels from {path}...")
    qrels = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid = parts[0]
                pid = parts[2]
                if qid not in qrels:
                    qrels[qid] = set()
                qrels[qid].add(pid)
    return qrels

def compute_metrics(qrels, results):
    print("Calculating metrics...")
    recall_list = []
    mrr_list = []
    
    for qid, doc_list in results.items():
        if qid not in qrels:
            continue
            
        gold_set = qrels[qid]
        candidates = doc_list[:TOP_K]
        
        hits = [1 if doc in gold_set else 0 for doc in candidates]
        recall = 1.0 if sum(hits) > 0 else 0.0
        recall_list.append(recall)
        
        mrr = 0.0
        for i, doc in enumerate(candidates):
            if doc in gold_set:
                mrr = 1.0 / (i + 1)
                break
        mrr_list.append(mrr)
        
    return np.mean(recall_list), np.mean(mrr_list)

# ================= 主流程 =================
def main():
    # 1. 准备数据映射 (使用 id2offset)
    index2docid = load_pid_map(DOCID_TO_INDEX_PATH)
    qrels = load_qrels(QRELS_PATH)
    
    # 2. 加载模型
    print(f"Loading BGE Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = BGEEncoder(MODEL_NAME).to(DEVICE)
    encoder.eval()
    
    # 3. 编码 Query
    q_dataset = QueryDataset(QUERY_PATH, tokenizer)
    q_loader = DataLoader(q_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    print("Encoding queries...")
    all_q_embeds = []
    all_qids = []
    
    with torch.no_grad():
        for qids, input_ids, mask in tqdm(q_loader, desc="Query Enc"):
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            emb = encoder(input_ids, mask)
            all_q_embeds.append(emb.cpu())
            all_qids.extend(qids)
            
    q_tensor = torch.cat(all_q_embeds, dim=0).to(DEVICE)
    num_queries = q_tensor.shape[0]

    # 4. 暴力检索
    print(f"Opening Doc Memmap from {MEMMAP_PATH}...")
    file_size = os.path.getsize(MEMMAP_PATH)
    float32_size = 4
    num_docs = file_size // (EMBEDDING_DIM * float32_size)
    
    # 简单的长度校验
    # 注意: id2offset 列表长度可能比 Memmap 大一点 (如果是稀疏 offset)，或者相等
    # 只要 index2docid[i] 存在即可
    print(f"Memmap Docs: {num_docs}, ID Map Size: {len(index2docid)}")

    global_topk_values = torch.full((num_queries, TOP_K), -float('inf'), device=DEVICE)
    global_topk_indices = torch.zeros((num_queries, TOP_K), dtype=torch.long, device=DEVICE)
    
    doc_memmap = np.memmap(MEMMAP_PATH, dtype='float32', mode='r', shape=(num_docs, EMBEDDING_DIM))
    
    print(f"Starting Brute Force Search...")
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, num_docs, DOC_CHUNK_SIZE), desc="Scanning Docs"):
            end_idx = min(start_idx + DOC_CHUNK_SIZE, num_docs)
            
            chunk_data = torch.from_numpy(doc_memmap[start_idx:end_idx]).to(DEVICE)
            scores = torch.matmul(q_tensor, chunk_data.T)
            
            curr_k = min(TOP_K, scores.shape[1])
            batch_vals, batch_rel_inds = torch.topk(scores, k=curr_k, dim=1)
            batch_abs_inds = batch_rel_inds + start_idx
            
            combined_vals = torch.cat([global_topk_values, batch_vals], dim=1)
            combined_inds = torch.cat([global_topk_indices, batch_abs_inds], dim=1)
            
            final_vals, indices_of_indices = torch.topk(combined_vals, k=TOP_K, dim=1)
            
            global_topk_values = final_vals
            global_topk_indices = torch.gather(combined_inds, 1, indices_of_indices)
            
            del chunk_data, scores
            torch.cuda.empty_cache()

    # 5. 结果生成
    print("Mapping indices to PIDs...")
    global_topk_indices = global_topk_indices.cpu().numpy()
    
    results = {}
    for i, qid in enumerate(all_qids):
        indices = global_topk_indices[i]
        try:
            # 这里使用 index2docid 列表进行查找
            doc_list = []
            for idx in indices:
                if idx < len(index2docid) and index2docid[idx] is not None:
                    doc_list.append(str(index2docid[idx]))
                else:
                    # 如果 offset 越界或为空，说明数据有问题
                    continue
            results[qid] = doc_list
        except IndexError:
            continue

    recall, mrr = compute_metrics(qrels, results)
    
    print("\n" + "="*30)
    print(f"Brute Force (BGE) Results")
    print("="*30)
    print(f"Recall@{TOP_K}: {recall:.4f}")
    print(f"MRR@{TOP_K}:    {mrr:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()