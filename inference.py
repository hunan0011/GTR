import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
from collections import defaultdict
import config

# =========================================================
# 【关键】直接从 model.py 导入定义
# 这样无论你的 model.py 里的 Similarity 长什么样（点积版还是交互版）
# 这里都能自动适配，不需要改代码。
# =========================================================
from model import Similarity, Encoder, Indexer

# ================= 配置 =================
BEAM_SIZE_LIST = [10, 20, 30, 40, 50]
CHECKPOINT_LIST = [
    "/home/jiangda/jiangyutao/Code/output/checkpoint-1.pt",
    "/home/jiangda/jiangyutao/Code/output/checkpoint-2.pt",
    "/home/jiangda/jiangyutao/Code/output/checkpoint-3.pt",
    "/home/jiangda/jiangyutao/Code/output/checkpoint-4.pt",
    "/home/jiangda/jiangyutao/Code/output/checkpoint-5.pt",
    "/home/jiangda/jiangyutao/Code/output/checkpoint-6.pt",
    "/home/jiangda/jiangyutao/Code/output/checkpoint-7.pt",
    "/home/jiangda/jiangyutao/Code/output/checkpoint-8.pt",
    "/home/jiangda/jiangyutao/Code/output/checkpoint-9.pt",
    "/home/jiangda/jiangyutao/Code/output/checkpoint-10.pt",
    "/home/jiangda/jiangyutao/Code/output/checkpoint-11.pt",
    "/home/jiangda/jiangyutao/Code/output/checkpoint-12.pt",
    "/home/jiangda/jiangyutao/Code/output/checkpoint-13.pt",
]
QRELS_FILE = "/home/jiangda/jiangyutao/Code/passages/qrels.dev.small.tsv"
FINAL_OUTPUT_FILE = os.path.join(config.OUTPUT_DIR, "latest_result.trec")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_PATH = config.MODEL_NAME
MEMMAP_PATH = config.MEMMAP_PATH
ID2INDEX_PATH = config.ID2OFFSET
QUERY_PATH = config.DEV_DOC_TRAIN_QUERIES
EMBEDDING_DIM = config.EMBEDDING_DIM
TREE_HEIGHT = config.TREE_HEIGHT
NODE_BALANCE = config.NODE_BALANCE
QUERY_INSTRUCTION = "" 
BATCH_SIZE = 128  # 【修改】恢复为 128，设为 1 推理太慢了
NUM_WORKERS = 4
EVAL_TOPK = 100

# ================= 数据与资源 =================
class QueryDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split("\t")
                if len(parts) >= 2: self.data.append((parts[0], QUERY_INSTRUCTION + parts[1]))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        qid, text = self.data[idx]
        enc = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return qid, enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

def collate_fn(batch):
    qids, ids, masks = zip(*batch)
    return list(qids), torch.stack(ids), torch.stack(masks)

class StaticResources:
    def __init__(self):
        self.doc_embeddings = np.fromfile(MEMMAP_PATH, dtype="float32").reshape(-1, EMBEDDING_DIM)
        self.docid2idx, self.idx2docid = {}, {}
        with open(ID2INDEX_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[1].isdigit():
                    self.docid2idx[parts[0]] = int(parts[1])
                    self.idx2docid[int(parts[1])] = parts[0]
        with open(config.ID2PATH, "rb") as f: m = pickle.load(f)
        self.leaf_to_offsets = defaultdict(list)
        for doc_id_str, paths in m.items():
            if len(paths) > 0 and isinstance(paths[0], int): paths = [paths]
            offset = self.docid2idx.get(doc_id_str)
            if offset is not None:
                for p in paths: self.leaf_to_offsets[p[-1]].append(offset)
        for k in self.leaf_to_offsets: self.leaf_to_offsets[k] = np.array(self.leaf_to_offsets[k], dtype=np.int64)
        with open(config.CHILDREN_EMBEDDINGS_PATH, "rb") as f: children = pickle.load(f)
        max_id = 0
        for p, cs in children.items():
            max_id = max(max_id, p)
            for c in cs: max_id = max(max_id, c["child_id"])
        self.pad = max_id + 1
        adj = torch.full((self.pad + 1, NODE_BALANCE), self.pad, dtype=torch.long)
        for p, cs in children.items():
            for i, c in enumerate(sorted(cs, key=lambda x: x["child_index"])):
                if i < NODE_BALANCE: adj[p, i] = c["child_id"]
        self.adjacency = adj.to(DEVICE)

class NeuralRetriever:
    def __init__(self, resources, model_path):
        ckpt = torch.load(model_path, map_location=DEVICE)
        
        # 使用导入的类实例化，传入 model.py 中 Indexer 需要的参数
        self.encoder = Encoder(BASE_MODEL_PATH).to(DEVICE)
        self.indexer = Indexer(H=TREE_HEIGHT, B=NODE_BALANCE).to(DEVICE)
        
        # 加载 Encoder
        self.encoder.load_state_dict(ckpt["encoder"], strict=False)
        
        # 处理 Centroids 
        # (如果 checkpoint 里有 fixed_centroids，重新初始化 embedding 层以匹配大小)
        if "fixed_centroids.weight" in ckpt["indexer"]:
            num_embeddings = ckpt["indexer"]["fixed_centroids.weight"].shape[0]
            self.indexer.fixed_centroids = nn.Embedding(num_embeddings, EMBEDDING_DIM).to(DEVICE)
            
        # 加载 Indexer (此时会自动加载里面的 scorers 权重)
        self.indexer.load_state_dict(ckpt["indexer"], strict=False)
        
        self.res = resources
        self.indexer.adjacency = self.res.adjacency
        self.scale = self.indexer.logit_scale.exp().clamp(max=100)
        self.encoder.eval()
        self.indexer.eval()

    def _layer_logits(self, q, parents, h):
        B, K = parents.shape
        child = self.indexer.adjacency[parents.view(-1)].view(B, K, -1)
        cent = self.indexer.fixed_centroids(child)
        
        # 核心逻辑：直接调用 scorer
        # 这里的 scorer 就是你在 model.py 定义的 Interaction MLP
        # 它接收 (query, candidates) -> 返回 [B, K] 的分数
        scores = self.indexer.scorers[h - 1](q, cent)
        
        return child, scores * self.scale

    @torch.no_grad()
    def beam_search(self, q, beam_size):
        B = q.size(0)
        nodes = torch.zeros((B, beam_size), dtype=torch.long, device=DEVICE)
        scores = torch.full((B, beam_size), -1e9, device=DEVICE)
        scores[:, 0] = 0
        for h in range(1, TREE_HEIGHT):
            child, logit = self._layer_logits(q, nodes, h)
            total = scores.unsqueeze(-1) + F.log_softmax(logit, -1)
            topk_s, topk_idx = total.view(B, -1).topk(beam_size, -1)
            nodes = child.view(B, -1).gather(1, topk_idx)
            scores = topk_s
        return nodes

    @torch.no_grad()
    def rerank(self, qids, q, leaf):
        leaf_nodes = leaf.view(-1).cpu().numpy()
        unique_leaves = np.unique(leaf_nodes[leaf_nodes != self.res.pad])
        batch_offsets = np.concatenate([self.res.leaf_to_offsets.get(l, np.array([], dtype=np.int64)) for l in unique_leaves])
        if len(batch_offsets) == 0: return []
        batch_offsets = np.unique(batch_offsets)
        cand_embs = torch.from_numpy(self.res.doc_embeddings[batch_offsets]).to(DEVICE)
        
        # 注意：这里 Rerank 依然使用点积 (matmul)。
        # 如果你的 Similarity MLP 学习到的特征空间与原始点积差异极大，这里可能效果一般。
        # 但通常作为 Tree Retrieval 的后处理，直接点积是可以接受的 baseline。
        scores = torch.matmul(q, cand_embs.T)
        
        k = min(EVAL_TOPK, scores.size(1))
        top_vals, top_inds = scores.topk(k, dim=1)
        top_vals, top_inds = top_vals.cpu().numpy(), top_inds.cpu().numpy()
        real_offsets = batch_offsets[top_inds]
        res = []
        for b, qid in enumerate(qids):
            for r in range(k):
                res.append((qid, self.res.idx2docid.get(real_offsets[b, r], str(real_offsets[b, r])), top_vals[b, r]))
        return res

def load_qrels(qrels_path):
    qrels = defaultdict(set)
    if os.path.exists(qrels_path):
        with open(qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4 and int(parts[3]) > 0: qrels[parts[0]].add(parts[2])
    return qrels

def compute_metrics(qrels, run, k=100):
    mrr_sum, recall_sum, checked = 0.0, 0.0, 0
    for qid in sorted(list(qrels.keys())):
        if qid not in run: 
            checked += 1
            continue
        relevant = qrels[qid]
        retrieved = [doc for doc, _ in run[qid][:k]]
        mrr_sum += next((1.0 / (i + 1) for i, d in enumerate(retrieved) if d in relevant), 0.0)
        recall_sum += sum(1 for d in retrieved if d in relevant) / len(relevant) if relevant else 0.0
        checked += 1
    return mrr_sum / checked if checked > 0 else 0.0, recall_sum / checked if checked > 0 else 0.0

# ================= 主流程 =================
def main():
    qrels_data = load_qrels(QRELS_FILE)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    dataloader = DataLoader(QueryDataset(QUERY_PATH, tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    static_resources = StaticResources()
    results_summary = []

    os.makedirs(os.path.dirname(FINAL_OUTPUT_FILE), exist_ok=True)

    # 1. 运行所有测试
    for model_path in CHECKPOINT_LIST:
        try:
            retriever = NeuralRetriever(static_resources, model_path)
            model_name = os.path.basename(model_path)
            for beam_size in BEAM_SIZE_LIST:
                all_results, total_time, total_queries = [], 0.0, 0
                for qids, ids, mask in tqdm(dataloader, desc=f"{model_name} B={beam_size}", leave=False):
                    ids, mask = ids.to(DEVICE), mask.to(DEVICE)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.no_grad():
                        q = retriever.encoder(ids, mask)
                        leaf = retriever.beam_search(q, beam_size=beam_size)
                        res = retriever.rerank(qids, q, leaf)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    total_time += time.time() - t0
                    total_queries += len(qids)
                    all_results.extend(res)

                aqt_ms = (total_time / total_queries * 1000) if total_queries > 0 else 0.0
                
                with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
                    f.writelines([f"{qid}\tQ0\t{did}\t0\t{s:.6f}\tTreeSearch\n" for qid, did, s in all_results])
                
                run_data = defaultdict(list)
                for qid, did, s in all_results: run_data[qid].append((did, s))
                for qid in run_data: run_data[qid].sort(key=lambda x: x[1], reverse=True)
                
                mrr, recall = compute_metrics(qrels_data, run_data, k=100)
                results_summary.append({"Model": model_name, "Beam": beam_size, "MRR": mrr, "R": recall, "AQT": aqt_ms})
                print(f"{model_name} B={beam_size} | MRR: {mrr:.4f} | R: {recall:.4f} | AQT: {aqt_ms:.2f} ms")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error {model_path}: {e}")

    # 2. 打印完整报告
    print("\n" + "="*65)
    print(f"{'FINAL EVALUATION REPORT':^65}")
    print("="*65)
    print(f"{'Model':<30} | {'Beam':<5} | {'MRR@100':<8} | {'R@100':<8} | {'AQT (ms)':<10}")
    print("-" * 65)
    for res in results_summary:
        print(f"{res['Model']:<30} | {res['Beam']:<5} | {res['MRR']:.4f}   | {res['R']:.4f}   | {res['AQT']:<10.2f}")
    print("-" * 65)

    # 3. 打印最佳 Checkpoint
    print("\n" + "="*65)
    print(f"{'BEST CHECKPOINT PER BEAM SIZE (Sorted by MRR)':^65}")
    print("="*65)
    print(f"{'Beam':<5} | {'Best Model':<30} | {'MRR':<8} | {'R':<8} | {'AQT':<10}")
    print("-" * 65)
    
    beam_groups = defaultdict(list)
    for res in results_summary:
        beam_groups[res['Beam']].append(res)
    
    for beam in sorted(beam_groups.keys()):
        best_run = max(beam_groups[beam], key=lambda x: x['MRR'])
        print(f"{best_run['Beam']:<5} | {best_run['Model']:<30} | {best_run['MRR']:.4f}   | {best_run['R']:.4f}   | {best_run['AQT']:<10.2f}")
    print("-" * 65)

if __name__ == "__main__":
    main()