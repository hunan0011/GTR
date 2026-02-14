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

# ================= 配置 =================
BEAM_SIZE_LIST = [10, 20, 30, 40, 50]
CHECKPOINT_LIST = [
    "/home/power/jiangyutao/GTR/output/checkpoint-1.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-2.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-3.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-4.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-5.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-6.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-7.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-8.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-9.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-10.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-11.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-12.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-13.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-14.pt",
    "/home/power/jiangyutao/GTR/output/checkpoint-15.pt",
]

# 优先使用配置中的路径，如果未定义则使用默认值
QRELS_FILE = config.DEV_DOC_TRAIN_QRELS
FINAL_OUTPUT_FILE = "./output/latest_result.trec"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_PATH = config.MODEL_NAME
MEMMAP_PATH = config.MEMMAP_PATH
ID2INDEX_PATH = config.ID2OFFSET
QUERY_PATH = config.DEV_DOC_TRAIN_QUERIES
EMBEDDING_DIM = config.EMBEDDING_DIM
TREE_HEIGHT = config.TREE_HEIGHT
NODE_BALANCE = config.NODE_BALANCE
QUERY_INSTRUCTION = "" 
BATCH_SIZE = 1
NUM_WORKERS = 4
EVAL_TOPK = 100

# ================= 模型 =================
class Similarity(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.q_mlp = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(input_dim, input_dim))
        self.c_mlp = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(input_dim, input_dim))
    
    def forward(self, query, candidates):
        return torch.sum(self.q_mlp(query) * self.c_mlp(candidates), dim=-1)

class Encoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, config=AutoConfig.from_pretrained(model_name))
    def forward(self, input_ids, attention_mask):
        return F.normalize(self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state[:, 0], dim=1)

class Indexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fixed_centroids = nn.Embedding(1, EMBEDDING_DIM)
        self.scorers = nn.ModuleList([Similarity(EMBEDDING_DIM) for _ in range(TREE_HEIGHT)])
        self.logit_scale = nn.Parameter(torch.tensor(np.log(20.0)))
        self.adjacency = None

# ================= 数据与资源 =================
class QueryDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"Loading queries from {path}...")
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
        print("Loading static resources (Memmap & ID Map)...")
        # [修改] 使用 mode='r' 确保是只读 Memmap，不加载进 RAM
        self.doc_embeddings = np.memmap(MEMMAP_PATH, dtype='float32', mode='r').reshape(-1, EMBEDDING_DIM)
        
        self.docid2idx, self.idx2docid = {}, {}
        print(f"Loading ID mapping form {ID2INDEX_PATH}")
        with open(ID2INDEX_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    # 兼容 id2offset 格式
                    if parts[1].isdigit():
                        self.docid2idx[parts[0]] = int(parts[1])
                        self.idx2docid[int(parts[1])] = parts[0]
                    else:
                        # 尝试兼容反向格式或 header
                        pass

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
        print(f"Loading model from {model_path}...")
        torch.cuda.empty_cache()
        
        ckpt = torch.load(model_path, map_location=DEVICE) # Removed weights_only=False for compatibility, add back if needed
        
        self.encoder = Encoder(BASE_MODEL_PATH).to(DEVICE)
        self.indexer = Indexer().to(DEVICE)
        
        self.encoder.load_state_dict(ckpt["encoder"], strict=False)
        if "fixed_centroids.weight" in ckpt["indexer"]:
            num_nodes = ckpt["indexer"]["fixed_centroids.weight"].shape[0]
            self.indexer.fixed_centroids = nn.Embedding(num_nodes, EMBEDDING_DIM).to(DEVICE)
            
        self.indexer.load_state_dict(ckpt["indexer"], strict=False)
        self.res = resources
        self.indexer.adjacency = self.res.adjacency
        self.scale = self.indexer.logit_scale.exp().clamp(max=100)
        
        self.encoder.eval()
        self.indexer.eval()

        # 预计算 Cached Centroids (保留)
        self.cached_layer_centroids = []
        with torch.no_grad():
            all_nodes_raw = self.indexer.fixed_centroids.weight
            for h in range(len(self.indexer.scorers)):
                transformed = self.indexer.scorers[h].c_mlp(all_nodes_raw)
                self.cached_layer_centroids.append(transformed)

        # [修改] 彻底移除了全量加载 GPU 的逻辑
        print("[INFO] Running in Low-Memory Mode: Fetching doc embeddings from disk on-the-fly.")
        self.use_gpu_index = False

    if hasattr(torch, "compile"):
        _layer_logits_fast = torch.compile(lambda self, q, p, h: self._internal_layer_logits(q, p, h))
    
    def _internal_layer_logits(self, q_proj, parents, h):
        B, K = parents.shape
        child_ids = self.indexer.adjacency[parents.view(-1)].view(B, K, -1)
        current_layer_cache = self.cached_layer_centroids[h - 1]
        child_embs = F.embedding(child_ids, current_layer_cache)
        logits = torch.sum(q_proj * child_embs, dim=-1) * self.scale
        return child_ids, logits

    def _layer_logits_fast(self, q_proj, parents, h):
        return self._internal_layer_logits(q_proj, parents, h)

    @torch.no_grad()
    def beam_search(self, q, beam_size):
        B = q.size(0)
        nodes = torch.zeros((B, beam_size), dtype=torch.long, device=DEVICE)
        scores = torch.full((B, beam_size), -1e9, device=DEVICE)
        scores[:, 0] = 0
        
        for h in range(1, TREE_HEIGHT):
            scorer = self.indexer.scorers[h - 1]
            q_proj = scorer.q_mlp(q)
            q_proj_exp = q_proj.unsqueeze(1).unsqueeze(1)
            
            child, logit = self._layer_logits_fast(q_proj_exp, nodes, h)
            
            total = scores.unsqueeze(-1) + F.log_softmax(logit, -1)
            topk_s, topk_idx = total.view(B, -1).topk(beam_size, -1)
            nodes = child.view(B, -1).gather(1, topk_idx)
            scores = topk_s
            
        return nodes

    @torch.no_grad()
    def rerank(self, qids, q, leaf):
        """
        返回: (results, compute_time_seconds)
        compute_time_seconds 仅包含 GPU 计算 (matmul + topk) 的时间，不包含 IO
        """
        results = []
        compute_time = 0.0
        
        leaf_nodes_batch = leaf.cpu().numpy()
        
        for i, qid in enumerate(qids):
            # 1. 准备阶段 (IO): 确定需要读取哪些文档 ID
            my_leaves = leaf_nodes_batch[i]
            doc_offsets = []
            for l in my_leaves:
                if l != self.res.pad:
                    docs = self.res.leaf_to_offsets.get(l, [])
                    if len(docs) > 0:
                        doc_offsets.append(docs)
            
            if not doc_offsets: continue
            doc_offsets = np.concatenate(doc_offsets)
            doc_offsets = np.unique(doc_offsets)
            if len(doc_offsets) == 0: continue
            
            # 2. IO 阶段 (读取 Memmap): 这里是从磁盘/缓存读取，比较慢，不计入时间
            # 注意：虽然切片操作很快，但在 memory map 上访问不连续内存会触发缺页中断读取磁盘
            cand_embs_np = self.res.doc_embeddings[doc_offsets]
            
            # 3. 传输阶段: 转 Tensor 并移至 GPU (通常这部分不计入纯算法延迟，或者即使计入也很快)
            # 为了严格符合"不计算IO"的要求，我们将 .to(DEVICE) 视为数据准备的一部分
            cand_embs = torch.from_numpy(cand_embs_np).to(DEVICE)
            curr_q_vec = q[i].unsqueeze(0)
            
            # 4. 计算阶段 (计时): 纯 GPU 操作
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_start = time.time()
            
            scores = torch.matmul(curr_q_vec, cand_embs.T).squeeze(0)
            
            k = min(EVAL_TOPK, scores.size(0))
            if k > 0:
                top_vals, top_inds = scores.topk(k)
                # 必须在这里结束计时，因为 topk 也是 GPU 操作
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_end = time.time()
                compute_time += (t_end - t_start)
                
                # 后处理结果
                top_vals = top_vals.cpu().numpy()
                top_inds = top_inds.cpu().numpy()
                
                for r in range(k):
                    real_doc_idx = doc_offsets[top_inds[r]]
                    doc_id_str = self.res.idx2docid.get(real_doc_idx, str(real_doc_idx))
                    results.append((qid, doc_id_str, top_vals[r]))
            else:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_end = time.time()
                compute_time += (t_end - t_start)
                    
        return results, compute_time

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
    
    os.makedirs(os.path.dirname(FINAL_OUTPUT_FILE), exist_ok=True)
    results_summary = []

    for model_path in CHECKPOINT_LIST:
        try:
            torch.cuda.empty_cache()
            
            retriever = NeuralRetriever(static_resources, model_path)
            model_name = os.path.basename(model_path)
            
            for beam_size in BEAM_SIZE_LIST:
                all_results = []
                # 累积纯计算时间 (Encoder + BeamSearch + RerankCompute)
                total_compute_time = 0.0
                total_queries = 0
                
                # ================= 核心循环 =================
                for qids, ids, mask in tqdm(dataloader, desc=f"{model_name} B={beam_size}", leave=False):
                    ids, mask = ids.to(DEVICE), mask.to(DEVICE)
                    
                    # 1. Encoder 计时
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.no_grad():
                        q = retriever.encoder(ids, mask)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    total_compute_time += (time.time() - t0)

                    # 2. Search 计时
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    t1 = time.time()
                    with torch.no_grad():
                        leaf = retriever.beam_search(q, beam_size=beam_size)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    total_compute_time += (time.time() - t1)

                    # 3. Rerank (内部计时，排除 IO)
                    with torch.no_grad():
                        res, rerank_time = retriever.rerank(qids, q, leaf)
                    
                    total_compute_time += rerank_time
                    total_queries += len(qids)
                    all_results.extend(res)

                # 计算 AQT (Average Query Time)
                aqt_ms = (total_compute_time / total_queries * 1000) if total_queries > 0 else 0.0
                
                # 评估指标
                run_data = defaultdict(list)
                for qid, did, s in all_results: run_data[qid].append((did, s))
                for qid in run_data: run_data[qid].sort(key=lambda x: x[1], reverse=True)
                
                mrr, recall = compute_metrics(qrels_data, run_data, k=100)
                results_summary.append({"Model": model_name, "Beam": beam_size, "MRR": mrr, "R": recall, "AQT": aqt_ms})
                
                print(f"{model_name} B={beam_size} | MRR: {mrr:.4f} | R: {recall:.4f} | AQT: {aqt_ms:.2f} ms")
                
                with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
                    f.writelines([f"{qid}\tQ0\t{did}\t0\t{s:.6f}\tTreeSearch\n" for qid, did, s in all_results])

        except Exception as e:
            print(f"Error {model_path}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*120)
    print(f"{'PERFORMANCE SUMMARY BY BEAM SIZE':^120}")
    print("="*120)
    
    beam_groups = defaultdict(list)
    for res in results_summary:
        beam_groups[res['Beam']].append(res)
    
    print(f"{'Beam Size':<10} {'Metric':<10} {'Best Value':<15} {'Model':<20} {'Other Metrics':<40}")
    print("-"*120)
    
    for beam in sorted(beam_groups.keys()):
        group = beam_groups[beam]
        best_mrr_run = max(group, key=lambda x: x['MRR'])
        print(f"{beam:<10} {'MRR':<10} {best_mrr_run['MRR']:.4f}<-(MAX) {'':<20} R: {best_mrr_run['R']:.4f}, AQT: {best_mrr_run['AQT']:.2f} ms | Model: {best_mrr_run['Model']}")
        
        best_recall_run = max(group, key=lambda x: x['R'])
        print(f"{beam:<10} {'Recall':<10} {best_recall_run['R']:.4f}<-(MAX) {'':<20} MRR: {best_recall_run['MRR']:.4f}, AQT: {best_recall_run['AQT']:.2f} ms | Model: {best_recall_run['Model']}")
        
        best_aqt_run = min(group, key=lambda x: x['AQT'])
        print(f"{beam:<10} {'AQT':<10} {best_aqt_run['AQT']:.2f}ms<-(MIN) {'':<20} MRR: {best_aqt_run['MRR']:.4f}, R: {best_aqt_run['R']:.4f} | Model: {best_aqt_run['Model']}")
        print("-"*120)

if __name__ == "__main__":
    main()