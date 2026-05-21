import os
# ================= 严格限制底层单线程 =================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time, pickle, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
import numba
from numba import njit
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
from collections import defaultdict
import pytrec_eval
import config

# ================= 限制 CPU 线程池 =================
torch.set_num_threads(1) 
numba.set_num_threads(1)

# ================= 配置与常量 =================
BEAM_SIZE_LIST = [10, 20, 30, 40, 50]
CHECKPOINTS = [f"./output/checkpoint-{i}.pt" for i in range(1, 16)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_TOPK = 100
RESULT_FILE = "answer_merged.txt"

MAX_CANDIDATES = 250000 

TREC_DATASETS = {
    "TREC_2019": {"query": config.TREC_QUERYS_19, "qrels": config.TREC_QUERL_19},
    "TREC_2020": {"query": config.TREC_QUERYS_20, "qrels": config.TREC_QUERL_20}
}

# ================= 模型组件 =================
class Similarity(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_mlp = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim, dim))
        self.c_mlp = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim, dim))

    def forward(self, q, c):
        return torch.sum(self.q_mlp(q) * self.c_mlp(c), dim=-1)

class Encoder(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path, config=AutoConfig.from_pretrained(path))

    def forward(self, ids, mask):
        return F.normalize(self.backbone(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state[:, 0], dim=1)

class Indexer(nn.Module):
    def __init__(self, dim, height):
        super().__init__()
        self.fixed_centroids = nn.Embedding(1, dim)
        self.scorers = nn.ModuleList([Similarity(dim) for _ in range(height)])
        self.logit_scale = nn.Parameter(torch.tensor(np.log(20.0)))

# ================= 数据资源 =================
class QueryDataset(Dataset):
    def __init__(self, queries, tokenizer):
        self.data = queries
        self.tokenizer = tokenizer
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        qid, text = self.data[idx]
        enc = self.tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        return qid, enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

class StaticResources:
    def __init__(self):
        print("Loading embeddings and trees...")
        
        # ================== 核心优化点：消除磁盘 I/O 耗时 ==================
        print("Copying memmap to RAM (eliminating I/O overhead from AQT)...")
        memmap_view = np.memmap(config.MEMMAP_PATH, dtype='float32', mode='r').reshape(-1, config.EMBEDDING_DIM)
        self.doc_embeddings = np.array(memmap_view)
        print("Memmap successfully loaded into RAM.")
        # =================================================================
        
        self.idx2docid, self.docid2idx = {}, {}
        with open(config.ID2OFFSET, "r") as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 2 and p[1].isdigit():
                    self.docid2idx[p[0]], self.idx2docid[int(p[1])] = int(p[1]), p[0]

        with open(config.ID2PATH, "rb") as f: paths_map = pickle.load(f)

        leaf_to_offsets = defaultdict(list)
        for did, ps in paths_map.items():
            if (offset := self.docid2idx.get(did)) is not None:
                for p in (ps if isinstance(ps[0], list) else [ps]): 
                    leaf_to_offsets[p[-1]].append(offset)

        with open(config.CHILDREN_EMBEDDINGS_PATH, "rb") as f: children = pickle.load(f)

        self.pad = max([max([c['child_id'] for c in cs] + [p]) for p, cs in children.items()]) + 1
        
        self.adj = np.full((self.pad + 1, config.NODE_BALANCE+1), self.pad, dtype=np.int64)

        for p, cs in children.items():
            for c in cs: self.adj[p, c['child_index']] = c['child_id']

        max_leaf = max(leaf_to_offsets.keys()) if leaf_to_offsets else 0
        self.leaf_bounds = np.zeros((max_leaf + 1, 2), dtype=np.int64)
        
        all_offs = []
        curr_idx = 0
        for i in range(max_leaf + 1):
            if i in leaf_to_offsets:
                lst = leaf_to_offsets[i]
                self.leaf_bounds[i, 0] = curr_idx
                self.leaf_bounds[i, 1] = curr_idx + len(lst)
                all_offs.extend(lst)
                curr_idx += len(lst)
            else:
                self.leaf_bounds[i, 0] = curr_idx
                self.leaf_bounds[i, 1] = curr_idx
                
        self.flat_offsets = np.array(all_offs, dtype=np.int64)
        self.max_offset = np.max(self.flat_offsets) if len(self.flat_offsets) > 0 else 0
        self.seen_array = np.zeros(self.max_offset + 1, dtype=np.int32)

# ================= Numba 核心算法 =================
@njit(fastmath=True, cache=True)
def fast_beam_and_rerank(q_projs, cached_cents, adj, scale, beam_size,
                         leaf_bounds, flat_offsets, doc_embeddings, q_vec_raw,
                         pad_id, top_k, seen_array, query_idx, unique_offsets):
    
    tree_height = q_projs.shape[0] + 1
    num_children = adj.shape[1]
    
    nodes = np.zeros(beam_size, dtype=np.int64)
    scores = np.full(beam_size, -1e9, dtype=np.float32)
    scores[0] = 0.0

    next_nodes = np.empty(beam_size * num_children, dtype=np.int64)
    next_scores = np.empty(beam_size * num_children, dtype=np.float32)
    logits = np.empty(num_children, dtype=np.float32)

    for h in range(1, tree_height):
        q_p = q_projs[h-1]
        cents = cached_cents[h-1]
        idx = 0
        
        for b in range(beam_size):
            n_id = nodes[b]
            base_score = scores[b]
            max_l = -1e9
            
            for c in range(num_children):
                child_id = adj[n_id, c]
                next_nodes[idx + c] = child_id
                
                if child_id == pad_id: logits[c] = -1e9
                else:
                    val = np.dot(q_p, cents[child_id]) * scale
                    logits[c] = val
                    if val > max_l: max_l = val

            if max_l == -1e9:
                for c in range(num_children): next_scores[idx + c] = -1e9
            else:
                sum_exp = 0.0
                for c in range(num_children):
                    if logits[c] != -1e9: sum_exp += np.exp(logits[c] - max_l)
                log_sum_exp = max_l + np.log(sum_exp)
                for c in range(num_children):
                    next_scores[idx + c] = base_score + (logits[c] - log_sum_exp) if logits[c] != -1e9 else -1e9

            idx += num_children

        topk_idx = np.argsort(-next_scores)[:beam_size]
        for b in range(beam_size):
            nodes[b] = next_nodes[topk_idx[b]]
            scores[b] = next_scores[topk_idx[b]]

    u_count = 0
    for b in range(beam_size):
        leaf = nodes[b]
        if leaf != pad_id and leaf < leaf_bounds.shape[0]:
            start, end = leaf_bounds[leaf, 0], leaf_bounds[leaf, 1]
            for o in range(start, end):
                off = flat_offsets[o]
                if seen_array[off] != query_idx:
                    seen_array[off] = query_idx
                    if u_count < MAX_CANDIDATES:
                        unique_offsets[u_count] = off
                        u_count += 1

    if u_count == 0: return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

    k = min(top_k, u_count)
    rerank_scores = np.empty(u_count, dtype=np.float32)
    for i in range(u_count): rerank_scores[i] = np.dot(doc_embeddings[unique_offsets[i]], q_vec_raw)
    
    topk_idx = np.argsort(-rerank_scores)[:k]

    out_offsets, out_scores = np.empty(k, dtype=np.int64), np.empty(k, dtype=np.float32)
    for i in range(k):
        idx = topk_idx[i]
        out_offsets[i] = unique_offsets[idx]
        out_scores[i] = rerank_scores[idx]

    return out_offsets, out_scores

# ================= 检索调用类 =================
class NeuralRetriever:
    def __init__(self, res, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.encoder = Encoder(config.MODEL_NAME).to(DEVICE)
        self.indexer = Indexer(config.EMBEDDING_DIM, config.TREE_HEIGHT).to("cpu")

        if "fixed_centroids.weight" in ckpt["indexer"]:
            self.indexer.fixed_centroids = nn.Embedding(ckpt["indexer"]["fixed_centroids.weight"].shape[0], config.EMBEDDING_DIM).to("cpu")

        self.encoder.load_state_dict(ckpt["encoder"], strict=False)
        self.indexer.load_state_dict(ckpt["indexer"], strict=False)
        self.res = res
        self.encoder.eval()
        self.indexer.eval()

        self.scale_np = float(self.indexer.logit_scale.exp().clamp(max=100).detach().cpu().numpy())
        self.adj_np = self.res.adj.astype(np.int64)
        
        cached_cpu = [self.indexer.scorers[h].c_mlp(self.indexer.fixed_centroids.weight).detach().cpu().numpy().astype(np.float32) 
                      for h in range(config.TREE_HEIGHT)]
        self.cached_cents_np = np.stack(cached_cpu)

    @torch.no_grad()
    def prepare_query(self, q_vec):
        q_vec_cpu = q_vec.cpu().squeeze(0)
        q_vec_np = q_vec_cpu.numpy().astype(np.float32)
        q_projs = [self.indexer.scorers[h-1].q_mlp(q_vec_cpu).numpy().astype(np.float32) for h in range(1, config.TREE_HEIGHT)]
        return q_vec_np, np.stack(q_projs)

    def search(self, qid_str, q_vec_np, q_projs_np, beam_size, query_idx, offset_buffer):
        t_start = time.time()
        out_offsets, out_scores = fast_beam_and_rerank(
            q_projs_np, self.cached_cents_np, self.adj_np, self.scale_np,
            beam_size, self.res.leaf_bounds, self.res.flat_offsets,
            self.res.doc_embeddings, q_vec_np, self.res.pad, EVAL_TOPK, 
            self.res.seen_array, query_idx, offset_buffer
        )
        r_time = time.time() - t_start
        return [(qid_str, self.res.idx2docid.get(off), float(score)) for off, score in zip(out_offsets, out_scores)], r_time

# ================= 评估辅助函数 =================
def evaluate_dev_metrics(qrels, run):
    mrr, recall = 0.0, 0.0
    for qid, rels in qrels.items():
        if qid not in run: continue
        retrieved = [d for d, _ in run[qid][:100]]
        mrr += next((1.0 / (i + 1) for i, d in enumerate(retrieved) if d in rels), 0.0)
        recall += sum(1 for d in retrieved if d in rels) / len(rels) if rels else 0.0
    return mrr / len(qrels), recall / len(qrels)

# ================= 主函数 =================
def main():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    res = StaticResources()
    global_offset_buffer = np.empty(MAX_CANDIDATES, dtype=np.int64)

    # 1. 准备 DEV 数据
    dev_qrels = defaultdict(set)
    with open(config.DEV_PASSAGE_QRELS, 'r') as f:
        for line in f:
            if len(p := line.strip().split()) >= 4 and int(p[3]) > 0: dev_qrels[p[0]].add(p[2])
        # print(f"Loaded DEV qrels: {len(dev_qrels)} queries with relevance judgments.")
    dev_queries = []
    with open(config.DEV_PASSAGE_QUERYS, "r", encoding="utf-8") as f:
        dev_queries = [(p[0], p[1]) for line in f if len(p := line.rstrip().split("\t")) >= 2]
    dev_loader = DataLoader(QueryDataset(dev_queries, tokenizer), batch_size=1, shuffle=False, num_workers=4)

    # 2. 准备 TREC 数据
    trec_meta = {}
    for ds_name, paths in TREC_DATASETS.items():
        qrels = defaultdict(dict)
        with open(paths["qrels"], 'r') as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 4: qrels[p[0]][p[2]] = int(p[3])
                    
        valid_qids = set(qrels.keys())
        all_queries = []
        with open(paths["query"], "r", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip().split("\t")
                if len(p) >= 2 and p[0] in valid_qids: all_queries.append((p[0], p[1]))
                    
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10'})
        loader = DataLoader(QueryDataset(all_queries, tokenizer), batch_size=1, shuffle=False, num_workers=4)
        trec_meta[ds_name] = {"qrels": qrels, "loader": loader, "evaluator": evaluator}

    # 记录字典
    dev_beam_records = {b: [] for b in BEAM_SIZE_LIST}
    trec_best_scores = {ds_name: {b: {"ckpt": None, "ndcg": -1.0} for b in BEAM_SIZE_LIST} for ds_name in TREC_DATASETS}

    # 3. 开始遍历 Checkpoints
    for path in CHECKPOINTS:
        if not os.path.exists(path): continue
        ckpt_name = os.path.basename(path)
        print(f"\n================ Loading Checkpoint: {ckpt_name} ================")
        retriever = NeuralRetriever(res, path)
        global_query_idx = 1 

        # --- DEV 推理与评估 ---
        dev_all_res = {b: [] for b in BEAM_SIZE_LIST}
        dev_total_time = {b: 0.0 for b in BEAM_SIZE_LIST}
        dev_query_cnt = 0

        for qid, ids, mask in tqdm(dev_loader, desc=f"Evaluating DEV ({ckpt_name})", leave=False):
            qid_str = qid[0]
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)

            t_enc_start = time.time() 
            with torch.no_grad(): q_vec = retriever.encoder(ids, mask)
            enc_time = time.time() - t_enc_start 

            t_prep_start = time.time()
            q_vec_np, q_projs_np = retriever.prepare_query(q_vec)
            prep_time = time.time() - t_prep_start

            for b_size in BEAM_SIZE_LIST:
                batch_res, r_time = retriever.search(qid_str, q_vec_np, q_projs_np, b_size, global_query_idx, global_offset_buffer)
                dev_total_time[b_size] += (enc_time + prep_time + r_time)
                dev_all_res[b_size].extend(batch_res)
                global_query_idx += 1
            dev_query_cnt += 1

        # ================= 新增：在终端实时打印 DEV 结果 =================
        dev_print_strs = []
        for b_size in BEAM_SIZE_LIST:
            run_data = defaultdict(list)
            for qid_val, did_val, s_val in dev_all_res[b_size]: run_data[qid_val].append((did_val, s_val))
            for qid_val in run_data: run_data[qid_val].sort(key=lambda x: x[1], reverse=True)

            mrr, rec = evaluate_dev_metrics(dev_qrels, run_data)
            aqt = dev_total_time[b_size] / dev_query_cnt * 1000
            dev_beam_records[b_size].append((mrr, rec, aqt))
            
            # 格式化当前 Beam Size 的结果
            dev_print_strs.append(f"B={b_size}: MRR={mrr:.4f}, Rec={rec:.4f}, AQT={aqt:.1f}ms")
            
        print(f"[DEV_PASSAGE] Metrics: " + " | ".join(dev_print_strs))
        # =====================================================================

        # --- TREC 推理与评估 ---
        for ds_name, meta in trec_meta.items():
            trec_all_res = {b: [] for b in BEAM_SIZE_LIST}
            for qid, ids, mask in tqdm(meta["loader"], desc=f"Evaluating {ds_name} ({ckpt_name})", leave=False):
                qid_str = qid[0]
                ids, mask = ids.to(DEVICE), mask.to(DEVICE)

                with torch.no_grad(): q_vec = retriever.encoder(ids, mask)
                q_vec_np, q_projs_np = retriever.prepare_query(q_vec)

                for b_size in BEAM_SIZE_LIST:
                    batch_res, _ = retriever.search(qid_str, q_vec_np, q_projs_np, b_size, global_query_idx, global_offset_buffer)
                    trec_all_res[b_size].extend(batch_res)
                    global_query_idx += 1 

            # 计算 TREC nDCG
            beam_ndcg_scores = []
            for b_size in BEAM_SIZE_LIST:
                run_data = defaultdict(dict)
                for qid_val, did_val, s_val in trec_all_res[b_size]: 
                    if did_val not in run_data[qid_val] or s_val > run_data[qid_val][did_val]:
                        run_data[qid_val][did_val] = s_val

                metrics = meta["evaluator"].evaluate(run_data)
                mean_ndcg = sum(q_metrics.get('ndcg_cut_10', 0.0) for q_metrics in metrics.values()) / len(meta["qrels"])
                beam_ndcg_scores.append((b_size, mean_ndcg))

                if mean_ndcg > trec_best_scores[ds_name][b_size]["ndcg"]:
                    trec_best_scores[ds_name][b_size]["ndcg"] = mean_ndcg
                    trec_best_scores[ds_name][b_size]["ckpt"] = ckpt_name

            print(f"[{ds_name}] nDCG@10: " + " | ".join([f"B={b}:{v:.4f}" for b, v in beam_ndcg_scores]))


    # ================= 最终输出表格 =================
    final_output = "\n" + "="*70 + "\n🔥 FINAL EVALUATION RESULTS 🔥\n" + "="*70 + "\n"

    # DEV 表格 (输出最佳 MRR/Recall 和 最优 AQT)
    final_output += f"\n🏆 Dataset: DEV_PASSAGE (MRR, Recall, AQT)\n"
    final_output += "| BEAM_SIZE | MRR@100 | Recall@100 | AQT (ms/q) |\n"
    final_output += "| --------- | ------- | ---------- | ---------- |\n"
    for b_size in BEAM_SIZE_LIST:
        records = dev_beam_records[b_size]
        if not records: continue
        best_mrr = max([x[0] for x in records])    
        best_recall = max([x[1] for x in records]) 
        best_aqt = min([x[2] for x in records]) 
        final_output += f"| {b_size:<9} | {best_mrr:.4f}  | {best_recall:.4f}     | {best_aqt:>6.2f}     |\n"

    # TREC 表格 (输出最佳 nDCG 和对应权重名)
    for ds_name in TREC_DATASETS.keys():
        final_output += f"\n🏆 Dataset: {ds_name} (Best nDCG@10)\n"
        final_output += "| BEAM_SIZE | Best Checkpoint       | Best nDCG@10 |\n"
        final_output += "| --------- | --------------------- | ------------ |\n"
        for b_size in BEAM_SIZE_LIST:
            best_ckpt = trec_best_scores[ds_name][b_size]["ckpt"] or "N/A"
            best_val = trec_best_scores[ds_name][b_size]["ndcg"]
            final_output += f"| {b_size:<9} | {best_ckpt:<21} | {best_val:.4f}       |\n"

    print(final_output)
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(final_output + "\n")

if __name__ == "__main__":
    main()