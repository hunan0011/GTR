import os, time, pickle, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
from collections import defaultdict
import config

# ================= 配置与常量 =================
BEAM_SIZE_LIST = [10, 20, 30, 40, 50]
CHECKPOINTS = [f"/home/power/jiangyutao/GTR/output/checkpoint-{i}.pt" for i in range(1, 16)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_TOPK = 100

# ================= 模型组件 =================
class Similarity(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_mlp = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim, dim))
        self.c_mlp = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim, dim))
    def forward(self, q, c): return torch.sum(self.q_mlp(q) * self.c_mlp(c), dim=-1)

class Encoder(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path, config=AutoConfig.from_pretrained(path))
    def forward(self, ids, mask): return F.normalize(self.backbone(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state[:, 0], dim=1)

class Indexer(nn.Module):
    def __init__(self, dim, height):
        super().__init__()
        self.fixed_centroids = nn.Embedding(1, dim)
        self.scorers = nn.ModuleList([Similarity(dim) for _ in range(height)])
        self.logit_scale = nn.Parameter(torch.tensor(np.log(20.0)))
        self.adjacency = None

# ================= 数据资源 =================
class QueryDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip().split("\t")
                if len(p) >= 2: self.data.append((p[0], p[1]))
        self.tokenizer = tokenizer
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        qid, text = self.data[idx]
        enc = self.tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        return qid, enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

class StaticResources:
    def __init__(self):
        self.doc_embeddings = np.memmap(config.MEMMAP_PATH, dtype='float32', mode='r').reshape(-1, config.EMBEDDING_DIM)
        self.idx2docid, self.docid2idx = {}, {}
        with open(config.ID2OFFSET, "r") as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 2 and p[1].isdigit():
                    self.docid2idx[p[0]], self.idx2docid[int(p[1])] = int(p[1]), p[0]
        
        with open(config.ID2PATH, "rb") as f: paths_map = pickle.load(f)
        self.leaf_to_offsets = defaultdict(list)
        for did, ps in paths_map.items():
            offset = self.docid2idx.get(did)
            if offset is not None:
                for p in (ps if isinstance(ps[0], list) else [ps]): self.leaf_to_offsets[p[-1]].append(offset)
        for k in self.leaf_to_offsets: self.leaf_to_offsets[k] = np.array(self.leaf_to_offsets[k], dtype=np.int64)

        with open(config.CHILDREN_EMBEDDINGS_PATH, "rb") as f: children = pickle.load(f)
        self.pad = max([max([c['child_id'] for c in cs] + [p]) for p, cs in children.items()]) + 1
        self.adj = torch.full((self.pad + 1, config.NODE_BALANCE), self.pad, dtype=torch.long).to(DEVICE)
        for p, cs in children.items():
            for c in cs: self.adj[p, c['child_index']] = c['child_id']

# ================= 检索核心 =================
class NeuralRetriever:
    def __init__(self, res, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        self.encoder = Encoder(config.MODEL_NAME).to(DEVICE)
        self.indexer = Indexer(config.EMBEDDING_DIM, config.TREE_HEIGHT).to(DEVICE)
        if "fixed_centroids.weight" in ckpt["indexer"]:
            self.indexer.fixed_centroids = nn.Embedding(ckpt["indexer"]["fixed_centroids.weight"].shape[0], config.EMBEDDING_DIM).to(DEVICE)
        self.encoder.load_state_dict(ckpt["encoder"], strict=False)
        self.indexer.load_state_dict(ckpt["indexer"], strict=False)
        self.res, self.scale = res, self.indexer.logit_scale.exp().clamp(max=100)
        self.cached_cents = [self.indexer.scorers[h].c_mlp(self.indexer.fixed_centroids.weight) for h in range(config.TREE_HEIGHT)]
        self.encoder.eval(); self.indexer.eval()

    @torch.no_grad()
    def beam_search(self, q, beam_size):
        B = q.size(0)
        nodes = torch.zeros((B, beam_size), dtype=torch.long, device=DEVICE)
        scores = torch.full((B, beam_size), -1e9, device=DEVICE); scores[:, 0] = 0
        for h in range(1, config.TREE_HEIGHT):
            q_proj = self.indexer.scorers[h-1].q_mlp(q).unsqueeze(1).unsqueeze(1)
            child_ids = self.res.adj[nodes.view(-1)].view(B, beam_size, -1)
            child_embs = F.embedding(child_ids, self.cached_cents[h-1])
            logits = torch.sum(q_proj * child_embs, dim=-1) * self.scale
            total = scores.unsqueeze(-1) + F.log_softmax(logits, -1)
            scores, topk_idx = total.view(B, -1).topk(beam_size, -1)
            nodes = child_ids.view(B, -1).gather(1, topk_idx)
        return nodes

    @torch.no_grad()
    def rerank(self, qids, q_vecs, leaves):
        results, compute_time = [], 0.0
        for i, qid in enumerate(qids):
            offsets = np.unique(np.concatenate([self.res.leaf_to_offsets[l] for l in leaves[i].cpu().numpy() if l != self.res.pad and l in self.res.leaf_to_offsets] or [[]]))
            if len(offsets) == 0: continue
            
            cand_embs = torch.from_numpy(self.res.doc_embeddings[offsets]).to(DEVICE)
            curr_q = q_vecs[i].unsqueeze(0)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_start = time.time()
            scores = torch.matmul(curr_q, cand_embs.T).squeeze(0)
            k = min(EVAL_TOPK, scores.size(0))
            top_vals, top_inds = scores.topk(k)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            compute_time += (time.time() - t_start)

            for r in range(k):
                results.append((qid, self.res.idx2docid.get(offsets[top_inds[r].item()]), top_vals[r].item()))
        return results, compute_time

# ================= 评估工具 =================
def evaluate(qrels, run):
    mrr, recall, cnt = 0.0, 0.0, len(qrels)
    for qid, rels in qrels.items():
        if qid not in run: continue
        retrieved = [d for d, _ in run[qid][:100]]
        mrr += next((1.0/(i+1) for i, d in enumerate(retrieved) if d in rels), 0.0)
        recall += sum(1 for d in retrieved if d in rels) / len(rels) if rels else 0.0
    return mrr / cnt, recall / cnt

def main():
    qrels = defaultdict(set)
    with open(config.DEV_PASSAGE_QRELS, 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 4 and int(p[3]) > 0: qrels[p[0]].add(p[2])
    
    res = StaticResources()
    loader = DataLoader(QueryDataset(config.DEV_PASSAGE_QUERYS, AutoTokenizer.from_pretrained(config.MODEL_NAME)), 
                        batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda b: (list(zip(*b))[0], torch.stack(list(zip(*b))[1]), torch.stack(list(zip(*b))[2])))

    for path in CHECKPOINTS:
        if not os.path.exists(path): continue
        retriever = NeuralRetriever(res, path)
        for b_size in BEAM_SIZE_LIST:
            all_res, total_c_time = [], 0.0
            for qids, ids, mask in tqdm(loader, desc=f"{os.path.basename(path)} B={b_size}"):
                ids, mask = ids.to(DEVICE), mask.to(DEVICE)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = time.time()
                q = retriever.encoder(ids, mask)
                leaf = retriever.beam_search(q, b_size)
                total_c_time += (time.time() - t0)
                
                batch_res, r_time = retriever.rerank(qids, q, leaf)
                all_res.extend(batch_res); total_c_time += r_time

            run_data = defaultdict(list)
            for qid, did, s in all_res: run_data[qid].append((did, s))
            for qid in run_data: run_data[qid].sort(key=lambda x: x[1], reverse=True)
            
            mrr, rec = evaluate(qrels, run_data)
            print(f"B={b_size} | MRR: {mrr:.4f} | R: {rec:.4f} | AQT: {total_c_time/len(loader)*1000:.2f} ms")

if __name__ == "__main__":
    main()