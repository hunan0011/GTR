import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
from collections import defaultdict
import argparse
import config

# ================= 参数配置 =================
parser = argparse.ArgumentParser()
parser.add_argument('--BEAM_SIZE', type=int, default=50, help="Beam search width")
parser.add_argument('--MODEL_PATH', type=str, required=True, help="Path to the trained checkpoint")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置
BASE_MODEL_PATH = config.MODEL_NAME
MEMMAP_PATH = config.MEMMAP_PATH
ID2INDEX_PATH = config.ID2OFFSET
QUERY_PATH = config.DEV_DOC_TRAIN_QUERIES
OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "dev_results_layerwise.trec")

# 推理参数
BATCH_SIZE = 128
NUM_WORKERS = 4
EVAL_TOPK = 100
BEAM_SIZE = args.BEAM_SIZE

# 模型参数
EMBEDDING_DIM = config.EMBEDDING_DIM
TREE_HEIGHT = config.TREE_HEIGHT
NODE_BALANCE = config.NODE_BALANCE
QUERY_INSTRUCTION = "" 

# ================= 模型组件 (Strictly Aligned with model.py) =================

# [核心修改] 这里的 Similarity 必须与 model.py 完全一致
class Similarity(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        # 定义 MLP 结构：Linear -> LayerNorm -> GELU -> Dropout -> Linear
        # 注意：虽然推理时 dropout 不生效，但结构必须定义，否则 load_state_dict 会报错找不到 key
        self.q_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim)
        )
        
        self.c_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, query, candidates):
        """
        推理时的维度略有不同，但 Linear 层只处理最后一维，所以可以直接复用
        query: [Batch, Beam, 1, Dim]
        candidates: [Batch, Beam, NodeBalance, Dim]
        """
        # 1. MLP 投影 (支持广播)
        q = self.q_mlp(query)      # [B, K, 1, D]
        c = self.c_mlp(candidates) # [B, K, NB, D]
        
        # 2. 点积相似度: Sum(q * c)
        # Broadcasting: (B, K, 1, D) * (B, K, NB, D) -> (B, K, NB, D)
        scores = torch.sum(q * c, dim=-1) # -> [B, K, NB]
        
        return scores

class Encoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=cfg)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        emb = out.last_hidden_state[:, 0]
        return F.normalize(emb, dim=1)

class Indexer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 占位 embedding
        self.fixed_centroids = nn.Embedding(1, EMBEDDING_DIM)

        # [对齐] 使用 MLP 版 Similarity
        self.scorers = nn.ModuleList([
            Similarity(EMBEDDING_DIM)
            for _ in range(TREE_HEIGHT)
        ])
        self.logit_scale = nn.Parameter(torch.tensor(np.log(20.0)))
        self.adjacency = None

# ================= 数据加载 =================
class QueryDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"Loading queries from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split("\t")
                if len(parts) >= 2:
                    qid, text = parts[:2]
                    self.data.append((qid, QUERY_INSTRUCTION + text))

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

class DocEmbeddingLookup:
    def __init__(self, memmap_path: str, id2idx_path: str):
        self.emb = np.memmap(memmap_path, dtype="float32", mode="r")
        dim = EMBEDDING_DIM
        self.emb = self.emb.reshape(-1, dim)
        self.size = self.emb.shape[0]

        self.docid2idx = {}
        self.docids = []
        print(f"Loading docid map from {id2idx_path}...")
        with open(id2idx_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[1].isdigit():
                    idx = int(parts[1])
                    if 0 <= idx < self.size:
                        self.docid2idx[parts[0]] = idx
                        self.docids.append(parts[0])
        self.docids = np.array(self.docids)

    def get(self, docids):
        pos = []
        valid_docids = []
        for d in docids:
            idx = self.docid2idx.get(d)
            if idx is not None:
                pos.append(idx)
                valid_docids.append(d)
        if not pos: return [], None
        return valid_docids, torch.from_numpy(self.emb[np.asarray(pos, dtype=np.int64)])

# ================= 检索器逻辑 =================
class NeuralRetriever:
    def __init__(self, model_path):
        print(f"Loading checkpoint: {model_path}")
        ckpt = torch.load(model_path, map_location=DEVICE)

        self.encoder = Encoder(BASE_MODEL_PATH).to(DEVICE)
        self.indexer = Indexer().to(DEVICE)

        # 加载 Encoder
        self.encoder.load_state_dict(ckpt["encoder"], strict=False)

        # 加载 Indexer
        indexer_state = ckpt["indexer"]
        
        # 检查参数名是否匹配 (Debugging Help)
        keys_in_ckpt = [k for k in indexer_state.keys() if 'scorers.0' in k]
        if keys_in_ckpt and 'q_mlp' not in keys_in_ckpt[0]:
            print("\n[WARNING] Checkpoint 似乎不包含 MLP 参数 (q_mlp/c_mlp)！")
            print(f"Checkpoint 里的 key 样例: {keys_in_ckpt[0]}")
            print("如果这是旧的 Linear Checkpoint，请重新训练，否则效果会很差。\n")

        # 动态调整 embedding 大小
        if "fixed_centroids.weight" in indexer_state:
            num_nodes = indexer_state["fixed_centroids.weight"].shape[0]
            self.indexer.fixed_centroids = nn.Embedding(num_nodes, EMBEDDING_DIM).to(DEVICE)

        # 加载参数 (现在 inference.py 和 model.py 结构一致，应该能完美加载)
        self.indexer.load_state_dict(indexer_state, strict=False)
        print("Indexer weights loaded.")

        self.encoder.eval()
        self.indexer.eval()
        self.scale = self.indexer.logit_scale.exp().clamp(max=100)
        
        self._load_tree()
        self._load_leaf_docs()
        self.doc_lookup = DocEmbeddingLookup(MEMMAP_PATH, ID2INDEX_PATH)

    def _load_tree(self):
        print("Loading tree structure...")
        with open(config.CHILDREN_EMBEDDINGS_PATH, "rb") as f:
            children = pickle.load(f)
        max_id = 0
        for p, cs in children.items():
            max_id = max(max_id, p)
            for c in cs: max_id = max(max_id, c["child_id"])
        
        self.pad = max_id + 1
        adj = torch.full((self.pad + 1, NODE_BALANCE), self.pad, dtype=torch.long)
        for p, cs in children.items():
            for i, c in enumerate(sorted(cs, key=lambda x: x["child_index"])):
                if i < NODE_BALANCE: adj[p, i] = c["child_id"]
        self.indexer.adjacency = adj.to(DEVICE)

    def _load_leaf_docs(self):
        print("Loading leaf mappings...")
        with open(config.ID2PATH, "rb") as f:
            m = pickle.load(f)
        self.leaf_docs = defaultdict(list)
        for d, paths in m.items():
            # 兼容处理
            if len(paths) > 0 and isinstance(paths[0], int): paths = [paths]
            for p in paths:
                self.leaf_docs[p[-1]].append(d)

    def _layer_logits(self, q, parents, h):
        """
        q: [Batch, Dim]
        parents: [Batch, Beam]
        """
        B, K = parents.shape
        child = self.indexer.adjacency[parents.view(-1)].view(B, K, -1)
        cent = self.indexer.fixed_centroids(child) # [B, K, NB, D]

        # 调整 Query 维度: [B, K, 1, D]
        # MLP 支持任意维度，只要最后一维是 Dim
        q_expand = q[:, None, None, :].expand(B, K, 1, EMBEDDING_DIM)
        
        # 计算 MLP 相似度
        scores = self.indexer.scorers[h - 1](q_expand, cent) # [B, K, NB]
        
        return child, (scores * self.scale)

    @torch.no_grad()
    def beam_search(self, q):
        B = q.size(0)
        nodes = torch.zeros((B, BEAM_SIZE), dtype=torch.long, device=DEVICE)
        scores = torch.full((B, BEAM_SIZE), -1e9, device=DEVICE)
        scores[:, 0] = 0 

        for h in range(1, TREE_HEIGHT):
            child, logit = self._layer_logits(q, nodes, h)
            lp = F.log_softmax(logit, -1)
            total = scores.unsqueeze(-1) + lp
            
            flat_score = total.view(B, -1)
            topk_s, topk_idx = flat_score.topk(BEAM_SIZE, -1)
            
            flat_child = child.view(B, -1)
            nodes = flat_child.gather(1, topk_idx)
            scores = topk_s
        return nodes

    @torch.no_grad()
    def rerank(self, qids, q, leaf):
        leaf = leaf.view(-1).cpu().numpy()
        leaf = leaf[leaf != self.pad]

        docs = []
        for l in np.unique(leaf):
            docs.extend(self.leaf_docs.get(l, []))
        
        # [多路径去重] 关键步骤
        docs = list(set(docs))
        
        if not docs: return []
        docids, dv = self.doc_lookup.get(docs)
        if dv is None: return []

        dv = dv.to(DEVICE)
        score = torch.matmul(q, dv.T)
        k = min(EVAL_TOPK, score.size(1))
        v, i = score.topk(k, 1)

        res = []
        for b, qid in enumerate(qids):
            for r in range(k):
                res.append((qid, docids[i[b, r]], v[b, r].item()))
        return res

# ================= 主流程 =================
def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    dataset = QueryDataset(QUERY_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    retriever = NeuralRetriever(args.MODEL_PATH)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for qids, ids, mask in tqdm(dataloader, desc="Inference"):
            ids = ids.to(DEVICE)
            mask = mask.to(DEVICE)
            with torch.no_grad():
                q = retriever.encoder(ids, mask)
                leaf = retriever.beam_search(q)
                res = retriever.rerank(qids, q, leaf)
            for qid, did, s in res:
                f.write(f"{qid}\tQ0\t{did}\t0\t{s:.6f}\tTreeSearch\n")
    print("Done.")

if __name__ == "__main__":
    main()