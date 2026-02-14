import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import pickle
import tqdm

class MsMarcoDataset(Dataset):
    def __init__(self, queries_path, qrels_path,  docid2path_path, leaf2docs_path, 
                 doc_embedding_path, docid_to_index_path, neg_num=1, embedding_dim=768):
        self.neg_num = neg_num
        
        # 1. 加载查询 (Query)
        self.queries = {}
        print(f"Loading queries from {queries_path}...")
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2: self.queries[parts[0]] = parts[1]

        # 2. 加载树结构映射 (Path & Leaf Mappings)
        print(f"Loading tree mappings from {docid2path_path}...")
        with open(docid2path_path, "rb") as f:
            self.docid2path = pickle.load(f)
        with open(leaf2docs_path, "rb") as f:
            self.leaf2docs = pickle.load(f)

        # 3. 加载 Qrels 并执行多路径样本展开 (Path Expansion)
        self.samples = [] 
        print(f"Loading qrels and expanding multi-path samples...")
        
        with open(qrels_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    qid, docid = parts[0], parts[2]
                    
                    if qid not in self.queries:
                        continue
                        
                    paths = self.docid2path.get(docid)
                    if not paths:
                        continue
                    
                    # 兼容单路径格式
                    if isinstance(paths[0], int):
                        paths = [paths]
                    
                    # 核心逻辑：一条路径生成一个样本
                    for path_idx in range(len(paths)):
                        self.samples.append({
                            'qid': qid,
                            'pos_docid': docid,
                            'path_idx': path_idx
                        })

        print(f"Dataset loaded. Total samples after expansion: {len(self.samples)}")

        # 4. 加载 DocID -> Memmap Index 索引
        self.docid2index = {}
        print(f"Loading DocID Index from {docid_to_index_path}...")
        with open(docid_to_index_path, "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f, desc="Loading Indices", disable=True):
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "offset":
                    try:
                        self.docid2index[parts[0]] = int(parts[1])
                    except ValueError:
                        continue
        
        self._docid_list = list(self.docid2index.keys())

        # 5. 初始化 Memmap
        self.embedding_dim = embedding_dim
        self.embedding_path = doc_embedding_path
        file_size = os.path.getsize(doc_embedding_path)
        float32_size = 4
        self.num_docs = file_size // (self.embedding_dim * float32_size)
        
        self.doc_embeddings = np.memmap(
            self.embedding_path, dtype='float32', mode='r', 
            shape=(self.num_docs, self.embedding_dim)
        )

    def get_doc_embedding(self, docid):
        idx = self.docid2index.get(docid)
        if idx is not None and 0 <= idx < self.num_docs:
            return self.doc_embeddings[idx].copy()
        else:
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        qid = sample['qid']
        pos_docid = sample['pos_docid']
        path_idx = sample['path_idx']
        
        query_text = self.queries[qid]
        pos_emb = self.get_doc_embedding(pos_docid)

        # --- 负采样逻辑 ---
        neg_docids = []
        
        # A. 硬负例 (Leaf-sharing Hard Negatives)
        paths = self.docid2path.get(pos_docid)
        if isinstance(paths[0], int): paths = [paths]
            
        current_path = paths[path_idx]
        leaf_node = current_path[-1]
        
        candidates = self.leaf2docs.get(leaf_node, [])
        candidates = [d for d in candidates if d != pos_docid]
        
        if candidates:
            if len(candidates) >= self.neg_num:
                neg_docids.extend(random.sample(candidates, self.neg_num))
            else:
                neg_docids.extend(candidates)

        # B. 随机负例 (Random Negatives)
        while len(neg_docids) < self.neg_num:
            rand_docid = random.choice(self._docid_list)
            if rand_docid != pos_docid and rand_docid not in neg_docids:
                neg_docids.append(rand_docid)
        
        neg_docids = neg_docids[:self.neg_num]
        neg_embs_np = np.stack([self.get_doc_embedding(nid) for nid in neg_docids])

        return {
            "qid": qid,
            "query": query_text,
            "pos_emb": pos_emb,
            "neg_emb": neg_embs_np,
            "pos_docid": pos_docid,
            "path_idx": path_idx
        }

def collate_fn(batch, tokenizer, max_len):
    queries = [b["query"] for b in batch]
    qids = [b["qid"] for b in batch]
    pos_docids = [b["pos_docid"] for b in batch]
    path_indices = [b["path_idx"] for b in batch]

    pos_tensor = torch.tensor(np.array([b["pos_emb"] for b in batch]), dtype=torch.float)
    neg_tensor = torch.tensor(np.array([b["neg_emb"] for b in batch]), dtype=torch.float)

    q_enc = tokenizer(queries, max_length=max_len, truncation=True, padding="longest", return_tensors="pt")

    return {
        "q_input_ids": q_enc["input_ids"],
        "q_attention_mask": q_enc["attention_mask"],
        "pos_emb": pos_tensor, 
        "neg_emb": neg_tensor, 
        "qids": torch.tensor([float(qid) for qid in qids]),
        "pos_docids": pos_docids,
        "path_indices": path_indices
    }

class GetTargetPaths:
    def __init__(self, docid2path_path, children_embeddings_path):
        with open(docid2path_path, "rb") as f:
            self.docid2path = pickle.load(f)
        with open(children_embeddings_path, "rb") as f:
            children_map = pickle.load(f)
        self.lookup = {}
        for parent_id, children in children_map.items():
            self.lookup[parent_id] = {}
            for idx, child in enumerate(sorted(children, key=lambda x: x['child_index'])):
                self.lookup[parent_id][child['child_id']] = idx
    
    def _get_path(self, docid, path_idx):
        paths = self.docid2path.get(docid)
        if not paths: return None
        if isinstance(paths[0], int): return paths
        if path_idx < len(paths):
            return paths[path_idx]
        return paths[0]

    # 获取父节点 ID (用于 Teacher Forcing)
    def __call__(self, docids, path_indices, height=None):
        target_paths = []
        for i, docid in enumerate(docids):
            path_idx = path_indices[i]
            path = self._get_path(docid, path_idx)
            
            if not path: 
                target_paths.append(0)
                continue
                
            if path[0] == 0:
                node_id = 0 if height == 0 else path[height]
            else:
                node_id = path[height]
            
            target_paths.append(node_id)
        return target_paths
    
    # 获取目标子节点索引 (用于计算 Loss)
    def get_index(self, docids, path_indices, height=None):
        target_indices = []
        for i, docid in enumerate(docids):
            path_idx = path_indices[i]
            path = self._get_path(docid, path_idx)

            if not path:
                target_indices.append(0)
                continue

            parent_id = path[height]
            target_id = path[height+1]

            if parent_id in self.lookup and target_id in self.lookup[parent_id]:
                real_index = self.lookup[parent_id][target_id]
            else:
                real_index = 0 
            target_indices.append(real_index)
        return torch.tensor(target_indices)

class MsMarcoDocVectorDataset(Dataset):
    def __init__(self, doc_embedding_path, docid_to_index_path, docid2path_path, embedding_dim=768, length=None):
        print(f"[DocVecAux] Loading indices from {docid_to_index_path}...")
        self.docid2index = {}
        with open(docid_to_index_path, "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f, desc="Loading Indices", disable=True):
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    self.docid2index[parts[0]] = int(parts[1])

        print(f"[DocVecAux] Loading doc paths from {docid2path_path}...")
        self.samples = [] 
        
        with open(docid2path_path, "rb") as f:
            docid2path_full = pickle.load(f)
            
            for docid, paths in docid2path_full.items():
                if docid in self.docid2index:
                    # 兼容性处理
                    if len(paths) > 0 and isinstance(paths[0], int):
                        paths = [paths]
                    
                    # 路径展开
                    for i in range(len(paths)):
                        self.samples.append({
                            "docid": docid,
                            "path_idx": i
                        })

        self.embedding_dim = embedding_dim
        self.embedding_path = doc_embedding_path
        file_size = os.path.getsize(doc_embedding_path)
        float32_size = 4
        self.num_docs = file_size // (self.embedding_dim * float32_size)
        
        self.doc_embeddings = np.memmap(
            self.embedding_path, dtype='float32', mode='r', 
            shape=(self.num_docs, self.embedding_dim)
        )
        print(f"[DocVecAux] Ready. Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        docid = sample['docid']
        path_idx = sample['path_idx']
        
        idx_in_memmap = self.docid2index[docid]
        vector = self.doc_embeddings[idx_in_memmap].copy() 
        
        return {
            "docid": docid, 
            "vector": vector,
            "path_idx": path_idx
        }

def doc_vector_collate_fn(batch):
    vectors = [b["vector"] for b in batch]
    docids = [b["docid"] for b in batch]
    path_indices = [b["path_idx"] for b in batch]
    
    return {
        "doc_emb": torch.tensor(np.array(vectors), dtype=torch.float),
        "docids": docids,
        "path_indices": path_indices
    }