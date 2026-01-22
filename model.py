import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
import config
import numpy as np

# =================================================================
# [修改] Similarity：Pro 优化版 (Pre-LN + 升维 + 残差)，去掉了零初始化
# =================================================================
class Similarity(nn.Module):
    def __init__(self, input_dim, dropout=0.1, expansion_factor=2):
        super().__init__()
        hidden_dim = int(input_dim * expansion_factor)

        self.q_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),            # 1. Pre-Norm: 放在最前面
            nn.Linear(input_dim, hidden_dim),   # 2. 升维 (Expansion)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)    # 3. 降维 (Projection)
        )
        
        self.c_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        

    def forward(self, query, candidates):
        """
        query: [Batch, Dim]
        candidates: [Batch, Num_Candidates, Dim]
        """
        # --- 1. 残差连接 (Residual Connection) ---
        q_delta = self.q_mlp(query)
        q_refined = query + q_delta  
        
        c_delta = self.c_mlp(candidates)
        c_refined = candidates + c_delta 
        
        # --- 2. 归一化 (F.normalize) ---
        q_refined = F.normalize(q_refined, p=2, dim=-1)
        c_refined = F.normalize(c_refined, p=2, dim=-1)
        
        # --- 3. 点积相似度 ---
        scores = torch.sum(q_refined.unsqueeze(1) * c_refined, dim=-1)
        
        return scores

class Encoder(nn.Module):
    def __init__(self, model_name: str, pooling: str = "mean", device: torch.device = None):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name, output_hidden_states=False)
        self.backbone = AutoModel.from_pretrained(model_name, config=cfg)
        self.pooling = pooling
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(1)
        denom = mask.sum(1).clamp(min=1e-9)
        return summed / denom

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        if self.pooling == "mean":
            embedding = self._mean_pooling(last_hidden_state, attention_mask)
        else:
            embedding = last_hidden_state[:, 0, :]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

class Indexer(nn.Module):
    def __init__(self, H, B, device: torch.device = None):
        super().__init__()
        self.H, self.B = H, B
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim = config.EMBEDDING_DIM
        
        with open(config.CHILDREN_EMBEDDINGS_PATH, "rb") as f:
            children_map = pickle.load(f)
        
        max_id = 0
        for pid, children in children_map.items():
            max_id = max(max_id, pid)
            for c in children:
                max_id = max(max_id, c['child_id'])
        num_nodes = max_id + 1
        
        adjacency = torch.zeros((num_nodes, self.B), dtype=torch.long)
        init_weights = torch.zeros((num_nodes, dim))
        for pid, children in children_map.items():
            for c in children:
                init_weights[c['child_id']] = torch.tensor(c['embedding'])
                adjacency[pid, c['child_index']] = c['child_id']

        self.fixed_centroids = nn.Embedding(num_nodes, dim)
        self.fixed_centroids.weight.data.copy_(F.normalize(init_weights, p=2, dim=1))
        self.fixed_centroids.weight.requires_grad = False 
        
        self.scorers = nn.ModuleList([
            Similarity(input_dim=dim, expansion_factor=2) # 使用 2 倍升维
            for _ in range(self.H) 
        ])
        
        self.logit_scale = nn.Parameter(torch.tensor(np.log(20.0))) 
        self.register_buffer("adjacency", adjacency)
        self.to(self.device)

    def forward(self, qids, query_embeddings, pos_docids, path_indices, path_getter):
        batch_size = query_embeddings.size(0)
        all_layer_logits, all_layer_targets = [], []
        
        l_scale = self.logit_scale.exp().clamp(max=100.0)
        
        for h in range(1, self.H):
            target_indices = path_getter.get_index(pos_docids, path_indices, height=h-1).to(self.device)
            
            if h == 1:
                p_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            else:
                p_list = path_getter(pos_docids, path_indices, height=h-1)
                p_ids = torch.tensor(p_list, dtype=torch.long, device=self.device)
            
            c_ids = F.embedding(p_ids, self.adjacency) 
            child_anchors = self.fixed_centroids(c_ids) 
            
            current_scorer = self.scorers[h-1]
            
            # 直接传入 query [B, Dim]
            raw_logits = current_scorer(query_embeddings, child_anchors)
            logits = raw_logits * l_scale
            
            all_layer_logits.append(logits)
            all_layer_targets.append(target_indices)
            
        return all_layer_logits, all_layer_targets