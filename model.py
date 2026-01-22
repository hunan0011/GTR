import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
import config
import numpy as np

# =================================================================
# [修改] Similarity：使用你提供的 4倍维度拼接 + MLP 交互结构
# =================================================================
class Similarity(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.1):
        super().__init__()

        # 输入维度是 4 * input_dim (因为拼接了 q, c, q*c, |q-c|)
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim * 4),
            nn.Linear(input_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, query, candidates):
        """
        query:      [B, D]
        candidates: [B, K, D]
        return:     [B, K]
        """
        B, K, D = candidates.shape

        # 1️⃣ query 扩展到和 candidates 对齐
        q = query.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]

        # 2️⃣ 构造 query–candidate 成对特征
        # 这一步是核心：显式构造了交互特征
        pair_feat = torch.cat(
            [
                q,                      # query 原始语义
                candidates,             # child 原始语义
                q * candidates,         # 逐维相似性 (Hadamard product)
                torch.abs(q - candidates)  # 逐维差异 (Absolute difference)
            ],
            dim=-1
        )  # 结果形状 [B, K, 4D]

        # 3️⃣ MLP 直接输出分数
        # 输出 [B, K, 1] -> squeeze -> [B, K]
        scores = self.mlp(pair_feat).squeeze(-1)

        return scores

# =================================================================
# Encoder 和 Indexer 保持不变，Indexer 会自动兼容新的 Similarity
# =================================================================

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
        
        # 实例化 Similarity
        # 这里会自动调用新的 __init__，input_dim 传进去，hidden_dim 默认 512
        self.scorers = nn.ModuleList([
            Similarity(input_dim=dim)
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
            
            # 这里调用新的 forward，不再有点积操作，直接由 MLP 输出 scores
            # raw_logits: [Batch, B]
            raw_logits = current_scorer(query_embeddings, child_anchors)
            
            # 注意：因为 MLP 输出的已经是一个标量分数，这里乘以 scale 依然适用
            # 这相当于调节 Softmax 的温度
            logits = raw_logits * l_scale
            
            all_layer_logits.append(logits)
            all_layer_targets.append(target_indices)
            
        return all_layer_logits, all_layer_targets