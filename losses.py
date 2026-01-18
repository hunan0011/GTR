import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 交叉熵损失 (保持不变)

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(target_log_probs)
        
        # 不要硬截断 confidence，让 (1-pt)^gamma 自动处理
        # 即使 pt=0.9, (1-0.9)^2 = 0.01，仍然保留微弱梯度进行精细化微调
        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * target_log_probs

        return loss.mean() if self.reduction == 'mean' else loss.sum()
    

# 2. 三元组对比损失 (修改为：点积相似度)
class TripletContrastiveLoss(nn.Module):
    """
    基于点积 (Dot Product) 的 Triplet Loss。
    目标：Positive 的点积得分，要比 Negative 的点积得分，高出一个 margin。
    公式：L = max(0, score_neg - score_pos + margin)
    """
    def __init__(self,margin=0.15):
        super().__init__()
        self.margin = margin

    def forward(self, query_emb, pos_emb, neg_emb):
        # print(pos_emb, neg_emb)
        """
        参数:
            query_emb: [Batch, Dim]
            pos_emb:   [Batch, Dim]
            neg_emb:   [Batch, Dim]
        """
        # 1. 计算点积 (Dot Product)
        # 逐元素相乘，然后在向量维度 (dim=1) 求和
        # shape: [Batch]
        score_pos = torch.sum(query_emb * pos_emb, dim=1)
        score_neg = torch.sum(query_emb * neg_emb, dim=1)
        # print(score_pos, score_neg)
        # 2. 计算损失
        # 我们希望: score_pos > score_neg + margin
        # 违背程度: score_neg - score_pos + margin
        # print(score_neg, score_pos)
        loss = torch.clamp(score_neg - score_pos + self.margin, min=0.0)
        
        # 3. 返回平均值
        return loss.mean()

# losses.py

class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, query_emb, pos_emb, neg_emb, logit_scale=1):
        """
        query_emb: [Batch, Dim]
        pos_emb:   [Batch, Dim]
        neg_emb:   [Batch, K, Dim]  <--- 支持多负例
        logit_scale: scalar
        """
        # 1. 正例分数 (Batch, 1)
        # (B, D) * (B, D) -> (B, 1)
        scores_pos = torch.sum(query_emb * pos_emb, dim=-1, keepdim=True) * logit_scale
        
        # 2. 负例分数 (Batch, K)
        # query_emb 需要扩展维度: (B, 1, D)
        # neg_emb: (B, K, D)
        # result: (B, K)
        scores_neg = torch.sum(query_emb.unsqueeze(1) * neg_emb, dim=-1) * logit_scale
        
        # 3. 拼接 Logits (Batch, 1 + K)
        # 正例在第 0 列
        logits = torch.cat([scores_pos, scores_neg], dim=1)
        
        # 4. Target 永远是 0
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return self.cross_entropy(logits, targets)


# 3. 知识蒸馏损失 (保持不变)
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        log_p_s = F.log_softmax(student_logits / self.temperature, dim=1)
        p_t = F.softmax(teacher_logits / self.temperature, dim=1)
        return self.kl_div(log_p_s, p_t) * (self.temperature ** 2)