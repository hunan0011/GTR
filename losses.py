import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Cross-entropy loss (unchanged)

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
        
        # Do not hard-clip confidence; let (1 - pt)^gamma handle it automatically
        # Even when pt=0.9, (1 - 0.9)^2 = 0.01, so a small gradient is still retained for fine-grained adjustment
        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * target_log_probs

        return loss.mean() if self.reduction == 'mean' else loss.sum()
    

# 2. Triplet contrastive loss using dot-product similarity
class TripletContrastiveLoss(nn.Module):
    """
    Triplet loss based on dot-product similarity.
    The positive dot-product score should exceed the negative score by a margin.
    Formula: L = max(0, score_neg - score_pos + margin)
    """
    def __init__(self,margin=0.15):
        super().__init__()
        self.margin = margin

    def forward(self, query_emb, pos_emb, neg_emb):
        # print(pos_emb, neg_emb)
        """
        Arguments:
            query_emb: [Batch, Dim]
            pos_emb:   [Batch, Dim]
            neg_emb:   [Batch, Dim]
        """
        # 1. Compute dot-product scores
        # Multiply element-wise and sum along the vector dimension (dim=1)
        # shape: [Batch]
        score_pos = torch.sum(query_emb * pos_emb, dim=1)
        score_neg = torch.sum(query_emb * neg_emb, dim=1)
        # print(score_pos, score_neg)
        # 2. Compute the loss
        # Desired condition: score_pos > score_neg + margin
        # Violation degree: score_neg - score_pos + margin
        # print(score_neg, score_pos)
        loss = torch.clamp(score_neg - score_pos + self.margin, min=0.0)
        
        # 3. Return the mean loss
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
        # 1. Positive scores (Batch, 1)
        # (B, D) * (B, D) -> (B, 1)
        scores_pos = torch.sum(query_emb * pos_emb, dim=-1, keepdim=True) * logit_scale
        
        # 2. Negative scores (Batch, K)
        # query_emb needs to be expanded to (B, 1, D)
        # neg_emb: (B, K, D)
        # result: (B, K)
        scores_neg = torch.sum(query_emb.unsqueeze(1) * neg_emb, dim=-1) * logit_scale
        
        # 3. Concatenate logits (Batch, 1 + K)
        # The positive sample is placed in column 0
        logits = torch.cat([scores_pos, scores_neg], dim=1)
        
        # 4. The target is always 0
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return self.cross_entropy(logits, targets)


# 3. Knowledge distillation loss (unchanged)
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        log_p_s = F.log_softmax(student_logits / self.temperature, dim=1)
        p_t = F.softmax(teacher_logits / self.temperature, dim=1)
        return self.kl_div(log_p_s, p_t) * (self.temperature ** 2)