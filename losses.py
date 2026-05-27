import torch
import torch.nn as nn
import torch.nn.functional as F

# Focal loss

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
        
        # Keep confidence unclipped so the focal weight controls the gradient.
        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * target_log_probs

        return loss.mean() if self.reduction == 'mean' else loss.sum()
    

# Triplet contrastive loss using dot-product similarity
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
        # Compute dot-product scores.
        score_pos = torch.sum(query_emb * pos_emb, dim=1)
        score_neg = torch.sum(query_emb * neg_emb, dim=1)
        # print(score_pos, score_neg)
        # Penalize negatives that are too close to the positive score.
        # print(score_neg, score_pos)
        loss = torch.clamp(score_neg - score_pos + self.margin, min=0.0)
        
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
        neg_emb:   [Batch, K, Dim]
        logit_scale: scalar
        """
        # Positive scores: (B, D) * (B, D) -> (B, 1).
        scores_pos = torch.sum(query_emb * pos_emb, dim=-1, keepdim=True) * logit_scale
        
        # Negative scores: expand query_emb to (B, 1, D), then score (B, K, D).
        scores_neg = torch.sum(query_emb.unsqueeze(1) * neg_emb, dim=-1) * logit_scale
        
        # Put the positive sample in column 0.
        logits = torch.cat([scores_pos, scores_neg], dim=1)
        
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return self.cross_entropy(logits, targets)


# Knowledge distillation loss
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        log_p_s = F.log_softmax(student_logits / self.temperature, dim=1)
        p_t = F.softmax(teacher_logits / self.temperature, dim=1)
        return self.kl_div(log_p_s, p_t) * (self.temperature ** 2)
