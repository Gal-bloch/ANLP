import torch
import torch.nn.functional as F


def cosine_similarity_loss(speech_emb, text_emb):
    """Loss function that maximizes cosine similarity"""
    return 1 - F.cosine_similarity(speech_emb, text_emb, dim=1).mean()

# Min-max normalization for loss computation
def min_max_normalize(tensor):
    """Apply min-max normalization to tensor values to range [0, 1]"""
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    if min_val == max_val:
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val)

def contrastive_loss(speech_emb, text_emb, neg_text_emb, margin=0.5):
    """
    Contrastive loss using min-max normalization:
    - Maximize similarity between speech and positive text
    - Minimize similarity between speech and negative text
    """
    # Positive similarity (speech and matching text)
    pos_sim = F.cosine_similarity(speech_emb, text_emb, dim=1)

    # Negative similarity (speech and negated text)
    neg_sim = F.cosine_similarity(speech_emb, neg_text_emb, dim=1)

    # Normalize similarities
    pos_sim_norm = min_max_normalize(pos_sim)
    neg_sim_norm = min_max_normalize(neg_sim)

    # Loss: minimize positive distance, maximize negative distance
    loss = (1 - pos_sim_norm) + torch.clamp(neg_sim_norm - margin, min=0)

    return loss.mean()

def info_nce_loss(speech_emb, text_emb, temperature=0.07):
    """
    InfoNCE loss with in-batch negatives (cross-modal)
    speech_emb: (B, D)
    text_emb: (B, D)
    """
    # Normalize
    speech_emb = F.normalize(speech_emb, p=2, dim=1)
    text_emb = F.normalize(text_emb, p=2, dim=1)

    logits = torch.matmul(speech_emb, text_emb.T) / temperature  # (B, B)
    labels = torch.arange(len(speech_emb)).to(speech_emb.device)  # Ground truth is the diagonal

    loss_i2t = F.cross_entropy(logits, labels)       # Speech → Text
    loss_t2i = F.cross_entropy(logits.T, labels)     # Text → Speech

    return (loss_i2t + loss_t2i) / 2
