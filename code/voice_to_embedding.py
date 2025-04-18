import torch
import torch.nn as nn
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Voice2Embedding(nn.Module):
    def __init__(self, voice_encoder: nn.Module, projection_dim: int = 768, nhead: int = 8, num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.base_model = voice_encoder
        self.input_dim = voice_encoder.linear.out_features

        # Normal transformer without potentially unstable components
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=nhead,
            dim_feedforward=4 * self.input_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Simple projection head to avoid instability
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),  # Back to ReLU for stability
            nn.Dropout(dropout),
            nn.Linear(1024, projection_dim)
        )

        # Gradient clipping in init to prevent explosion
        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        logger.info(f"Number of parameters in base model: {sum(p.numel() for p in self.base_model.parameters())}")
        logger.info(f"Number of parameters in new model: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x):
        with torch.no_grad():  # Don't backprop through base model initially
            speech_embedding = self.base_model(x)

        # Handle different input dimensions
        if speech_embedding.dim() == 2:
            speech_embedding = speech_embedding.unsqueeze(1)

        # Add small epsilon to prevent zeros
        speech_embedding = speech_embedding + 1e-8

        # Apply transformer with stability check
        encoded = self.transformer(speech_embedding)

        # Simple mean pooling is more stable
        pooled = encoded.mean(dim=1)

        # Check for NaNs and replace if needed
        if torch.isnan(pooled).any():
            pooled = torch.where(torch.isnan(pooled), torch.ones_like(pooled) * 1e-8, pooled)

        projected = self.projection(pooled)

        # Softly normalize to avoid division by zero
        norm = torch.norm(projected, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        normalized = projected / norm

        return normalized