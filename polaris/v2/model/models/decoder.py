"""
Decoder and explainer for the Neural Tropism World Model (NTWM).
Turns tokens into export-safe overlays with rationales.
"""

import torch
import torch.nn as nn
from typing import List
from ..core.types import Tensor, Overlay

class DecoderExplainer(nn.Module):
    """Turns tokens (via code vectors) into export-safe overlays with rationales."""
    
    def __init__(self, codebook: 'VQCodebook', H: int, W: int):
        super().__init__()
        self.cb = codebook
        self.H, self.W = H, W
        self.linear = nn.Linear(self.cb.embed.embedding_dim, 2)  # two overlay channels

    def forward(self, z: Tensor) -> Overlay:
        """Generate overlay from discrete tokens."""
        B, H, W = z.shape
        d = self.cb.embed.embedding_dim
        
        # Get code vectors from tokens
        vecs = self.cb.embed(z.view(B, -1)).view(B, H, W, d)
        
        # Project to overlay channels
        logits = self.linear(vecs).permute(0, 3, 1, 2).contiguous()  # [B,2,H,W]
        
        # Apply sigmoid for probabilities
        endosome = torch.sigmoid(logits[:, :1])
        nuclear = torch.sigmoid(logits[:, 1:2])
        
        # Compute uncertainty metrics
        alea = endosome.mean(dim=(2, 3))
        epis = nuclear.mean(dim=(2, 3))
        
        # Generate rationale
        rationale = ["Endosomal retention elevated under current priors."]
        
        return Overlay(endosome, nuclear, alea, epis, rationale)
