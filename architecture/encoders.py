"""
Frozen sequence encoders for the Neural Tropism World Model (NTWM).
These are publicly-trained models that return embeddings without generation capabilities.
"""

import torch
import torch.nn as nn
from .safety import SealedSequenceHandle
from .types import Tensor

class FrozenNucEncoder(nn.Module):
    """Publicly-trained nucleotide LM (frozen). Returns embeddings. No generation."""
    
    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim, bias=False)
        
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, sealed: SealedSequenceHandle) -> Tensor:
        """Encode sealed sequence into embedding."""
        h = torch.zeros(self.dim)
        h[: min(8, self.dim)] = torch.tensor([
            float(bool(sealed.meta["has_bytes"])),
            float(sealed.meta["len"] > 0),
            float(sealed.meta["len"] % 7) / 7.0,
            0.0, 0.0, 0.0, 0.0, 0.0
        ])
        return h

class FrozenProtEncoder(nn.Module):
    """Publicly-trained protein LM (frozen). Returns embeddings. No generation."""
    
    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim, bias=False)
        
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, sealed: SealedSequenceHandle) -> Tensor:
        """Encode sealed sequence into embedding."""
        h = torch.zeros(self.dim)
        if sealed.meta.get("serotype_class"):
            h[0] = 0.5
        return h
