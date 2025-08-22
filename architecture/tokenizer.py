"""
Spatiotemporal tokenizer for the Neural Tropism World Model (NTWM).
Implements VQ-VAE style encoding/decoding with discrete tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .types import Tensor

class VQEncoder(nn.Module):
    """Convolutional encoder for VQ-VAE."""
    
    def __init__(self, in_ch: int, h: int, w: int, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.h, self.w, self.hidden = h, w, hidden

    def forward(self, x: Tensor) -> Tensor:
        """Encode input to hidden representation."""
        return self.conv(x)  # [B, hidden, H, W]

class VQCodebook(nn.Module):
    """Vector quantized codebook for discrete tokenization."""
    
    def __init__(self, K: int, dim: int):
        super().__init__()
        self.K = K
        self.embed = nn.Embedding(K, dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def quantize(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize hidden representation to discrete tokens."""
        # h: [B, hidden, H, W] -> [B,H,W,D]
        B, D, H, W = h.shape
        h = h.permute(0, 2, 3, 1).contiguous()  # [B,H,W,D]
        flat = h.view(B*H*W, D)
        
        with torch.no_grad():
            cb = self.embed.weight  # [K,D]
            d2 = (flat.pow(2).sum(1, keepdim=True)
                  - 2 * flat @ cb.t()
                  + cb.pow(2).sum(1).unsqueeze(0))
            idx = torch.argmin(d2, dim=1)  # [B*H*W]
        
        z = idx.view(B, H, W)
        q = self.embed(idx).view(B, H, W, D)
        return z, q

class VQDecoder(nn.Module):
    """Convolutional decoder for VQ-VAE."""
    
    def __init__(self, out_ch: int, hidden: int = 128):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_ch, 1),
        )

    def forward(self, q: Tensor) -> Tensor:
        """Decode quantized representation to output."""
        # q: [B,H,W,D] -> [B,C,H,W]
        q = q.permute(0, 3, 1, 2).contiguous()
        return torch.sigmoid(self.deconv(q))

class SpatiotemporalTokenizer(nn.Module):
    """Encoder + VQ + (train-time) decoder. Produces tokens z_t."""
    
    def __init__(self, channels: int, H: int, W: int, hidden: int = 128, K: int = 8192):
        super().__init__()
        self.encoder = VQEncoder(channels, H, W, hidden)
        self.codebook = VQCodebook(K, hidden)
        self.decoder = VQDecoder(channels, hidden)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input to discrete tokens and quantized vectors."""
        h = self.encoder(x)          # [B, D, H, W]
        z, q = self.codebook.quantize(h)  # [B,H,W], [B,H,W,D]
        return z, q

    def decode(self, q: Tensor) -> Tensor:
        """Decode quantized vectors to reconstruction."""
        return self.decoder(q)

    def recon_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """Reconstruction loss for training."""
        return F.l1_loss(x_hat, x)
