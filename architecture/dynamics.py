"""
Dynamics models for the Neural Tropism World Model (NTWM).
Implements inverse dynamics and spatiotemporal dynamics.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .types import Tensor

class ActionVQ(nn.Module):
    """Vector quantized action codebook."""
    
    def __init__(self, A: int, dim: int):
        super().__init__()
        self.A = A
        self.codebook = nn.Embedding(A, dim)
        nn.init.normal_(self.codebook.weight, std=0.02)

    def quantize(self, e: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize action embedding to discrete action codes."""
        with torch.no_grad():
            d2 = (e.pow(2).sum(1, keepdim=True) 
                  - 2 * e @ self.codebook.weight.t()
                  + self.codebook.weight.pow(2).sum(1).unsqueeze(0))
            idx = torch.argmin(d2, dim=1)
        q = self.codebook(idx)
        return idx, q

class InverseDynamics(nn.Module):
    """Learns discrete action codes from (z_t, z_{t+1})."""
    
    def __init__(self, K: int, A: int, dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(K, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim*2, dim), 
            nn.GELU(), 
            nn.Linear(dim, dim)
        )
        self.vq = ActionVQ(A, dim)

    def forward(self, z_t: Tensor, z_tp1: Tensor) -> Tuple[Tensor, Tensor]:
        """Infer action from state transition."""
        B, H, W = z_t.shape
        e_t = self.embed(z_t.view(B, -1)).mean(dim=1)      # [B,D]
        e_tp1 = self.embed(z_tp1.view(B, -1)).mean(dim=1)  # [B,D]
        e = self.mlp(torch.cat([e_t, e_tp1], dim=-1))      # [B,D]
        ids, q = self.vq.quantize(e)                       # discrete actions and embeddings
        return ids, q

class STDynamics(nn.Module):
    """Autoregressive next-z model conditioned on action and contexts."""
    
    def __init__(self, K: int, H: int, W: int, d_model: int = 512, 
                 nhead: int = 8, nlayers: int = 8):
        super().__init__()
        self.K, self.H, self.W = K, H, W
        self.token_embed = nn.Embedding(K, d_model)
        self.pos = nn.Parameter(torch.randn(1, H*W, d_model) * 0.01)
        self.action_proj = nn.Linear(d_model, d_model)
        self.ctx_proj = nn.Linear(512+256+128+16, d_model)  # concat dims

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, K)

    def forward(self,
                z_hist: Tensor,          # [B,T,H,W]
                a_t_emb: Tensor,         # [B,d_model]
                ctx_cell: Tensor,        # [B,512]
                ctx_tissue: Tensor,      # [B,256]
                ctx_omics: Optional[Tensor],  # [B,128] or None
                prior_vec: Tensor        # [B,16]
                ) -> Tensor:
        """Predict next state given history, action, and context."""
        B, T, H, W = z_hist.shape
        
        # Build sequence of token embeddings
        seq = []
        for t in range(T):
            x = self.token_embed(z_hist[:, t].view(B, H*W)) + self.pos  # [B,H*W,d]
            seq.append(x)
        X = torch.cat(seq, dim=1)  # [B, T*H*W, d]

        # Prepend context + action tokens
        ctx = [
            ctx_cell, 
            ctx_tissue, 
            ctx_omics if ctx_omics is not None else torch.zeros(B,128,device=X.device), 
            prior_vec
        ]
        ctx_cat = torch.cat(ctx, dim=-1)                    # [B,512+256+128+16]
        ctx_token = self.ctx_proj(ctx_cat).unsqueeze(1)     # [B,1,d]
        a_token = self.action_proj(a_t_emb).unsqueeze(1)    # [B,1,d]

        # Encode with transformer
        Henc = self.encoder(torch.cat([ctx_token, a_token, X], dim=1))
        Hlast = Henc[:, -H*W:, :]                           # last frame positions
        logits = self.head(Hlast).view(B, H, W, self.K)     # [B,H,W,K]
        return logits
