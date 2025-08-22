"""
Types and dataclasses for the Neural Tropism World Model (NTWM).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch

Tensor = torch.Tensor

# -------------------------------
# Context & Priors
# -------------------------------

@dataclass(frozen=True)
class CellContext:
    """Cell context embedding."""
    embedding: Tensor   # [d_cell]

@dataclass(frozen=True)
class TissueContext:
    """Tissue context with optional region masks."""
    embedding: Tensor   # [d_tissue]
    region_masks: Optional[Tensor] = None  # [H,W,#regions] optional

@dataclass(frozen=True)
class OmicsContext:
    """Omics context embedding."""
    embedding: Tensor   # [d_omics]

@dataclass(frozen=True)
class StructuralPrior:
    """Structural properties prior."""
    endosomal_escape: str  # "low"|"med"|"high"
    nuclear_entry: str     # "low"|"med"|"high"
    stability: str         # "ok"|"uncertain"

@dataclass(frozen=True)
class RegulatoryPrior:
    """Regulatory properties prior."""
    promoter_active_prob_mean: float
    promoter_active_prob_std: float

@dataclass(frozen=True)
class MiscPrior:
    """Miscellaneous properties prior."""
    bins: Dict[str, str]  # key->"low"|"med"|"high"

@dataclass(frozen=True)
class SequencePrior:
    """Sequence-specific properties prior."""
    endosomal_escape: str
    nuclear_entry: str
    stability: str
    tropism_bias: Optional[str] = None  # "weak"|"med"|"strong"|None

# -------------------------------
# Decoder Output Types
# -------------------------------

@dataclass(frozen=True)
class Overlay:
    """Export-safe overlay with rationales."""
    endosome_prob: Tensor   # [B,1,H,W]
    nuclear_barrier: Tensor # [B,1,H,W]
    aleatoric: Tensor       # [B,1]
    epistemic: Tensor       # [B,1]
    rationale: List[str]
