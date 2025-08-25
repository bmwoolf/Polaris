"""
Utility functions and demo helpers for the Neural Tropism World Model (NTWM).
"""

import torch
from .types import CellContext, TissueContext, OmicsContext, StructuralPrior, RegulatoryPrior, MiscPrior

def dummy_contexts(B: int = 1, H: int = 32, W: int = 32):
    """Generate dummy context data for testing and demos."""
    cell = CellContext(embedding=torch.randn(B, 512))
    tis = TissueContext(embedding=torch.randn(B, 256), region_masks=None)
    omx = OmicsContext(embedding=torch.randn(B, 128))
    pr_s = StructuralPrior("med", "low", "uncertain")
    pr_r = RegulatoryPrior(0.62, 0.08)
    pr_m = MiscPrior(bins={})
    return cell, tis, omx, pr_s, pr_r, pr_m
