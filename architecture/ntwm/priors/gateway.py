"""
Priors gateway and utilities for the Neural Tropism World Model (NTWM).
Maps frozen sequence embeddings to coarse, discrete priors.
"""

import torch
from typing import Optional
from ..core.types import Tensor, SequencePrior

def _bin_from_scalar(x: float) -> str:
    """Convert scalar value to discrete bin."""
    if x < 0.33: 
        return "low"
    if x < 0.66: 
        return "med"
    return "high"

class PriorsGateway:
    """Maps frozen sequence embeddings to coarse, discrete priors. Non-differentiable."""
    
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def from_sequence(self, 
                     z_nuc: Optional[Tensor], 
                     z_prot: Optional[Tensor], 
                     serotype: Optional[str]) -> SequencePrior:
        """Generate sequence priors from embeddings."""
        s = float((0.0 if z_nuc is None else z_nuc.abs().mean()) + 
                  (0.0 if z_prot is None else z_prot.abs().mean()))
        
        esc = _bin_from_scalar((s * 0.17) % 1.0)
        nuc = _bin_from_scalar((s * 0.37) % 1.0)
        stab = "ok" if s < 0.5 else "uncertain"
        bias = None if serotype is None else "med"
        
        return SequencePrior(esc, nuc, stab, bias)

# -------------------------------
# Utility: pack discrete priors
# -------------------------------

_PRIOR_MAP = {"low": 0.0, "med": 0.5, "high": 1.0}
_STAB_MAP = {"ok": 0.25, "uncertain": 0.75}
_BIAS_MAP = {"weak": 0.25, "med": 0.5, "strong": 0.75}

def pack_priors(pr_struct: 'StructuralPrior',
                pr_reg: 'RegulatoryPrior',
                pr_misc: 'MiscPrior',
                pr_seq: Optional[SequencePrior]) -> Tensor:
    """Pack discrete priors into a tensor vector."""
    v = [
        _PRIOR_MAP.get(pr_struct.endosomal_escape, 0.5),
        _PRIOR_MAP.get(pr_struct.nuclear_entry, 0.5),
        _STAB_MAP.get(pr_struct.stability, 0.5),
        float(pr_reg.promoter_active_prob_mean),
        float(pr_reg.promoter_active_prob_std),
    ]
    
    # Add misc bins (sorted for consistency)
    for k in sorted(pr_misc.bins.keys())[:5]:
        v.append(_PRIOR_MAP.get(pr_misc.bins[k], 0.5))
    
    # Pad to 10 elements
    while len(v) < 10:
        v.append(0.0)
    
    # Add sequence priors if available
    if pr_seq is not None:
        v.append(_PRIOR_MAP.get(pr_seq.endosomal_escape, 0.5))
        v.append(_PRIOR_MAP.get(pr_seq.nuclear_entry, 0.5))
        v.append(_STAB_MAP.get(pr_seq.stability, 0.5))
        v.append(_BIAS_MAP.get(pr_seq.tropism_bias, 0.0) if pr_seq.tropism_bias else 0.0)
    else:
        v += [0.0, 0.0, 0.0, 0.0]
    
    return torch.tensor(v).unsqueeze(0)  # [1,16]
