"""
Tests for priors components of the NTWM.
"""

import pytest
import torch
from ntwm.priors import PriorsGateway, pack_priors
from ntwm.core import (
    StructuralPrior,
    RegulatoryPrior,
    MiscPrior,
    SequencePrior
)

def test_priors_gateway():
    """Test PriorsGateway functionality."""
    gateway = PriorsGateway()
    
    # Test with nucleotide embedding only
    z_nuc = torch.randn(1, 512)
    prior = gateway.from_sequence(z_nuc, None, None)
    
    assert isinstance(prior, SequencePrior)
    assert prior.endosomal_escape in ["low", "med", "high"]
    assert prior.nuclear_entry in ["low", "med", "high"]
    assert prior.stability in ["ok", "uncertain"]
    assert prior.tropism_bias is None
    
    # Test with protein embedding and serotype
    z_prot = torch.randn(1, 512)
    prior = gateway.from_sequence(z_nuc, z_prot, "serotype1")
    
    assert prior.tropism_bias == "med"

def test_pack_priors():
    """Test pack_priors functionality."""
    # Create test priors
    pr_struct = StructuralPrior("high", "med", "ok")
    pr_reg = RegulatoryPrior(0.75, 0.15)
    pr_misc = MiscPrior(bins={"key1": "low", "key2": "high"})
    pr_seq = SequencePrior("med", "high", "uncertain", "strong")
    
    # Test packing with sequence priors
    packed = pack_priors(pr_struct, pr_reg, pr_misc, pr_seq)
    
    assert packed.shape == (1, 16)
    assert packed.dtype == torch.float32
    
    # Test values
    assert packed[0, 0] == 1.0  # high endosomal_escape
    assert packed[0, 1] == 0.5  # med nuclear_entry
    assert packed[0, 2] == 0.25  # ok stability
    assert packed[0, 3] == 0.75  # promoter_active_prob_mean
    assert packed[0, 4] == 0.15  # promoter_active_prob_std
    
    # Test packing without sequence priors
    packed_no_seq = pack_priors(pr_struct, pr_reg, pr_misc, None)
    assert packed_no_seq.shape == (1, 16)
    
    # Last 4 values should be 0.0 when no sequence priors
    assert torch.all(packed_no_seq[0, -4:] == 0.0)

def test_pack_priors_edge_cases():
    """Test pack_priors with edge cases."""
    # Test with empty misc bins
    pr_struct = StructuralPrior("low", "low", "uncertain")
    pr_reg = RegulatoryPrior(0.0, 0.0)
    pr_misc = MiscPrior(bins={})
    
    packed = pack_priors(pr_struct, pr_reg, pr_misc, None)
    assert packed.shape == (1, 16)
    
    # Test with many misc bins (should cap at 5)
    pr_misc_many = MiscPrior(bins={
        f"key{i}": "high" for i in range(10)
    })
    
    packed_many = pack_priors(pr_struct, pr_reg, pr_misc_many, None)
    assert packed_many.shape == (1, 16)
    
    # Test with unknown values (should use defaults)
    pr_struct_unknown = StructuralPrior("unknown", "invalid", "bad")
    packed_unknown = pack_priors(pr_struct_unknown, pr_reg, pr_misc, None)
    
    # Should use default values (0.5) for unknown
    assert packed_unknown[0, 0] == 0.5  # default for unknown endosomal_escape
    assert packed_unknown[0, 1] == 0.5  # default for unknown nuclear_entry
    assert packed_unknown[0, 2] == 0.5  # default for unknown stability
