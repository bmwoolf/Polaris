"""
Tests for core components of the NTWM.
"""

import pytest
import torch
from ntwm.core import (
    CellContext,
    TissueContext,
    OmicsContext,
    StructuralPrior,
    RegulatoryPrior,
    MiscPrior,
    SequencePrior,
    SafetyGate,
    SealedSequenceHandle,
    dummy_contexts
)

def test_cell_context():
    """Test CellContext creation and properties."""
    embedding = torch.randn(1, 512)
    ctx = CellContext(embedding=embedding)
    assert ctx.embedding.shape == (1, 512)
    assert ctx.embedding is embedding

def test_tissue_context():
    """Test TissueContext creation and properties."""
    embedding = torch.randn(1, 256)
    ctx = TissueContext(embedding=embedding)
    assert ctx.embedding.shape == (1, 256)
    assert ctx.region_masks is None

def test_omics_context():
    """Test OmicsContext creation and properties."""
    embedding = torch.randn(1, 128)
    ctx = OmicsContext(embedding=embedding)
    assert ctx.embedding.shape == (1, 128)

def test_structural_prior():
    """Test StructuralPrior creation and properties."""
    prior = StructuralPrior("med", "low", "uncertain")
    assert prior.endosomal_escape == "med"
    assert prior.nuclear_entry == "low"
    assert prior.stability == "uncertain"

def test_regulatory_prior():
    """Test RegulatoryPrior creation and properties."""
    prior = RegulatoryPrior(0.62, 0.08)
    assert prior.promoter_active_prob_mean == 0.62
    assert prior.promoter_active_prob_std == 0.08

def test_misc_prior():
    """Test MiscPrior creation and properties."""
    bins = {"key1": "high", "key2": "low"}
    prior = MiscPrior(bins=bins)
    assert prior.bins == bins

def test_sequence_prior():
    """Test SequencePrior creation and properties."""
    prior = SequencePrior("high", "med", "ok", "strong")
    assert prior.endosomal_escape == "high"
    assert prior.nuclear_entry == "med"
    assert prior.stability == "ok"
    assert prior.tropism_bias == "strong"

def test_safety_gate():
    """Test SafetyGate functionality."""
    gate = SafetyGate(max_len=1000)
    
    # Test valid sequence
    handle = gate.seal(b"ATCG", "serotype1")
    assert handle.meta["has_bytes"] is True
    assert handle.meta["len"] == 4
    assert handle.meta["serotype_class"] == "serotype1"
    
    # Test empty sequence
    with pytest.raises(Exception):
        gate.seal(None, None)
    
    # Test sequence too long
    long_seq = b"A" * 1001
    with pytest.raises(Exception):
        gate.seal(long_seq, "serotype1")

def test_dummy_contexts():
    """Test dummy context generation."""
    cell, tis, omx, pr_s, pr_r, pr_m = dummy_contexts(B=2, H=64, W=64)
    
    assert cell.embedding.shape == (2, 512)
    assert tis.embedding.shape == (2, 256)
    assert omx.embedding.shape == (2, 128)
    assert pr_s.endosomal_escape == "med"
    assert pr_r.promoter_active_prob_mean == 0.62
