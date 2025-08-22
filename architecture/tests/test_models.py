"""
Tests for neural network models of the NTWM.
"""

import pytest
import torch
from ntwm.models import (
    FrozenNucEncoder,
    FrozenProtEncoder,
    SpatiotemporalTokenizer,
    VQEncoder,
    VQCodebook,
    VQDecoder,
    STDynamics,
    InverseDynamics,
    ActionVQ,
    DecoderExplainer
)
from ntwm.core import SealedSequenceHandle

def test_frozen_nuc_encoder():
    """Test FrozenNucEncoder functionality."""
    encoder = FrozenNucEncoder(dim=256)
    
    # Test parameters are frozen
    for param in encoder.parameters():
        assert not param.requires_grad
    
    # Test forward pass
    handle = SealedSequenceHandle({"has_bytes": True, "len": 100})
    output = encoder(handle)
    assert output.shape == (256,)
    assert output.dtype == torch.float32

def test_frozen_prot_encoder():
    """Test FrozenProtEncoder functionality."""
    encoder = FrozenProtEncoder(dim=128)
    
    # Test parameters are frozen
    for param in encoder.parameters():
        assert not param.requires_grad
    
    # Test forward pass
    handle = SealedSequenceHandle({"serotype_class": "test"})
    output = encoder(handle)
    assert output.shape == (128,)
    assert output.dtype == torch.float32

def test_vq_encoder():
    """Test VQEncoder functionality."""
    encoder = VQEncoder(in_ch=3, h=32, w=32, hidden=64)
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    output = encoder(x)
    assert output.shape == (2, 64, 32, 32)

def test_vq_codebook():
    """Test VQCodebook functionality."""
    codebook = VQCodebook(K=1024, dim=128)
    
    # Test quantization
    h = torch.randn(2, 128, 16, 16)
    z, q = codebook.quantize(h)
    
    assert z.shape == (2, 16, 16)
    assert z.dtype == torch.long
    assert q.shape == (2, 16, 16, 128)
    assert q.dtype == torch.float32

def test_vq_decoder():
    """Test VQDecoder functionality."""
    decoder = VQDecoder(out_ch=3, hidden=64)
    
    # Test forward pass
    q = torch.randn(2, 16, 16, 64)
    output = decoder(q)
    assert output.shape == (2, 3, 16, 16)
    assert torch.all((output >= 0) & (output <= 1))  # sigmoid output

def test_spatiotemporal_tokenizer():
    """Test SpatiotemporalTokenizer functionality."""
    tokenizer = SpatiotemporalTokenizer(channels=3, H=16, W=16, K=512)
    
    # Test encode
    x = torch.randn(2, 3, 16, 16)
    z, q = tokenizer.encode(x)
    
    assert z.shape == (2, 16, 16)
    assert q.shape == (2, 16, 16, 128)  # hidden dim
    
    # Test decode
    x_hat = tokenizer.decode(q)
    assert x_hat.shape == (2, 3, 16, 16)
    
    # Test reconstruction loss
    loss = tokenizer.recon_loss(x, x_hat)
    assert loss.shape == ()
    assert loss.dtype == torch.float32

def test_action_vq():
    """Test ActionVQ functionality."""
    action_vq = ActionVQ(A=16, dim=64)
    
    # Test quantization
    e = torch.randn(2, 64)
    idx, q = action_vq.quantize(e)
    
    assert idx.shape == (2,)
    assert idx.dtype == torch.long
    assert q.shape == (2, 64)
    assert q.dtype == torch.float32

def test_inverse_dynamics():
    """Test InverseDynamics functionality."""
    model = InverseDynamics(K=512, A=16, dim=128)
    
    # Test forward pass
    z_t = torch.randint(0, 512, (2, 16, 16))
    z_tp1 = torch.randint(0, 512, (2, 16, 16))
    
    a_ids, a_emb = model(z_t, z_tp1)
    
    assert a_ids.shape == (2,)
    assert a_emb.shape == (2, 128)

def test_st_dynamics():
    """Test STDynamics functionality."""
    model = STDynamics(K=512, H=16, W=16, d_model=256, nlayers=2)
    
    # Test forward pass
    z_hist = torch.randint(0, 512, (2, 3, 16, 16))
    a_t_emb = torch.randn(2, 256)
    ctx_cell = torch.randn(2, 512)
    ctx_tissue = torch.randn(2, 256)
    prior_vec = torch.randn(2, 16)
    
    logits = model(
        z_hist=z_hist,
        a_t_emb=a_t_emb,
        ctx_cell=ctx_cell,
        ctx_tissue=ctx_tissue,
        ctx_omics=None,
        prior_vec=prior_vec
    )
    
    assert logits.shape == (2, 16, 16, 512)

def test_decoder_explainer():
    """Test DecoderExplainer functionality."""
    codebook = VQCodebook(K=512, dim=128)
    decoder = DecoderExplainer(codebook, H=16, W=16)
    
    # Test forward pass
    z = torch.randint(0, 512, (2, 16, 16))
    overlay = decoder(z)
    
    assert overlay.endosome_prob.shape == (2, 1, 16, 16)
    assert overlay.nuclear_barrier.shape == (2, 1, 16, 16)
    assert overlay.aleatoric.shape == (2, 1)
    assert overlay.epistemic.shape == (2, 1)
    assert isinstance(overlay.rationale, list)
