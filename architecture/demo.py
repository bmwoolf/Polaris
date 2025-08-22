"""
Example: minimal train/serve skeleton using the modular NTWM architecture.
"""

import torch
from . import (
    SpatiotemporalTokenizer, 
    InverseDynamics, 
    STDynamics, 
    DecoderExplainer,
    dummy_contexts, 
    pack_priors
)

def main():
    """Run a minimal training and serving demo."""
    B, C, H, W, K, A = 2, 8, 32, 32, 1024, 16
    cell, tis, omx, pr_s, pr_r, pr_m = dummy_contexts(B, H, W)

    # Initialize models
    tok = SpatiotemporalTokenizer(channels=C, H=H, W=W, hidden=128, K=K)
    inv = InverseDynamics(K=K, A=A, dim=256)
    dyn = STDynamics(K=K, H=H, W=W, d_model=512, nhead=8, nlayers=4)
    dec = DecoderExplainer(tok.codebook, H=H, W=W)

    # ---- Training-time (toy) ----
    x = torch.rand(B, C, H, W)             # frame t
    y = torch.rand(B, C, H, W)             # frame t+1
    z_t, q_t = tok.encode(x)
    z_tp1, q_tp1 = tok.encode(y)

    # Inverse dynamics learns latent action
    a_ids, a_emb = inv(z_t, z_tp1)

    # Dynamics learns next-z
    pr_vec = pack_priors(pr_s, pr_r, pr_m, pr_seq=None).expand(B, -1)
    logits = dyn(z_hist=torch.stack([z_t, z_tp1], dim=1),
                 a_t_emb=a_emb,
                 ctx_cell=cell.embedding,
                 ctx_tissue=tis.embedding,
                 ctx_omics=omx.embedding,
                 prior_vec=pr_vec)
    z_next = torch.argmax(logits, dim=-1)

    # Reconstruction loss (tokenizer) — training-only
    x_hat = tok.decode(q_t)
    recon_loss = tok.recon_loss(x, x_hat)

    print("latent_actions:", a_ids.tolist())
    print("z_next:", tuple(z_next.shape))
    print("recon_loss:", float(recon_loss.detach()))

    # ---- Serving-time (toy) ----
    overlay = dec(z_next)
    print("overlay endosome:", tuple(overlay.endosome_prob.shape),
          "rationale:", overlay.rationale)

if __name__ == "__main__":
    main()
