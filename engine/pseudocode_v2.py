"""
Computational scAAVengr v2.0

Customer: gene therapy researcher

Goal: predict AAV capsid x promoter transduction / specific cells

Inputs:
- species, tissue, target_cell_types
- atlas/*.h5ad                          # retina/brain atlas
- capsids.fasta                         # viral protein (wild-type + variants)
- promoters.yaml                        # {name: {sequence or cell_type_weights}}
- barcodes.csv                          # serotype, 8nt_barcode
- priors.yaml                           # optional: weak capsid cell type biases (from expert biologists)

Outputs:
- outputs/predicted_tropism.csv
- outputs/synthetic_scRNAseq.h5ad
- analysis/celltype_serotype_matrix.csv
- analysis/figures/*, outputs/report.md
- ref_build/GRCh38_plusAAV/             # barcoded-augmented reference, for whatever species
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# ---------- 0) utilities ----------
class Config:
    inputs_dir = Path("inputs")
    outputs_dir = Path("outputs")
    work_dir = Path("pipeline")
    ref_dir = Path("ref_build")
    base_ref = Path("/refs/GRCh38")
    rng_seed = 42 # random- will outsource to some library

def load_capsids(fasta_path: Path) -> Dict[str, str]:
    """Return {serotype: AA_sequence} from FASTA"""
    return 

def load_barcodes(csv_path: Path) -> pd.DataFrame:
    """Columns: ["serotype","barcode_8nt"] (validated unique)"""
    df = pd.read_csv(csv_path)
    assert df["barcode_8nt"].nunique() == len(df)
    return df

def load_promoters(yaml_path: Path) -> Dict:
    """Return promoter metadata; allow weights per cell_type or raw sequence"""
    return

def load_atlas(h5ad_path: Path):
    """Return AnnData with .obs["cell_type"] present"""
    # or some urls to Lamin/Arc/Tahoe
    return

# optional
def load_priors(yaml_path: Path):
    """Weak biases per (serotype, cell_type)"""
    return

# ---------- 1) representations ----------
def embed_capsids(seqs: Dict[str, str]) -> pd.DataFrame:
    """
    Use protein LLM (ESM-2) to embed VP3 AA sequences.
    Return DataFrame index=serotype, cols=embedding dims
    """
    # model = load_protein_llm("esm2_t33_650M_UR50D")
    # emb = {s: model.embed(seq) for s, seq in seqs.items()}
    return

def encode_promoters(promoters: Dict, cell_types: List[str]) -> pd.DataFrame:
    """
    If promoter weights per cell_type exist, use them directly
    Else, derive features from sequence (k-mer, motif hits) and map to cell_types
    Return DataFrame index=promoter, cols=cell_types [weights in [0...1]]
    """
    return

def embed_cell_types(atlas) -> pd.DataFrame:
    """
    Build low-dim cell-type vectors (PCA on marker panels)
    Return DataFrame index=cell_type, cols=embedding_dims
    """
    return

# ---------- 2) predict tropism (cell_type x serotype x promoter) ----------
def apply_priors(logits: pd.DataFrame, priors: Dict) -> pd.DataFrame:
    """
    Add small logit bases from priors.yaml:
    logits.loc[cell_type, serotype] += bias
    """
    return

def calibrate(probs: pd.DataFrame) -> pd.DataFrame:
    """
    Temperature scaling / isotonic to improve probability calibration
    """
    return

# only part that i am still pretty unsure of
def predict_tropism(
        capsid_emb: pd.DataFrame,
        promoter_weights: pd.DataFrame,
        celltype_emb: pd.DataFrame,
        priors: Dict | None
) -> pd.DataFrame:
    """
    For each (cell_type, serotype, promoter), predict transduction rate in [0...1]
    Model: stacked ensemble (GBDT + small MLP) -> blended logits -> calibrated probabilities
    Return long-form DataFrame: ["cell_type", "serotype", "promoter", "rate_mean", "rate_lower", "rate_upper"]
    """
    # construct feature triplets
    # X = concat([capsid_emb[serotype], promoter_weights[promoter][cell_type], cell_type_emb[cell_type]])
    # y = None (self-supervised + priors, we just outline inference)
    # logits = model(X)
    # if priors: logits = apply_prioirs(logits, priors)
    # include uncertainty via MC dropout to get (mean, lower, upper)
    return

# ---------- 3) synthetic scRNA-seq generation ----------
def synthesize_scrna(
        atlas, 
        predicted: pd.DataFrame, 
        barcodes: pd.DataFrame, 
        n_cells_per_type: int=2000
):
    """
    Use atlas stats to sample per-cell expression, then inject AAV-barcode counts according 
    to predicted rates per (cell_type, serotype, promoter).
    Return AnnData (cells x genes + barcode_features) wtih realistic llibrary size/dropout
    """
    # for each cell_type:
    #   simulate counts ~ NB(mu, theta) matched to atlas QC
    #   for each serotype: draw barcode_presence ~ Bernoulli(rate)
    #   add a sparse 'GFP_<serotype>' feature with UMI counts if present
    # save as outputs/synthetic_scRNAseq.h5ad
    return

# ---------- 4) reference augmentation ----------
def make_barcode_fasta_gtf(barcodes: pd.DataFrame, out_dir: Path):
    """
    Emit aav_barcodes.fa / aav_barcodes.gtf with one mini-gene per barcode,
    then shell out to CellRanger mkref or build STAR indices
    """
    # aav_barcodes.fa
    # >GFP_<serotype>\nTAAATCGATCG<barcode_8nt>\n
    # # aav_barcodes.gtf: single-exon entries with gene_id/transcript_id = GFP_<serotype>
    return

# ---------- 5) analysis (ScanPy/Seurat-like) ----------
def compute_celltype_serotype_matrix(h5ad_path: Path) -> pd.DataFrame:
    """
    Cluster (Leiden), embed (UMAP), annotate (pre-labeled from atlas),
    then compute % cells positive for each (cell_type, serotype) barcode for each cell.
    Return DataFrame rows=cell_type, cols=serotype
    """
    return

# ---------- 6) reporting ----------
def write_report(
        pred_csv: Path, 
        cxs_csv: Path, 
        figs_dir: Path, 
        out_md: Path
):
    """
    Summarize: top-1 serotype per cell type, heatmap paths, QC summaries,
    calibration notes, assumptions with minimal markdown
    """
    return

# ---------- main ----------
def main():
    cfg = Config()

    # inputs
    capsids = load_capsids(cfg.data_dir / "capsids.fasta")
    barcodes = load_barcodes(cfg.data_dir / "barcodes.csv")
    promoters = load_promoters(cfg.data_dir / "promoters.yaml")
    atlas_retina = load_atlas(cfg.data_dir / "atlas" / "retina.h5ad")  # or brain.h5ad
    priors = load_priors(cfg.data_dir / "priors.yaml") if (cfg.data_dir / "priors.yaml").exists() else None

    # representations
    capsid_emb = embed_capsids(capsids)
    celltype_emb = embed_cell_types(atlas_retina)
    promoter_weights = encode_promoters(promoters, cell_types=list(celltype_emb.index))

    # predict tropism
    predicted = predict_tropism(capsid_emb, promoter_weights, celltype_emb, priors)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(cfg.out_dir / "predicted_tropism.csv", index=False)

    # scRNA-seq
    adata = synthesize_scrna(atlas_retina, predicted, barcodes, n_cells_per_type=2000)
    # adata.write_h5ad(cfg.out_dir / "synthetic_scRNAseq.h5ad")

    # build barcode reference (for aligners; optional if not generating FASTQs)
    cfg.ref_dir.mkdir(parents=True, exist_ok=True)
    make_barcode_fasta_gtf(barcodes, cfg.ref_dir)
    # -> run CellRanger mkref or STAR index externally

    # analysis (from synthetic data)
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    cxs = compute_celltype_serotype_matrix(cfg.out_dir / "synthetic_scRNAseq.h5ad")
    cxs.to_csv(cfg.analysis_dir / "celltype_serotype_matrix.csv")

    # reporting
    write_report(
        pred_csv = cfg.out_dir / "predicted_tropism.csv",
        cxs_csv  = cfg.analysis_dir / "celltype_serotype_matrix.csv",
        figs_dir = cfg.analysis_dir / "figures",
        out_md   = cfg.out_dir / "report.md",
    )

if __name__ == "__main__":
    main()