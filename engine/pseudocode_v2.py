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
import pandas ad pd
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

