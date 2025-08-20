# based on outline.txt
# hyperparameters: no idea yet, need to get data first

# goal: predict % expression for (cell_type x capsid x promoter)

from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm, Linear
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from copy import deepcopy
import pandas as pd

# config
class Config:
    # paths
    ARC_DATA_DIR  = "data/arc_vca/"
    CAPSIDS_FASTA = "data/capsids.fasta"          # VP3 sequences (WT + mutants)
    PROMOTER_META = "data/promoters.tsv"          # name, strength, len, GC, CpG
    CACHE_DIR     = "cache/"
    REF_FASTA_OUT = "out/ref_with_barcodes.fasta"
    STAR_INDEX    = "out/star_index/"
    ALIGN_OUT     = "out/align/"
    H5AD_OUT      = "out/scaavengr.h5ad"

    # embedding dimensions
    D_CELL = 32         # scVI latent dimensions
    D_EVO2 = 1024       # TODO: set to actual Evo2 width
    D_ESM2 = 1280       # TODO: set to actual ESM-2 650M hidden width
    D_PROM = 3 + 4      # 3 one-hot + 4 scalar features

    # model widths
    D_PROJ = 256        # projection dimensions
    D_IN = D_PROJ * 3   # cell + capsid + promoter blocks

    # training
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    BATCH = 256
    PATIENCE = 10

    # knobs
    USE_EVO2 = True
    USE_ESM2 = True
    POLLING = "mean"    # "mean" || "attn"
    FINE_TUNE_ESM = False
    DELTA_EMBED = True  # use z(mut) - z(wt) as extra capsid feature
    LOSS = "gauss_null" # predict mean and log_sigma

# data schemas
"""
Cell:   id: str, species: str, tissue: str, cell_type: str, z_cell: ℝ^{D_CELL}
Capsid: id: str, vp3_seq: str, parent_id: str | None, z_evo2: ℝ^{D_EVO2}, z_esm: ℝ^{D_ESM2}
Prom:   id: str ∈ {CAG,CMV,RHO}, onehot: ℝ^{3}, scalars: [strength, GC, len, CpG] -> ℝ^{4}

* {} == PyTorch tensor

Construct (sample): (cell_id, capsid_id, promoter_id) -> y: % expression (synthetic label)
Input x: [ẑ_cell ∥ ẑ_capsid ∥ ẑ_prom]  ∈ ℝ^{D_IN}

* ẑ = normalized/processed version of z
"""

# cell embeddings
def load_arc_cells(dir=Config.ARC_DATA_DIR) -> List[dict]:
    # parse profiles + annotations
    # aggregate to cell types or keep single cells
    # return fetched records: {"cell_id", "species", "tissue", "cell_type", "expr_vector"}
    return

def train_or_load_scvi(cells) -> "scVI_model":
    # train or load scVI model  (latent D_CELL)
    # save latent mapping (in memory for access?)
    return

def make_cell_embeddings(scvi_model, cells) -> Dict[str, np.ndarray]:
    # z_cell ∈ ℝ^{D_CELL} per cell_id (or per cell_type centroid)
    return

# capsid embeddings (Evo2 + ESM-2)
class Evo2Embedder:
    def __init__(self, ckpt="..."):
        return
    
    def embed(self, aa_seq:str) -> np.ndarray: # ℝ^{D_EVO2}
        return
    
class ESM2Embedder:
    def __init__(self, model_name="esm2_650M", fine_tune=Config.FINE_TUNE_ESM):
        return
    
    def token_ids(self, aa_seq:str) -> List[int]:
        return

    def pool(self, states:List[np.ndarray]) -> np.ndarray: # mean or attn to ℝ^{D_ESM2}
        return
    
    def embed(self, aa_seq:str) -> np.ndarray: # ℝ^{D_ESM2}
        return self.pool(self.forward_state(self.token_ids(aa_seq)))
    
def load_capsid_fasta(path=Config.CAPSIDS_FASTA) -> List[str]:
    # return [{"capsid_id", "vp3_seq", "parent_id"(optional)}]
    return

def make_capsid_embeddings(capsids:List[str], evo2:Evo2Embedder, esm2:ESM2Embedder) -> Dict[str, dict]:
    table = {}
    for c in capsids:
        z_evo = evo2.embed(c["vp3_seq"]) if Config.USE_EVO2 else None
        z_esm = esm2.embed(c["vp3_seq"]) if Config.USE_ESM2 else None
        table[c["capsid_id"]] = {"z_evo2": z_evo, "z_esm": z_esm, parent_id: c.get("parent_id")}
    return table

def capsid_feature(capsid_id, cap_tbl) -> np.ndarray:
    # concat available embeddings (+ optional delta vs parent)
    feats = []
    if Config.USE_EVO2: feats.append(cap_tbl[capsid_id]["z_evo2"])
    if Config.USE_ESM2: feats.append(cap_tbl[capsid_id]["z_esm"])
    z = np.concatenate(feats)
    if Config.DELTA_EMBED and capt_tbl[capsid_id]["parent_id"]:
        p = cap_tbl[capsid_id]["parent_id"]
        z_parent = []
        if Config.USE_EVO2: z_parent.append(cap_tbl[p]["z_evo2"])
        if Config.USE_ESM2: z_parent.append(cap_tbl[p]["z_esm"])
        z_parent = np.concatenate([z, np.concatenate(z_parent, -1) * 0 + (z - np.concatenate(z_parent, -1))], -1)  # [z, Δζ]
    return z # ℝ^{D_EVO2 + D_ESM2 [+ same for Δ]}

# promoter encoding
def load_promoters(path=Config.PROMOTER_META) -> Dict[str, dict]:
    # rows: name, strength, len, GC, CpG
    return

def encode_promoter(name: str, meta_tbl) -> np.ndarray: # ℝ^{D_PROM}
    one_hot = onehot(name, classes=["CAG", "CMV", "RHO"])
    scalars = normalize([
        meta_tbl[name]["strength"], 
        meta_tbl[name]["GC"], 
        meta_tbl[name]["len"], 
        meta_tbl[name]["CpG"]])
    return np.concatenate([one_hot, scalars], -1)

# construct table + synthetic labels
def build_constructs(cells, capsids, promoters) -> List[tuple]:
    # cartesian or filtered combos producing (cell_id, capsid_id, promoter_id)
    return

def synthesize_label(cell_id, capsid_id, promoter_id) -> dict:
    # pseudo-ground truth y and noise σ based on scAAVengr heuristics
        # base serotype x cell prior + promoter strength modifier + stochasticity
    return {"y_mean": float, "y_sigma": float}

def make_dataset(constructs, z_cell_map, cap_tbl, prom_tbl) -> "Dataset":
    # materialize X, targets; split by construct (capsid x promoter) to test generalization
    return

# model- PyTorch pseudocode
class TropismNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell_proj = nn.Sequential(LayerNorm(Config.D_CELL), nn.Linear(Config.D_CELL, Config.D_PROJ))
        d_capsid_in = (Config.D_EVO2 if Config.USE_EVO2 else 0) + (Config.D_ESM2 if Config.USE_ESM2 else 0)
        if Config.DELTA_EMBED: d_capsid_in *= 2
        self.capsid_proj = nn.Sequential(LayerNorm(d_capsid_in), nn.Linear(d_capsid_in, Config.D_PROJ))
        self.promoter_proj = nn.Sequential(LayerNorm(Config.D_PROM), nn.Linear(Config.D_PROM, Config.D_PROJ))
        layers = []
        d = Config.D_IN
        for h in Config.LAYERS[:-1]:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.out_mean = nn.Linear(d, 1)
        self.out_log_sigma = nn.Linear(d, 1)

    def forward(self, z_cell, z_capsid, z_promoter):
        x = torch.cat([self.cell_proj(z_cell), self.capsid_proj(z_capsid), self.promoter_proj(z_promoter)], dim=-1)
        h = self.mlp(x)
        mean = self.out_mean(h)
        log_sigma = self.out_log_sigma(h)
        return mean, log_sigma
    

# loss, metrics, split
def gaussian_nll(mean, logS, y_true):
    var = torch.exp(2*logS)
    return 0.5 * (torch.log(2*np.pi*var) + (y_true - mean)**2 / var).mean()

def metrics(pred, y):
    return {"R2": r2_score(y, pred), "Spearman": spearmanr(y, pred).statistic}

def split_by_construct(dataset) -> (train, val, test):
    # stratify on unique (capsid_id, promoter_id); no leakage
    return

# training
def train_model(model, train_loader, val_loader):
    opt = AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WD)
    best = {"val_R2": -1e9, "state": None, "epochs_no_improve": 0}
    for epoch in range(Config.EPOCHS):
        model.train()
        for batch in train_loader:
            mean, logS = model(batch.z_cell, batch.z_capsid, batch.z_prom)
            loss = gaussian_nll(mean, logS, batch.y) if Config.LOSS=="gauss_nll" else mse(mean, batch.y)
            loss.backward(); opt.step(); opt.zero_grad()
        # validate
        model.eval(); preds, ys = [], []
        with torch.no_grad():
            for vb in val_loader:
                m, s = model(vb.z_cell, vb.z_capsid, vb.z_prom)
                preds.append(m); ys.append(vb.y)
        R2 = r2_score(cat(ys), cat(preds))
        if R2 > best["val_R2"]:
            best.update({"val_R2": R2, "state": deepcopy(model.state_dict()), "epochs_no_improve": 0})
        else:
            best["epochs_no_improve"] += 1
            if best["epochs_no_improve"] >= Config.PATIENCE: break
    model.load_state_dict(best["state"]); return model, best

# inference
def predict_grid(model, cells, capsids, promoters, z_cell_map, cap_tbl, prom_tbl) -> DataFrame:
    rows = []
    for ci in cells:
        for vi in capsids:
            for pi in promoters:
                zc = torch.tensor(z_cell_map[ci["cell_id"]])
                zv = torch.tensor(capsid_feature(vi["capsid_id"], cap_tbl))
                zp = torch.tensor(encode_promoter(pi["id"], prom_tbl))
                mean, logS = model(zc, zv, zp)
                rows.append({"cell_id": ci["cell_id"], "capsid_id": vi["capsid_id"], "promoter_id": pi["id"],
                             "pred_mean": float(sigmoid_or_scale(mean)), "pred_sigma": float(torch.exp(logS))})
    return df(rows)

# scRNA-seq simulation (scDesign3 wrapper)
def simulate_scrnaseq(pred_df, n_cells_per_type=200) -> AnnData:
    """
    For each (cell_type, capsid, promoter):
      - convert pred_mean to per-cell transduction probability or expression uplift
      - call scDesign3 to sample counts & integrate 8nt barcodes
    """
    return

# reference genome with barcodes
def build_reference_genome(barcode_map, out_fa=Config.REF_FASTA_OUT):
    """
    Append synthetic barcode sequences as extra contigs or embed into a reporter cassette.
    Maintain a TSV: barcode ↔ (capsid, promoter, cell_type).
    """
    return

# alignment (STARsolo)
def align_with_starsolo(fq1, fq2, ref_fa=Config.REF_FASTA_OUT, index_dir=Config.STAR_INDEX, out_dir=Config.ALIGN_OUT):
    """
    if not exists(index_dir): run: STAR --runMode genomeGenerate --genomeDir index_dir --genomeFastaFiles ref_fa ...
    run: STAR --soloType CB_UMI_Simple --readFilesIn fq1 fq2 --genomeDir index_dir --outFileNamePrefix out_dir ...
    """
    return

# Scanpy analysis
def scanpy_qc_cluster_umap(aln_mtx_dir=Config.ALIGN_OUT) -> AnnData:
    """
    Load matrix; QC filters; normalize; neighbors; Leiden; UMAP.
    Add obs[["cell_type","capsid","promoter"]] via barcode_map.
    Compute CellType x Serotype heatmap on predicted/observed expression.
    Save .h5ad at Config.H5AD_OUT (CellxGene-ready).
    """
    return

# pipeline
def main():
    # cells
    cells = load_arc_cells()
    scvi = train_or_load_scvi(cells)
    z_cell_map = make_cell_embeddings(scvi, cells)

    # capsids
    capsids = load_capsid_fasta()
    evo2 = Evo2Embedder() if Config.USE_EVO2 else None
    esm2 = ESM2Embedder()
    cap_tbl = make_capsid_embeddings(capsids, evo2, esm2)

    # promoters
    prom_tbl = load_promoters()

    # dataset
    constructs = build_constructs(cells, capsids, prom_tbl.keys())
    dataset = make_dataset(constructs, z_cell_map, cap_tbl, prom_tbl)
    train, val, test = split_by_construct(dataset)

    # train
    model = TropismNet()
    model, best = train_model(model, DataLoader(train, Config.BATCH), DataLoader(val, Config.BATCH))

    # eval
    y_hat = predict_on(test, model)
    print(metrics(y_hat, test.targets))

    # predict full grid
    pred_df = predict_grid(model, cells, capsids, list(prom_tbl.values()), z_cell_map, cap_tbl, prom_tbl)

    # simulate scRNA-seq + barcodes
    adata = simulate_scrnaseq(pred_df, n_cells_per_type=200)
    barcode_map = extract_barcode_map(adata)
    build_reference_genome(barcode_map)

    # align & analyze
    align_with_starsolo("reads_R1.fq.gz", "reads_R2.fq.gz")
    ad = scanpy_qc_cluster_umap()
    ad.write(Config.H5AD_OUT)

    # share
    print(f"CellxGene-ready file at: {Config.H5AD_OUT}")


if __name__ == "__main__":
    main()