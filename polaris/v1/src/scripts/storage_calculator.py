#!/usr/bin/env python3
"""
Storage budget calculator for in-silico scAAVengr runs.

Two modes:
  A) keep_raw=True   → includes compressed FASTQs, STAR index, matrices, models.
  B) keep_raw=False  → matrices + latents + models (delete FASTQs/BAMs immediately).

Assumptions are parameterized; defaults reflect scAAVengr (100k read pairs/cell, 2x150bp).
"""

from __future__ import annotations
import argparse, math, json
from dataclasses import dataclass, asdict

# helpers
GB = 1024**3
def bytes_to_gb(x: float) -> float: return x / GB
def gb(x: float) -> float: return float(x)  # for readability

@dataclass
class Assumptions:
    # sequencing / compression
    read_pairs_per_cell: int = 100_000           # pairs/cell (paper target)
    gb_per_10m_pairs: float = 1.0                # compressed FASTQ ≈ GB / 10M read pairs (0.5–1.5 typical)
    # matrices
    genes: int = 20_000
    matrix_density: float = 0.05                 # fraction nonzero (5% default)
    bytes_per_nnz: int = 12                      # sparse CSR/COO approx (value+indices)
    # models / indices
    esm2_gb: float = 2.6                         # ESM-2 650M
    evo2_gb: float = 3.0                         # Evo2 1B base
    star_index_gb: float = 3.0
    # embeddings
    n_capsids: int = 1_000
    capsid_dims: int = 1_024 + 1_280            # Evo2 + ESM-2
    include_delta_embed: bool = False            # doubles capsid dims if True
    # scVI latents
    scvi_latent_dim: int = 32
    # other
    overhead_gb: float = 5.0                     # logs, QC, metadata, safety cushion

@dataclass
class BudgetResult:
    mode: str
    cells: int
    total_gb: float
    components_gb: dict

# core sizing
def size_fastqs_gb(cells: int, a: Assumptions) -> float:
    total_pairs = cells * a.read_pairs_per_cell
    return (total_pairs / 10_000_000.0) * a.gb_per_10m_pairs

def size_matrix_gb(cells: int, a: Assumptions) -> float:
    nnz = cells * a.genes * a.matrix_density
    return bytes_to_gb(nnz * a.bytes_per_nnz)

def size_scvi_latents_gb(cells: int, a: Assumptions) -> float:
    bytes_ = cells * a.scvi_latent_dim * 4
    return bytes_to_gb(bytes_)

def size_capsid_embeds_gb(a: Assumptions) -> float:
    dims = a.capsid_dims * (2 if a.include_delta_embed else 1)
    bytes_ = a.n_capsids * dims * 4
    return bytes_to_gb(bytes_)

def size_models_and_index_gb(a: Assumptions, include_star_index: bool) -> float:
    total = a.esm2_gb + a.evo2_gb
    if include_star_index:
        total += a.star_index_gb
    return total

# forward compute (given cells)
def compute_total_gb(cells: int, keep_raw: bool, a: Assumptions) -> BudgetResult:
    capsids = size_capsid_embeds_gb(a)
    models = size_models_and_index_gb(a, include_star_index=True)
    latents = size_scvi_latents_gb(cells, a)
    matrix = size_matrix_gb(cells, a)

    components = {
        "models+STAR": round(models, 3),
        "capsid_embeddings": round(capsids, 3),
        "scVI_latents": round(latents, 3),
        "sparse_matrix": round(matrix, 3),
        "overhead": round(a.overhead_gb, 3)
    }

    total = models + capsids + latents + matrix + a.overhead_gb

    if keep_raw:
        fastqs = size_fastqs_gb(cells, a)
        components["FASTQs_compressed"] = round(fastqs, 3)
        total += fastqs

    return BudgetResult(
        mode="RAW-HEAVY" if keep_raw else "LEAN",
        cells=cells,
        total_gb=round(total, 3),
        components_gb=components
    )

# inverse compute (max cells under budget)
def max_cells_under_budget(budget_gb: float, keep_raw: bool, a: Assumptions) -> BudgetResult:
    # Non-cell-dependent constants
    base_const = size_capsid_embeds_gb(a) + size_models_and_index_gb(a, True) + a.overhead_gb
    # Per-cell linear terms
    per_cell_matrix_gb = size_matrix_gb(1, a)
    per_cell_latent_gb = size_scvi_latents_gb(1, a)
    per_cell_fastq_gb = (a.read_pairs_per_cell / 10_000_000.0) * a.gb_per_10m_pairs if keep_raw else 0.0

    per_cell_total = per_cell_matrix_gb + per_cell_latent_gb + per_cell_fastq_gb

    if per_cell_total <= 0:
        raise ValueError("Per-cell storage computed as <= 0; check assumptions.")

    max_cells = int(max(0, math.floor((budget_gb - base_const) / per_cell_total)))

    # Compose components at that cell count
    return compute_total_gb(max_cells, keep_raw, a)

# CLI
def main():
    p = argparse.ArgumentParser(description="Storage budget calculator for scAAVengr.")
    p.add_argument("--budget_gb", type=float, default=1024.0, help="Total storage budget in GB (default: 1024 ≈ 1 TB).")
    p.add_argument("--cells", type=int, default=None, help="If set, compute total GB for this many cells.")
    p.add_argument("--keep_raw", action="store_true", help="Include compressed FASTQs in sizing.")
    p.add_argument("--read_pairs_per_cell", type=int, default=100_000)
    p.add_argument("--gb_per_10m_pairs", type=float, default=1.0)
    p.add_argument("--genes", type=int, default=20_000)
    p.add_argument("--matrix_density", type=float, default=0.05)
    p.add_argument("--bytes_per_nnz", type=int, default=12)
    p.add_argument("--esm2_gb", type=float, default=2.6)
    p.add_argument("--evo2_gb", type=float, default=3.0)
    p.add_argument("--star_index_gb", type=float, default=3.0)
    p.add_argument("--n_capsids", type=int, default=1000)
    p.add_argument("--capsid_dims", type=int, default=2304)  # Evo2 1024 + ESM2 1280
    p.add_argument("--include_delta_embed", action="store_true")
    p.add_argument("--scvi_latent_dim", type=int, default=32)
    p.add_argument("--overhead_gb", type=float, default=5.0)
    p.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = p.parse_args()

    a = Assumptions(
        read_pairs_per_cell=args.read_pairs_per_cell,
        gb_per_10m_pairs=args.gb_per_10m_pairs,
        genes=args.genes,
        matrix_density=args.matrix_density,
        bytes_per_nnz=args.bytes_per_nnz,
        esm2_gb=args.esm2_gb,
        evo2_gb=args.evo2_gb,
        star_index_gb=args.star_index_gb,
        n_capsids=args.n_capsids,
        capsid_dims=args.capsid_dims,
        include_delta_embed=args.include_delta_embed,
        scvi_latent_dim=args.scvi_latent_dim,
        overhead_gb=args.overhead_gb
    )

    if args.cells is not None:
        res = compute_total_gb(args.cells, args.keep_raw, a)
    else:
        res = max_cells_under_budget(args.budget_gb, args.keep_raw, a)

    if args.json:
        out = { "assumptions": asdict(a), "result": {
            "mode": res.mode,
            "cells": res.cells,
            "total_gb": res.total_gb,
            "components_gb": res.components_gb
        }}
        print(json.dumps(out, indent=2))
    else:
        print("\n=== STORAGE BUDGET ===")
        print(f"Mode: {res.mode}")
        print(f"Total budget target: {args.budget_gb} GB" if args.cells is None else f"Cells requested: {args.cells}")
        print(f"Computed total: {res.total_gb:.3f} GB")
        print("Components (GB):")
        for k, v in res.components_gb.items():
            print(f"  - {k:18s}: {v:.3f}")
        if args.cells is None:
            print(f"\nMax cells under budget: {res.cells:,}")
        print("\nAssumptions:")
        for k, v in asdict(a).items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
