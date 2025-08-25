# make_inputs_from_paper.py
# Generate capsid FASTA and promoter metadata from Öztürk et al., eLife 2021 (scAAVengr).
# - Pulls VP1 protein sequences from UniProt for natural serotypes.
# - Builds engineered AAV2 variants by inserting the reported 10-aa peptides at VP1 position 588.
# - Writes data/capsids.fasta and data/promoters.tsv.
#
# Notes:
# * Engineered inserts and position are taken from the paper:
#   K916: PAPQDTTKKA @ ~588; K912: LAPDSTTRSA @ ~588; K91: LAHQDTTKNA @ ~588; K94: LATTSQNKPA @ ~588.
# * The paper primarily used the CAG promoter (for scCAG-GFP); CMV was used for the SaCas9 validation.
# * Tyrosine-mutant variants (AAV2-4YF, AAV2-4YFTV, AAV8-2YF, AAV9-2YF) are listed in the paper, but
#   exact mutation coordinates vary by numbering scheme; implement them only if you supply a vetted map.
#
# Usage:
#   python make_inputs_from_paper.py
#
# Requires: requests, pandas
import os
import sys
import requests
import pandas as pd

UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

# UniProt accessions for VP1 (or "capsid protein") per serotype
ACCESSIONS = {
    "AAV1_VP1": "Q9WBP8",   # Adeno-associated virus 1 capsid protein
    "AAV2_VP1": "P03135",   # Adeno-associated virus 2 VP1
    "AAV5_VP1": "Q9YIJ1",   # AAV5 capsid protein
    "AAV8_VP1": "Q8JQF8",   # AAV8 capsid protein
    "AAV9_VP1": "Q6JC40",   # AAV9 VP1
    # "AAVrh10_VP1": <add accession or RefSeq>,
}

# Engineered AAV2 10-aa insert variants at VP1 position ~588 (1-based index in VP1)
ENGINEERED_INSERTS = {
    "AAV2_K916_ins588_PAPQDTTKKA": "PAPQDTTKKA",
    "AAV2_K912_ins588_LAPDSTTRSA": "LAPDSTTRSA",
    "AAV2_K91_ins588_LAHQDTTKNA":  "LAHQDTTKNA",
    "AAV2_K94_ins588_LATTSQNKPA":  "LATTSQNKPA",
}

def fetch_uniprot_fasta(accession: str) -> str:
    r = requests.get(UNIPROT_FASTA.format(acc=accession), timeout=30)
    r.raise_for_status()
    return r.text

def fasta_to_seq(fasta_text: str) -> str:
    lines = [ln.strip() for ln in fasta_text.splitlines() if ln.strip()]
    seq = "".join(ln for ln in lines if not ln.startswith(">"))
    return seq

def write_fasta(records, path):
    with open(path, "w") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            # wrap at 80
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

def build_engineered_from_aav2(aav2_seq: str, pos_1based: int, name_to_insert: dict):
    # Insert AA peptides into VP1 at 1-based index (before original aa at that index)
    recs = []
    if pos_1based < 1 or pos_1based > len(aav2_seq)+1:
        raise ValueError("Insertion position out of bounds for AAV2 sequence length %d" % len(aav2_seq))
    for name, peptide in name_to_insert.items():
        new_seq = aav2_seq[:pos_1based-1] + peptide + aav2_seq[pos_1based-1:]
        recs.append((name, new_seq))
    return recs

def main():
    out_dir = os.path.join("data")
    os.makedirs(out_dir, exist_ok=True)

    # Fetch natural serotypes
    fasta_records = []
    aav2_seq = None
    for label, acc in ACCESSIONS.items():
        try:
            fa = fetch_uniprot_fasta(acc)
            seq = fasta_to_seq(fa)
            fasta_records.append((label, seq))
            if label.startswith("AAV2"):
                aav2_seq = seq
        except Exception as e:
            print(f"[warn] Failed to fetch {label} ({acc}): {e}", file=sys.stderr)

    if aav2_seq is None:
        print("[fatal] Could not retrieve AAV2 VP1 sequence; engineered variants cannot be built.", file=sys.stderr)
        sys.exit(2)

    # Build engineered AAV2 variants at VP1 position 588 (per paper; ~588). Using 588 exactly.
    engineered = build_engineered_from_aav2(aav2_seq, pos_1based=588, name_to_insert=ENGINEERED_INSERTS)
    fasta_records.extend(engineered)

    fasta_path = os.path.join(out_dir, "capsids.fasta")
    write_fasta(fasta_records, fasta_path)
    print(f"[ok] Wrote {fasta_path} with {len(fasta_records)} entries.")

    # Promoters table (minimal, paper-grounded). Strengths are placeholders (normalize to max=1).
    rows = [
        {"name": "CAG", "strength": 1.0, "len": 1700, "GC": 0.56, "CpG": 0.012},
        {"name": "CMV", "strength": 0.9, "len": 600,  "GC": 0.62, "CpG": 0.015},
    ]
    df = pd.DataFrame(rows, columns=["name","strength","len","GC","CpG"])
    df.to_csv(os.path.join(out_dir, "promoters.tsv"), sep="\t", index=False)
    print(f"[ok] Wrote {os.path.join(out_dir, 'promoters.tsv')}.")

if __name__ == "__main__":
    main()
