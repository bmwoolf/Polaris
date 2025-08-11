# rna_dna_delivery_model
Gene therapy model #1

Problem: Predict if RNA/DNA edits reach intended cells for given vector + context

Scope: 3–4 tissues, AAV only for v1, human + mouse

Metric: AUROC ≥ 0.8 on held-out study

## Getting Started 
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Sources
NCBI SRA  
PDB  
AddGene  
SCGE Consortium Data Portal  