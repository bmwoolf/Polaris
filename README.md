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

#### Databases
NCBI SRA  
PDB  
AddGene  
SCGE Consortium Data Portal  

#### Papers
![Organoids with Single Cell Resolution](https://www.nature.com/articles/s41434-022-00360-3)
![scAAVengr Pipeline: In Vivo Single-Cell AAV Tropism](https://elifesciences.org/articles/64175)
![BRAVE (Barcoded Rational AAV Vector Evolution)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6936499)
![Multiplexed In Vivo Barcoded AAV Screens in Large Animals](https://pmc.ncbi.nlm.nih.gov/articles/PMC10503678)