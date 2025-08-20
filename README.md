![Banner](assets/github_banner.png)

# rna_dna_delivery_model
We want to:
1. Predict if RNA/DNA edits reach intended cells for given vector + context
2. Convert all wet lab steps to be computational
3. Do so while being as quantitatively accurate as what we expect in the wet lab (evals per step)

This is inspired by the ![scAAVengr Pipeline: In Vivo Single-Cell AAV Tropism](https://elifesciences.org/articles/64175) paper that creates a new system on vector delivery into reinal + brain cells. I have written more about this on ![my website](https://bradleywoolf.com/wip-model-the-wet-lab-parts-of-the-scaavengr-pipeline). 


## Getting Started 
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Inputs
Inputs from the scAAVengr experiment (Öztürk et al., eLife 2021)
- `make_inputs_from_paper.py`: script to build `data/capsids.fasta` and `data/promoters.tsv`.
- `data/promoters.tsv`: prewritten example (strengths are placeholders; paper does not report numeric promoter strengths).

To generate `capsids.fasta` with AAV1/2/5/6/8/9 (UniProt) and AAV2 engineered variants (K91, K912, K916, K94), run:

```bash
python make_inputs_from_paper.py
```


## Caveats
- AAVrh10: included in the study, but I didn’t hard-code an accession to avoid guessing, you can add it if you want it in capsids.fasta (the RefSeq entries exist, or you can map to UniProt for your preferred isolate) 
- 4YF/4YFTV/2YF mutants: listed in the paper, but the exact residue map depends on numbering conventions. I left them out by default to avoid silent misannotation (avoid future debugging), can be added after mutation map


## Papers that inspired this
![Organoids with Single Cell Resolution](https://www.nature.com/articles/s41434-022-00360-3)  
![scAAVengr Pipeline: In Vivo Single-Cell AAV Tropism](https://elifesciences.org/articles/64175)  
![BRAVE (Barcoded Rational AAV Vector Evolution)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6936499)  
![Multiplexed In Vivo Barcoded AAV Screens in Large Animals](https://pmc.ncbi.nlm.nih.gov/articles/PMC10503678)