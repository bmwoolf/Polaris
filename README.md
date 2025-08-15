# rna_dna_delivery_model
We want to:
1. Predict if RNA/DNA edits reach intended cells for given vector + context
2. Convert all wet lab steps to be computational
3. Do so while being as quantitatively accurate as what we expect in the wet lab (evals per step)

## Getting Started 
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Papers that inspired this
![Organoids with Single Cell Resolution](https://www.nature.com/articles/s41434-022-00360-3)
![scAAVengr Pipeline: In Vivo Single-Cell AAV Tropism](https://elifesciences.org/articles/64175)
![BRAVE (Barcoded Rational AAV Vector Evolution)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6936499)
![Multiplexed In Vivo Barcoded AAV Screens in Large Animals](https://pmc.ncbi.nlm.nih.gov/articles/PMC10503678)