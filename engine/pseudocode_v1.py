# end to end computational scAAVengr

# ---------- inputs ----------
cell_atlas = load_atlas("inputs/atlas/*")                       # Arc Virtual Cell Atlas- human retina + brain
capsid_sequences = load_sequences("inputs/capsids.fasta")       # AAV2, AAV6, DJ, Anc80
promoter_sequences = load_sequences("inputs/promoters.yaml")    # CAG, CMV, RHO
barcode_sequences = load_sequences("inputs/barcodes.csv")       # 8nt barcodes for AAV

# ---------- 1) cell model ----------
cells = arc_state.query(atlas, tissues=["retina", "brain"])
scvi_model = scvi.fit(cells)                                    # batch-correct + latent
cell_vectors = scvi_model.encode(cells)                         # |C| x d_cell

# ---------- 2) capsid / promoter encoding ----------
capsid_vectors = {s: evo2.embed(seq) or esm2.embed(seq)
                  for s, seq in capsid_sequences.items()}       # |S| x d_capsid
prom_vectors = one_hot_or_learned(promoter_sequences)              # |P| x d_prom

# ---------- 3) tropism prediction ----------
# training data: published scAAVengr cell-type x serotype x promoter rates
train = load_scAAVengr_labels()                                 # {(cell_type, serotype, promoter) -> rate}
X = []
y = []
for (ct, s, p), rate in train.items():
    X.append(concat(cell_vectors[ct], capsid_vectors[s], prom_vectors[p]))
    y.append(rate)
tropism_head = MLP()
tropism_head.fit(X, y)

# optional physics sanity
if USE_PHYSICS:
    phys = {s: rosetta_capsid_scores(capsids[s]) for s in capsids}
    tropism_head = augment_with_physics(tropism_head, phys)

# inference grid
pred = {}
for ct in cell_vecs:
    for s in capsid_vecs:
        for p in prom_vectors:
            x = concat(call_vectors[ct], capsid_vectors[s], prom_vectors[p])
            pred[(ct, s, p)] = tropism_head.predict(x)
save_csv(pred, "outputs/predicted_tropism.csv")

# ---------- 4) synthetic scRNA-seq ----------
# label cells stochastically by predicted transduction (multiplicity of infection-aware)
labels = sample_transduction_labels(cells, pred, moi=TARGET_MOI)
# generate counts conditioned on labels (preserve gene-gene covariance)
synthetic = scvi_model.generate(cells, labels, n_reads=TARGET_DEPTH)

# attach AAV barcodes to transduced reads
synthetic_fastq = inject_barcodes(synthetic, barcodes)

# ---------- 5) reference build ----------
aug_fasta, aug_gtf = add_barcodes_to_reference(barcodes, base_fasta, base_gtf)
write_reference("pipeline/04_reference_build/", aug_fasta, aug_gtf)

# ---------- 6) alignment and analysis ----------
h5ad = STARsolo.align(synthetic_fastqs, aug_fasta, aug_gtf)
adata = scanpy.read_h5ad(h5ad)
adata = scanpy_basic_qc(adata)
adata = scvi.reembed(adata, scvi_model)                         # joint latent space
clusters = scanpy_umap(adata)
heatmap = celltype_serotype_heatmap(adata, labels)

save_h5ad(adata, "outputs/synthetic_scRNAseq.h5ad")
save_figs({"umap": umap, "heatmap": heatmap}, "outputs/figures/")

# ---------- 7) evaluation (vs scAAVengr) ----------
priors = load_scAAVengr_priors()
corr = correlation(heatmap, priors)
knn_pres = knn_structure_preservation(adata, atlas)
uncert = celltype_cv(adata)

assert corr >= 0.6
assert knn_pres >= 0.8
assert uncert <= 0.225

# ---------- 8) share ----------
cellxgene.export(adata, "outputs/cellxgene/")

# ---------- example directory structure ----------
# inputs/ -> capsids.fasta, barcodes.csv, promoters.yaml, atlas/
# pipeline/01_cell_model_embeddings -> scVI training artifacts
# pipeline/01_capsid_embeddings     -> Evo2/ESM2 embeddings
# pipeline/02_tropism_prediction    -> MLP weights, calibration
# pipeline/03_synthetic_scRNAseq    -> FASTQs with barcodes
# pipeline/04_reference_build       -> augmented FASTA/GTF
# pipeline/05_alignment_analysis    -> STARsolo outputs, h5ad
# pipeline/06_reporting             -> figures/, exports/