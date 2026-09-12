
# Peptide clustering + consensus train/test/CV split

Five notebooks exploring whether a **consensus of independent clustering methods**
produces a better-justified train/test/CV split for `data/mic_classification_dataset.csv`
than the dataset's existing `split` column (which comes from a single method — QMAP's
own sequence-identity Leiden splitter). All five deliberately ignore the existing
`split` column and build a fresh one from scratch.

This is exploratory, self-contained notebook work — no wandb, no pipeline
orchestration, no CI. Every clustering call lives in its own cell with parameters as
named constants near the top of that cell, so you can rerun any one method with
different parameters without rerunning anything upstream of it.

## Run order

```
01_fingerprint_and_descriptor_clustering.ipynb   (methods a + b)
02_peptideclm_clustering.ipynb                    (method c)
03_qmap_sequence_clustering.ipynb                 (method d)
        \_______________ all three feed into ______/
04_consensus_comparison_and_split.ipynb           (consensus + train/test/CV split)
05_diagnostics.ipynb                              (coverage, ARI, UMAP, kNN checks)
```

01, 02, and 03 can run in any order relative to each other (each rebuilds its own
base frame from the source CSVs and merges in whatever voter columns the shared
table `data/clustering/peptide_voters.parquet` already has) — but all three must run
before 04, and 04 must run before 05.

## What each notebook produces

| Notebook | Method | Output column(s) | Coverage |
|---|---|---|---|
| 01 | (a) Morgan/ECFP + Tanimoto + Butina | `cluster_fingerprint` | 12,371/12,371 (100%) |
| 01 | (b) RDKit descriptors + k-means | `cluster_descriptor` | 12,371/12,371 (100%) |
| 02 | (c) PeptideCLM embeddings + k-means | `cluster_peptideclm` | 12,371/12,371 (100%) |
| 03 | (d) QMAP sequence identity + Leiden | `cluster_qmap` | 10,888/12,371 (88.0%) — see below |

Method (d)'s coverage gap is real, not a bug: 1,483 peptides contain DBAASP's own
unresolved-residue placeholder (`X`/`x`) in their `sequence` column, for which no
valid sequence representation exists (confirmed — `p2smi`, this repo's own
sequence↔SMILES tool, has no SMILES-to-sequence reverse capability either). These
get a null vote in `cluster_qmap`, never a guessed cluster. A further 999
non-canonical peptides *are* scoreable via case-fold linearization (D-form residues
are lowercase in DBAASP's encoding) — notebook 03's coverage table breaks this down
in full, further cross-tabulated against linear vs. non-linear (`bond_type`).

Shared artifacts, all under `data/clustering/`:
- `peptide_voters.parquet` — one row per unique `peptide_id` (12,371 rows,
  independent of which/how-many organisms it was tested against), accumulating one
  column per method as 01/02/03 run.
- `peptideclm_embeddings.npy` — the raw 768-dim PeptideCLM embeddings (notebook 02),
  reused by notebook 05 for UMAP.
- `consensus_split.csv` — notebook 04's final output: `peptide_id`,
  `cluster_fingerprint`, `cluster_descriptor`, `cluster_peptideclm`, `cluster_qmap`,
  `consensus_coassoc`, `consensus_votegraph`, `community` (mirrors whichever of the
  two consensus columns was chosen via `CHOSEN_CONSENSUS`), `split` (`train`/`test`),
  `fold_id` (0-4, `NaN` for test rows), `is_linear`. Shaped like the existing
  `data/train_folds_leiden.csv` (`node_id`/`community`/`fold_id`) so it's a drop-in
  alternative input to the `build_dataset(row_groups=...)` k-fold pattern already
  demonstrated in `scripts/EDA/build_dataset_build_model_kfold_demo.ipynb`.

## The two consensus mechanisms, side by side

Both are computed from the same pairwise co-association matrix (for each peptide
pair, the fraction of the four methods that agree, among only the methods that
scored both peptides in the pair) and both take an explicit `voter_columns` list, so
comparing e.g. all-four vs. without-QMAP is a one-line change in notebook 04 with no
upstream rerun needed.

| | 4a. Co-association consensus | 4b. Vote-graph + Leiden |
|---|---|---|
| Mechanism | Agglomerative (average-linkage) clustering on `1 - co-association` | `leidenalg` Leiden community detection on a co-association-weighted graph |
| Key parameter | `N_CONSENSUS_CLUSTERS` (default 40) | `MIN_EDGE_WEIGHT` (default 0.7) |
| Clusters produced (defaults, this dataset) | 40 | 556 (mostly small — see notebook 04's note on why 0.7 was chosen) |
| Adjusted Rand Index between the two | **0.462** (moderate agreement — informative to inspect further, not a rubber stamp for either) | |
| Peptides landing in a different group depending on method | 37.3% (relative to the majority vote-graph cluster for each co-association cluster) | |

**`CHOSEN_CONSENSUS`** in notebook 04 is the single switch that decides which of the
two feeds the final train/test/CV split (`"consensus_coassoc"` by default) — read
notebook 04's step 5 comparison and notebook 05's diagnostics before deciding whether
to change it.

## Final split (with the defaults as shipped)

Test carved out by *whole consensus cluster* (never splitting a cluster across
train/test), targeting 20% of peptides:

- **Test:** 2,700 peptides (21.8%)
- **Train pool → 5 CV folds:** 498 / 2,109 / 3,218 / 1,712 / 2,134 peptides — some
  imbalance (fold 0 is notably smaller; see notebook 05's linearity-per-fold table,
  which also shows fold 0 skews non-linear at 44.8% vs. the ~20% dataset average) —
  worth a look before treating all folds as interchangeable.

Every number above is regenerated by rerunning the notebooks — this table just
reflects what the defaults produced during development, not a fixed target.
