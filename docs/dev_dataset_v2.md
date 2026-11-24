# Dev dataset v2: seven strains, recomputed MIC, NA rows, cluster-stratified split

Everything is written to `data/dev_v2/` and `reports/dev_v2/`; `data/dev/*` is untouched, so earlier
dev results stay reproducible. Run order (each stage re-runnable):

```
python pipeline/data/04_recompute_mic.py                                              # MIC in ug/ml from concentration
python pipeline/data/03_build_dev_classification_dataset.py --config config/data/dev_v2.yaml   # labels + split
python scripts/splitting/measure_dev_split.py --config config/splitting/dev_v2.yaml \
    --split data/dev_v2/dbaasp_dev_split_final.parquet --restrict-to-split-peptides \
    --dataset-csv data/dev_v2/mic_classification_dataset.csv                          # leakage / balance check
```

## 1. Scope
Seven strains (E. coli ATCC 25922, S. aureus ATCC 25923, P. aeruginosa ATCC 27853, A. baumannii ATCC 19606,
K. pneumoniae ATCC 700603, E. faecalis ATCC 29212, B. subtilis ATCC 6633): 30,568 activity rows of
`data/dev/dbaasp_dev_processed.parquet`.

## 2. MIC recomputation (`src/soamp/data/mic_recompute.py`)
DBAASP's `activity` is the concentration converted with DBAASP's own molecular weight, which is unusable for
some peptides (implied MW 17.03, 1.01, 0.0, 45.09 g/mol; 8 peptides, list in
`reports/dev_v2/dbaasp_bogus_mw_peptides.csv`). The MIC is therefore recomputed from `concentration` + `unit`:

| case | `mic_ug_ml` |
|---|---|
| unit ug/ml | the number as written |
| unit uM | uM x RDKit MolWt(SMILES) / 1000 |
| concentration `NA` (any unit) | **1,111,111** (DBAASP sentinel) |
| no unit / empty / unparsable / unsupported unit | dropped, never guessed |

The number picked from the text follows DBAASP's convention (it reproduces `activity` for ug/ml rows,
99.98% within 0.1%): range or `<x` -> upper bound, `>x` -> x, `v±e` -> v. `direction` = `gt` / `lt` /
`exact` from the text (`>x` is right-censored, `<x` left-censored; ranges are points at their upper bound).
Exactly **3 rows** cannot be recomputed in this scope (1 empty concentration, 2 numbers without a unit) and
are dropped (a regression guard in the stage stops the run if that number changes).
uM rows differ from DBAASP by a small MW offset (median +1.2%); report: `reports/dev_v2/mic_recompute_report.md`.

## 3. Labels (`src/soamp/data/dev_dataset.py`, thresholds `config/thresholds/dev_dbaasp_thresholds.csv`)
* Rows kept: `measure == MIC` **or** concentration `NA` (all NA rows have measure `-` or empty, so a MIC-only
  filter would never admit them): 25,422 MIC + 1,113 NA = 26,535 records.
* Per record, in ug/ml with DBAASP's cutoffs for every strain: **active < 25, inactive > 100** (strict; exactly 25
  or 100 and the 25-100 band are uncertain). One-sided intervals for `>x` / `<x`: `>8` cannot be called active.
  NA records (1,111,111) are inactive.
* Per (peptide, strain) pair: agreeing decisive labels keep that label; uncertain records are ignored; disagreeing
  decisive labels drop the pair (285 pairs, 34 of them containing an NA row; list in
  `reports/dev_v2/label_conflicts.csv`).
* Result: **16,189 labelled pairs = 10,112 active + 6,077 inactive (62.5% active), 8,478 peptides, 21.4% non-linear.**
  Per strain (active/inactive): E. coli 3,433/1,910; S. aureus 2,165/1,310; P. aeruginosa 1,937/1,252;
  A. baumannii 945/434; E. faecalis 676/560; K. pneumoniae 418/417; B. subtilis 538/194.
* `mic_value_uM` keeps its legacy name but holds ug/ml (geometric mean over the non-NA records; 1,111,111 for
  NA-only pairs). `mic_type` is `na` for NA-only pairs (nothing branches on it).

### The NA assumption
NA means "no activity measured / not active up to X". Labelling all 1,038 NA-only pairs (689 peptides) inactive is an
assumption: notes with a parseable limit say not active up to > 100 ug/ml in 432 rows, 25-100 in 121 and <= 25 in
14 rows (546 rows state no limit). The audit file `reports/dev_v2/na_only_pairs.csv` carries each pair's note and
parsed limit so they can be filtered out later. The NA rows move the active share from 66.9% to 62.5%.
32 pairs with an active MIC record lose it because another record of the same pair is an NA row.

## 4. Split (`src/soamp/splitting/dev_final_split.py`)
Test + 5 CV folds (val = the held-out fold, as everywhere in the repo). The clusters are the ones built on all 11,088
peptides of the seven strains (`data/dev/dbaasp_dev_split_v2.parquet`: alignment identity >= 0.8 with D-forms mapped to
L for linear peptides, Murcko ring systems for non-linear ones; `docs/splitting_methodology.md`). Only the assignment
of whole clusters to buckets is redone, with the repo's stratified swap search (`stratified_cluster_split`,
100,000 iterations, seed 42), weighted by **labelled pairs** and balancing bucket size, non-linear share (lambda 30)
and active share (lambda 30). 2,610 clustered peptides without a labelled pair carry no weight and are not in the
dataset; they stay in the clustering so homology that passes through them still ties neighbours together.

| bucket | peptides | pairs | % of pairs (target) | non-linear % | active % |
|---|---|---|---|---|---|
| test | 1,681 | 3,239 | 20.01 (20) | 21.43 | 62.46 |
| fold_0..4 | 1,301-1,414 | 2,589-2,592 | 15.99-16.01 (16) | 21.40-21.43 | 62.46-62.47 |

Leakage (measured on the saved split, `scripts/splitting/measure_dev_split.py`, 4,223 linear sequence pairs >= 0.9):
0 clusters span two buckets, 0 identical SMILES, 0 identical token sequences, **0 of 4,223 pairs >= 0.9 across buckets**.
Pairs between 0.6 and 0.9 are not guaranteed together: 12 of 11,354 (0.8-0.9), 2,431 of 11,579 (0.7-0.8, 21%),
10,830 of 20,571 (0.6-0.7, 53%); 27.8% of all pairs >= 0.6 cross buckets.

## 5. Limitations
* The active share is balanced overall, not per strain: the small strains vary a lot between buckets (K. pneumoniae
  37.6-65.2% active, B. subtilis 62.9-86.3%, E. faecalis 44.9-59.4%; E. coli 62.5-66.0%).
* NA -> inactive is an assumption (section 3); 32 active pairs are lost to NA conflicts.
* Range concentrations are points at their upper bound (DBAASP's convention), not intervals.
* Moderate similarity (identity 0.6-0.8) can cross buckets (section 4).
* Not comparable with the earlier dev CV runs (three strains, DBAASP `activity`, no NA rows, other clusters and buckets).

## 6. Features and the W&B cache (phone experiments)
Features are built locally once and shipped with the dataset, so rented instances never rebuild them
(`config/features/dev_v2_*.yaml`, `pipeline/features/01-03`; scalers are fit on the `train` split only):

| file in `data/dev_v2/` | content | size |
|---|---|---|
| `peptide_features_rdkit.csv` + `peptide_feature_scaler_rdkit.json` | 13 RDKit descriptors, 8,478 peptides | 1.0 MB |
| `peptide_features_peptideclm.csv` + `peptide_feature_scaler_peptideclm.json` | frozen PeptideCLM-23M embedding, `dim_0..dim_767`, no NaN | 130 MB |
| `peptide_feature_scaler_molgraph.json` | stub only (graphs are built from the SMILES at train time) | tiny |
| `organism_vocab_vocab_embedding.json` | 7 strains + OOV index 0 | tiny |
| `mic_classification_dataset.csv`, `peptide_split.csv` | the dataset and the split (section 4) | 7 MB |

`scripts/exp/publish_data.py --dataset dev_v2` uploads exactly these 8 files (138 MB) as the W&B artifact
`<entity>/soamp-ops/soamp_dev_v2_data:latest`, with metadata: per-file size + sha256, counts, strains, the git sha.
`scripts/exp/fetch_data.py <ref> [dir]` restores them and verifies every file against those checksums (exit 1 on a
mismatch). The registry entry is `DATASETS["dev_v2"]` in `src/soamp/runs/request.py`; it is now the default dataset
of a request (`dataset: dev` = the older three-strain data); cells inherit `config/train/cv_dev_v2_base.yaml`.
The code bundle an instance downloads is built from `git ls-files`: everything new must be committed and pushed first.

## 7. XGBoost (`xgboost_classifier`)
Second tree architecture next to scikit-learn's `hist_gradient_boosting_classifier`; same featurisation, class
weighting, weighted BCE, W&B schema and per-round curves (one boosting round = one epoch). Hyperparameters
`n_estimators, learning_rate, max_depth, min_child_weight, reg_lambda, reg_alpha, gamma, subsample, colsample_bytree`.
It is fitted and scored in an isolated worker process (`src/soamp/engine/xgb_worker.py`): on macOS XGBoost's OpenMP
runtime (Homebrew libomp) collides with torch's and scikit-learn's in one process (segmentation fault with torch loaded
first, a hang with all three; reproduced with xgboost 3.4.1), and the worker keeps the training process free of it on
every platform. Per-round margins are built incrementally in the worker and checked against a full prediction.
Local smoke (`config/requests/dev_v2_smoke.yaml`, RDKit x vocab, 5-fold CV, W&B offline), mean over folds, last round:

| cell | val AUROC | val MCC | val F1 | fit AUROC |
|---|---|---|---|---|
| MLP, 5 epochs | 0.661 | 0.229 | 0.634 | 0.686 |
| scikit-learn boosting, 60 rounds, 15 leaves | 0.728 | 0.307 | 0.698 | 0.872 |
| XGBoost, 60 rounds, depth 4 | 0.714 | 0.288 | 0.689 | 0.843 |

These are smoke settings, not tuned models. The first XGBoost run on a rented Linux instance has not happened yet.
