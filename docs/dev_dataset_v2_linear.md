# Dev dataset v2, linear peptides only (`dev_v2_linear`)

A linear-peptides-only copy of `dev_v2`, built to answer one question: does a model trained on
linear peptides alone do better than one trained on the full mixture of linear, disulfide-bridged
and cyclized peptides? It is a **row filter on the already built `dev_v2`** — no record is
relabelled, no cluster search is re-run, and every surviving (peptide, strain) pair keeps the
split and fold it has in `dev_v2`, so a `dev_v2_linear` cell is comparable to the `dev_v2` cell
with the same architecture, representation and seed apart from the rows that are gone.

```
python pipeline/data/05_build_linear_subset.py --config config/data/dev_v2_linear.yaml
python pipeline/features/03_fit_peptide_scaler.py --config config/features/dev_v2_linear_peptide_rdkit.yaml
python pipeline/features/03_fit_peptide_scaler.py --config config/features/dev_v2_linear_peptide_peptideclm.yaml
python pipeline/features/03_fit_peptide_scaler.py --config config/features/dev_v2_linear_peptide_molgraph.yaml
python scripts/exp/publish_data.py --dataset dev_v2_linear      # ship it to the instances
```

Everything lands in `data/dev_v2_linear/` and `reports/dev_v2_linear/`; `data/dev_v2/` is
untouched, so the earlier results stay reproducible.

## 1. The linearity rule

A peptide is kept only when **both** sources call it linear:

1. **DBAASP records no intrachain bond.** That flag already travels with the built dataset as
   `has_noncanonical` (`soamp.data.dev_dataset` writes `not is_linear`, and `is_linear` is
   `not peptide["intrachainBonds"]` from the DBAASP record), so no DBAASP pull is needed.
2. **The SMILES has no macrocycle and no disulfide**: no ring of 9+ atoms and no `S-S`
   (`[#16X2][#16X2]`). The ring size is the point of the test — every amino-acid side-chain ring
   (Trp, His, Phe, Tyr, Pro) is 5- or 6-membered, so a 9+ ring can only be a cyclized backbone.

The two sources agree on 8,449 of the 8,478 `dev_v2` peptides (99.66%). Requiring both drops the
29 that disagree: 4 peptides DBAASP calls linear whose generated SMILES carries a macrocycle or an
S-S bond, and 25 DBAASP records a bond for although the SMILES looks linear (the SMILES generator
falls back to the plain backbone when it cannot place the bond — `soamp.curation.smiles_gen`).
Dropping them means every peptide kept is linear both in the database annotation and in the
structure the model actually sees.

The flag `is_linear` is the only linearity source in the repo, and it is what rule 1 uses -- the
SMILES test is a cross-check on top of it, not a replacement. It changes the kept set by 4
peptides and 5 pairs out of 12,722: peptides 17214 and 17215 (a tert-butyl disulfide protecting
group on a cysteine) and 20063 and 20064 (a lipoic-acid cap, whose 1,2-dithiolane is a ring with
an S-S). Their sulfur is in a side chain or a cap rather than a backbone bridge, so DBAASP's flag
is arguably right about them and the conservative rule costs 5 pairs to be sure. Pass a dataset
built from rule 1 alone if you would rather keep them: the difference is far below fold noise.

Audit files: `reports/dev_v2_linear/peptide_linearity.csv` (one row per peptide: both flags and
the drop reason) and `reports/dev_v2_linear/linear_subset_per_strain.csv`.

## 2. What is left

**6,665 of 8,478 peptides (78.6%) and 12,717 of 16,189 labelled pairs (78.6%): 7,623 active +
5,094 inactive (59.9% active, down from 62.5%).** Peptides dropped: 1,784 with an intrachain bond
that the SMILES confirms, 25 with a bond but a linear SMILES, 4 linear in DBAASP with a cyclic
SMILES.

Split: 10,173 train + 2,544 test pairs (5,364 train + 1,301 test peptides), folds 0-4 holding
1,132 / 1,085 / 1,062 / 1,054 / 1,031 peptides. The 20% test share and the fold balance survive
the filter because non-linear share was one of the three quantities the `dev_v2` bucket search
balanced (`docs/dev_dataset_v2.md` §4).

| strain | pairs before | pairs kept | dropped | dropped % | active % before | active % kept |
|---|---|---|---|---|---|---|
| E. coli ATCC 25922 | 5,343 | 4,183 | 1,160 | 21.7 | 64.3 | 63.7 |
| S. aureus ATCC 25923 | 3,475 | 2,871 | 604 | 17.4 | 62.3 | 60.7 |
| P. aeruginosa ATCC 27853 | 3,189 | 2,470 | 719 | 22.5 | 60.7 | 57.0 |
| A. baumannii ATCC 19606 | 1,379 | 1,110 | 269 | 19.5 | 68.5 | 66.4 |
| E. faecalis ATCC 29212 | 1,236 | 943 | 293 | 23.7 | 54.7 | 47.9 |
| K. pneumoniae ATCC 700603 | 835 | 651 | 184 | 22.0 | 50.1 | 41.9 |
| B. subtilis ATCC 6633 | 732 | 489 | 243 | 33.2 | 73.5 | 69.9 |

The strains lose between 17% and 33% of their pairs, so no strain disappears, but the two smallest
(B. subtilis, K. pneumoniae) end up with 489 and 651 pairs — read their per-strain metrics with
that in mind. The filter also shifts the class balance: non-linear peptides were more often active
(K. pneumoniae 50.1% → 41.9% active, E. faecalis 54.7% → 47.9%), which is part of what any
difference in results will reflect.

## 3. Files and how a run picks it up

`data/dev_v2_linear/` holds the same file set as `data/dev_v2/`: `mic_classification_dataset.csv`,
`peptide_split.csv`, `peptide_features_rdkit.csv`, `peptide_features_peptideclm.csv` (both
filtered row-wise — the per-peptide vectors themselves are unchanged), the organism vocab
(identical, all seven strains survive) and the three peptide feature scalers, **refit on this
dataset's train rows** so no `dev_v2` statistics leak in. The molecular graph needs no feature
file: it is a fixed function of the SMILES.

The launcher knows the dataset as `dev_v2_linear` (`soamp.runs.request.DATASETS`), with
`config/train/cv_dev_v2_linear_base.yaml` as the base config every generated cell inherits, so a
request file only needs:

```yaml
dataset: dev_v2_linear
```

Peptide representations `rdkit`, `peptideclm`, `molgraph`; organism `vocab`, as for `dev_v2`.
