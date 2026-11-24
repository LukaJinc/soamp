# Dev split: sequence clustering methodology (v2)

How `data/dev/dbaasp_dev_split_v2.parquet` is built. Code: `src/soamp/splitting/`
(`tokens.py`, `alignment.py`, `dev_config.py`, `bucketing.py`), config:
`config/splitting/dev.yaml` (baseline) and `dev_v2.yaml` (current), entrypoint:
`scripts/EDA/09_data_processing.ipynb` (sections "Clustering" onward). Numbers below
come from `scripts/splitting/measure_dev_split.py` on the saved splits.

## Scope
Seven target strains (E. coli ATCC 25922, S. aureus ATCC 25923, P. aeruginosa ATCC 27853,
A. baumannii ATCC 19606, K. pneumoniae ATCC 700603, E. faecalis ATCC 29212,
B. subtilis ATCC 6633): **11,088 peptides, 30,568 activity records**, of which 8,743 linear
and 2,345 non-linear. (The first dev split, `dbaasp_dev_split.parquet`, covered only the
first three strains; it is kept unchanged and still reproduces exactly with the refactored code.)

## 1. Tokens (linear peptides)
DBAASP writes every unusual residue as `X` and names it in `unusualAminoAcids[]`. Each
linear peptide becomes a list of tokens: the canonical letter, or `mod:<name>`.

* **D to L before alignment (`d_residues: map_to_L`).**
  * Canonical residues: DBAASP writes D-forms in lower case; tokens are upper-cased, so
    `k` and `K` are the same token (this was always the case).
  * Unusual residues: a leading `D-` (case-insensitive) is stripped from the name, so
    `D-Orn` and `Orn` are one token. Five names are mapped by an explicit table because the
    marker is not a prefix or the case collides: `4-NH2-D-Pro`, `cis-4-D-Hyp`,
    `C10-N6-D-Lys`, `β-Thi-D-Ala`, and `D-TIC` -> `Tic`.
  * Nothing else is touched: names are never case-folded, `(S,S)-`/`(R,R)-ACPC`,
    `cis-`/`trans-` and `Allo-` forms are different residues, and names that merely start
    with the letter D (`DAB`, `DAP`, `Dab(Cys)`) are left alone. No DBAASP name uses `L-`.
  * Why: a peptide and its D-substituted variant (or enantiomer) is the same design, and
    splitting the pair across train and test is leakage. Stereochemistry stays available to
    the model through the SMILES.
  * Default in code is `separate` (D and L names kept apart), which reproduces the first split.
* **Unusual residues beyond that: one symbol per distinct name** (241 names before mapping,
  225 after, 4,457 occurrences). No parent-residue table: 342 names exist in the full dev
  corpus, the top 20 cover 60% of occurrences and only 950 of 2,735 peptides with an
  unusual residue, so a curated similarity table would be long and subjective.

## 2. Alignment and identity
Own aligner (`alignment.py`), because QMAP's engine merges every unusual character into one
"unknown". Global (Needleman-Wunsch) alignment, BLOSUM45 for canonical pairs, gap open 5 /
extend 1. An unusual residue scores the mean BLOSUM45 diagonal (+5.3) against itself and the
BLOSUM45 `X` score (flat) against anything else. **Identity = matches / alignment length**
(QMAP's convention; the shorter-sequence denominator was considered and not adopted).
It approximates QMAP (65% of pairs identical, correlation 0.985 on canonical-only peptides).

## 3. Graph, hard links, Leiden
Identical token sequences are one node. Edge = identity >= **0.8** (v1: 0.6). Pairs >= **0.9**
are merged into one node first so Leiden can never separate near-duplicates (all original
edges kept, as parallel edges and self-loops). Leiden, seed 42, run to convergence.

## 4. Non-linear peptides, joint split
Murcko ring-system key of the SMILES (stereo ignored: `isomericSmiles=False`). All clusters
(alignment + Murcko) are then assigned whole to the test set (20% of records) or one of five
folds (16% each) by a seeded swap search (100,000 iterations, seed 42) balancing bucket size
and non-linear share. No activity labels or thresholds are used.

## 5. What changed v1 -> v2 (same seven strains)
| | v1: 0.6, `separate` | 0.8, `separate` | v2: 0.8, `map_to_L` |
|---|---|---|---|
| distinct token sequences | 7,571 | 7,571 | 7,561 |
| distinct unusual residues | 241 | 241 | 225 |
| alignment clusters | 1,538 | 3,308 | 3,301 |
| largest alignment cluster | 747 peptides (8.8% of records) | 225 (1.7%) | 225 (1.7%) |
| joint clusters (with Murcko 824) | 2,362 | 4,132 | 4,125 |
| sequence pairs >= 0.6 in different buckets | 3,056 / 70,399 (4.3%) | 19,125 (27.2%) | 19,304 / 70,253 (27.5%) |
| pairs 0.7-0.8 in different buckets | 1.8% | 18.9% | 18.5% |
| pairs 0.6-0.7 in different buckets | 8.7% | 51.4% | 52.3% |

* **D to L mapping is a small effect here**: 16 residue names merge, 10 token sequences merge,
  7 fewer clusters. It matters for correctness (a D-variant can no longer evade the aligner),
  not for the size of the clusters.
* **The threshold is the large effect.** At 0.8 the 0.6-0.8 band no longer links peptides, so
  about 27% of the pairs with identity >= 0.6 end up in different buckets (4.3% at 0.6),
  concentrated at 0.6-0.7. The price of a higher threshold is leakage between moderately
  similar peptides; the gain is much smaller clusters (largest 1.7% of records instead of 8.8%)
  and an easier balance. This was a deliberate choice, not a free improvement.
* Unchanged and verified on v2: 0 clusters span two buckets, 0 identical SMILES and 0
  identical token sequences in more than one bucket, 0 of 5,928 pairs >= 0.9 across buckets;
  every bucket within 0.01 points of its target size; non-linear share 18.81% in every bucket.

### Threshold sweep (seven strains, D-to-L tokens, hard links 0.9, Leiden seed 42)
| identity threshold | clusters | largest cluster (peptides) | its share of records |
|---|---|---|---|
| 0.5 | 788 | 1,191 | 12.9% |
| 0.6 | 1,538 | 635 | 7.8% |
| 0.7 | 2,333 | 401 | 6.1% |
| **0.8** | **3,301** | **225** | **1.7%** |
| 0.9 | 4,982 | 133 | 1.9% |

A fold holds 16% of the records, so a cluster must stay well below that; 0.6 also fits, so
cluster size alone does not force 0.8. At 0.4 one cluster already holds about 22% of all records.
Connected components of the identity graph (no clustering): largest component is 53.6% / 25.2% /
6.0% / 1.9% of records at 0.6 / 0.7 / 0.8 / 0.9. At 0.8 plain components therefore fit in a fold and
would cut no pair >= 0.8, where Leiden cuts 13 pairs in the 0.8-0.9 band; Leiden is kept for
continuity with the first split (open simplification, not made).

## 6. Stereochemistry in the data (audit, `reports/stereo_audit/`)
For the 11,088 peptides: 9,501 all-L, 249 all-D, 1,314 mixed, 4 without a chiral residue, 20
undetermined (1,572 peptides, 14%, contain a D-residue). Of 188,508 residues, 6,329 are D
(5,564 canonical, 765 unusual in 55 distinct D-names). Notation and SMILES (read with
backbone-role chirality, not CIP labels) agree on 174,000 of the 176,000 residues that could be
compared position by position; 3 real disagreements; 1,564 unusual residues the notation
calls L by default but whose alpha carbon is achiral in the SMILES (N-substituted glycines such
as NLys). Stereo-only twins (identical after D-to-L mapping, different stereo): 2,501 pairs among
956 peptides; groups of several variants inflate the pair count, so peptides-involved is the
number to quote.

## Limitations
* Our aligner approximates QMAP; identities are close, not identical.
* Unusual residues match only by exact (D-stripped) name; chemically close residues
  (Orn/Lys/Dab) count as different; different spellings of one monomer would too.
* R/S, cis/trans and Allo- descriptors are not treated as D/L.
* The 0.6-0.8 identity band is deliberately not linked (see section 5).
* Non-linear peptides are grouped by ring systems; similarity across those groups is not measured.
* Earlier dev experiments used the three-strain, identity-0.6 split and are not comparable
  with results on v2.
