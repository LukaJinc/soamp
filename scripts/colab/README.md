# Running soamp on Google Colab

Four notebooks:

| Notebook | What it does |
|---|---|
| `01_smoke_overfit.ipynb` | Builds every (peptide × organism) featurization in the 3×3 grid (the `dnabert_s_16s` ones only once `data/organism_16s_dnabert_s.json` is committed), checks each produces the dimensions it claims and forwards through `attention_fusion_classifier`, then overfits ~256 rows per cell and asserts the loss reaches zero. Gate for notebooks 02 and 03. |
| `04_embed_organisms.ipynb` | **One-off.** Embeds each organism's 16S rRNA gene with frozen DNABERT-S on a Colab GPU (needs `transformers<5`, installed in that runtime only) and downloads the small table `data/organism_16s_dnabert_s.json` to commit. Only re-run when `config/organism_genomes/organism_genome_accessions.csv` gains organisms. |
| `02_run_experiments.ipynb` | Builds the PeptideCLM feature artifact on the GPU (cached to Drive), then runs the four `config/train/exp_*.yaml` cells through `pipeline/train.py` (single held-out train/val/test split) and collects results from wandb. |
| `03_run_cv_experiments.ipynb` | CV counterpart to 02: builds/reuses the same PeptideCLM feature artifact, then runs the `config/train/cv_*.yaml` cells you select (any subset of the 3×3 grid) through `pipeline/train_cv.py` (5-fold CV over the train peptides of the committed `data/peptide_split.csv` — no checkpoints, `test` never touched; per-epoch fit/val loss curves are logged to wandb and plotted in the notebook) and collects results from wandb. |

In `03_run_cv_experiments.ipynb`, **section 0 is the only cell to edit**: set `PEPTIDE` / `ORGANISM` (cross product), `EXCLUDE`, or `ONLY` to pick which `config/train/cv_<peptide>_<organism>.yaml` cells run; the genome download, PeptideCLM featurization, artifact checks, comparison, plots and Drive copy all adapt to the selection.

Run 01 first as a gate; 02 and 03 are independent of each other (each rebuilds/reuses its own PeptideCLM cache) and can run in either order.

A third path exists: driving Google's `google-colab-cli` (`colab new`/`colab
exec`/`colab download`/...) directly from a local terminal instead of either
notebook — no browser tab needed at all. It hits several real problems
(a packaging bug in the CLI itself, an SSL cert issue, a kernel-state gotcha
around editable installs, transient connection drops mid-run) — see
[`CLI_TROUBLESHOOTING.md`](CLI_TROUBLESHOOTING.md) for all of them and their
fixes before attempting this route.

## Colab Secrets

Set these under the key icon in the left sidebar, with notebook access enabled:

- `GITHUB_TOKEN` — a fine-grained, **read-only** PAT for this private repo.
  All three notebooks clone with the token inline, scrub it from the printed
  output, then rewrite the remote to the plain URL so it never lands in
  `.git/config`.
- `WANDB_API_KEY` — required by notebooks 02 and 03; notebook 01 never opens a tracker.
- `WANDB_ENTITY` — optional, only if your runs shouldn't go to your default entity.

## Experiment grid

`attention_fusion_classifier` is held fixed across all cells so the
comparison isolates the representation. `BaselineClassifier` projects only the
organism side and feeds peptide features in raw, so a 768-dim PeptideCLM vector
concatenated with an 8-dim organism embedding would be ~99% peptide — not a
comparison worth running against the 13-dim RDKit cell.

| `exp_id` | peptide | organism |
|---|---|---|
| `rdkit_vocab_attnfusion` | `rdkit_descriptors` (13d) | `vocab_embedding` (index) |
| `rdkit_kmer_attnfusion` | `rdkit_descriptors` (13d) | `kmer_composition` (340d vector) |
| `peptideclm_vocab_attnfusion` | `peptideclm_embedding` (768d) | `vocab_embedding` (index) |
| `peptideclm_kmer_attnfusion` | `peptideclm_embedding` (768d) | `kmer_composition` (340d vector) |
| `molgraph_{vocab,kmer,dnabert}_attnfusion` | `molecular_graph` (small trainable GINE GNN, ~40k params, → 64d) | vocab / k-mer / DNABERT-S |
| `{rdkit,peptideclm}_dnabert_attnfusion` | RDKit / PeptideCLM | `dnabert_s_16s` (768d vector) |

The grid is 3×3: peptide {`rdkit_descriptors` 0D, `peptideclm_embedding` 1D, `molecular_graph` 2D} ×
organism {`vocab_embedding`, `kmer_composition`, `dnabert_s_16s`}. `molecular_graph` is the only
peptide method trained end to end (the GNN lives in the model; its hyperparameters are
`model.graph_encoder` in the train config). `dnabert_s_16s` is a frozen embedding of the
organism's 16S rRNA gene, read from a committed table (see notebook 04).

All cells land in wandb project `soamp`, group `featurization_grid_v1`.

The CV counterpart (`03_run_cv_experiments.ipynb`, `config/train/cv_*.yaml`,
`exp_id`s `<peptide>_<organism>_cv`) runs the same cells
through `pipeline/train_cv.py`'s 5-fold CV instead, logging to a separate
group, `featurization_grid_cv_v2` — cross-fold `{fit,val}_<metric>_{mean,std}`
summary keys, no `test_*` keys (CV never touches test) and no checkpoints
(`pipeline/train_cv.py` never invokes a `Checkpointer`). `_v2` marks the new folds from `peptide_split.csv`; `_v1` runs used the old QMAP-only folds and aren't comparable.

## Feature artifacts

Each cell reads method-suffixed filenames, so all of them coexist in `data/`:

| artifact | committed? | built by |
|---|---|---|
| `peptide_features_rdkit.csv` + `peptide_feature_scaler_rdkit.json` | yes (1.4 MB) | `features/01`+`03 --config config/features/peptide_rdkit.yaml` |
| `organism_vocab_vocab_embedding.json` | yes (184 B) | `features/02 --config config/features/organism_vocab.yaml` |
| `organism_vocab_kmer_composition.json` | yes (28 KB) | `features/02 --config config/features/organism_kmer.yaml` |
| `peptide_feature_scaler_molgraph.json` | yes (stub, 300 B) | `features/03 --config config/features/peptide_molgraph.yaml` (no feature CSV: graphs are built from SMILES) |
| `organism_16s_dnabert_s.json` → `organism_vocab_dnabert_s_16s.json` | yes (once built) | notebook `04_embed_organisms.ipynb`, then `features/02 --config config/features/organism_dnabert.yaml` |
| `peptide_features_peptideclm.csv` + its scaler | **no** (~190 MB) | `features/01`+`03 --config config/features/peptide_peptideclm.yaml` |

The PeptideCLM artifact exceeds GitHub's per-file limit, so it is gitignored and
rebuilt per environment — seconds on a GPU, ~5 min on CPU — then cached to
`MyDrive/soamp_cache/`.

The k-mer organism artifact bakes its genome vectors in, so `02`'s training run
needs neither the RefSeq FASTAs nor network access — it reconstructs the
featurizer straight from that committed artifact (`train.py`'s artifact mode,
`row_groups=None`). Notebooks 01 and 03 both fetch genomes anyway, because
both compute featurization fresh from rows (`row_groups=...`, via
`build_dataset` and `train_cv.py` respectively) rather than reading that
artifact — `train_cv.py` refits the organism featurizer per fold on purpose,
so it needs the raw FASTAs even though the same committed artifact exists.

## Is Colab actually the right place for this?

Mostly no, and it's worth knowing why before you spend a session on it.

The classifier is a small MLP over ~11k rows — **a GPU buys it essentially
nothing**. The entire GPU win is the PeptideCLM featurization pass, and that is
a one-off that produces a file. Measured on an M-series CPU: all four training
runs complete in seconds each, and the PeptideCLM pass takes ~5 minutes.

So the leanest workflow is often:

1. Run notebook 02's featurization step on Colab (or just locally, if you don't
   mind the 5 minutes).
2. Keep `data/peptide_features_peptideclm.csv`.
3. Run all four trainings locally:
   ```
   for cell in rdkit_vocab rdkit_kmer peptideclm_vocab peptideclm_kmer; do
       python pipeline/train.py --config config/train/exp_${cell}.yaml
   done
   ```
   And/or the CV counterpart (`data/peptide_split.csv` is committed, so
   no extra setup is needed):
   ```
   for cell in rdkit_vocab rdkit_kmer peptideclm_vocab peptideclm_kmer; do
       python pipeline/train_cv.py --config config/train/cv_${cell}.yaml
   done
   ```

Same configs, same entrypoints, same wandb grids — no session timeouts, no
clone, full git provenance. Use the notebooks when you want the GPU or a
clean-room environment; use the loops above otherwise.

## Local equivalent of notebook 01

```
python pipeline/features/00_fetch_organism_genomes.py   # once, ~13 MB
```
then the construction/overfit checks from the notebook, which use only
`build_dataset(row_groups=...)` / `build_model` / `Trainer` — no Colab-specific
code in those cells.
