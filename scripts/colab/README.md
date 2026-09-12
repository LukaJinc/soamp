# Running soamp on Google Colab

Two notebooks, run in order:

| Notebook | What it does |
|---|---|
| `01_smoke_overfit.ipynb` | Builds all four (peptide × organism) featurizations, checks each produces the dimensions it claims and forwards through `attention_fusion_classifier`, then overfits ~256 rows per cell and asserts the loss reaches zero. Gate for notebook 02. |
| `02_run_experiments.ipynb` | Builds the PeptideCLM feature artifact on the GPU (cached to Drive), then runs the four `config/train/exp_*.yaml` cells through `pipeline/train.py` and collects results from wandb. |

## Colab Secrets

Set these under the key icon in the left sidebar, with notebook access enabled:

- `GITHUB_TOKEN` — a fine-grained, **read-only** PAT for this private repo.
  Both notebooks clone with the token inline, scrub it from the printed output,
  then rewrite the remote to the plain URL so it never lands in `.git/config`.
- `WANDB_API_KEY` — required by notebook 02; notebook 01 never opens a tracker.
- `WANDB_ENTITY` — optional, only if your runs shouldn't go to your default entity.

## Experiment grid

`attention_fusion_classifier` is held fixed across all four cells so the
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

All four land in wandb project `soamp`, group `featurization_grid_v1`.

## Feature artifacts

Each cell reads method-suffixed filenames, so all four coexist in `data/`:

| artifact | committed? | built by |
|---|---|---|
| `peptide_features_rdkit.csv` + `peptide_feature_scaler_rdkit.json` | yes (1.4 MB) | `features/01`+`03 --config config/features/peptide_rdkit.yaml` |
| `organism_vocab_vocab_embedding.json` | yes (184 B) | `features/02 --config config/features/organism_vocab.yaml` |
| `organism_vocab_kmer_composition.json` | yes (28 KB) | `features/02 --config config/features/organism_kmer.yaml` |
| `peptide_features_peptideclm.csv` + its scaler | **no** (~190 MB) | `features/01`+`03 --config config/features/peptide_peptideclm.yaml` |

The PeptideCLM artifact exceeds GitHub's per-file limit, so it is gitignored and
rebuilt per environment — seconds on a GPU, ~5 min on CPU — then cached to
`MyDrive/soamp_cache/`.

The k-mer organism artifact bakes its genome vectors in, so training needs
neither the RefSeq FASTAs nor network access. Only notebook 01 fetches genomes,
because it computes featurization fresh from rows rather than from that artifact.

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

Same configs, same entrypoint, same wandb grid — no session timeouts, no clone,
full git provenance. Use the notebooks when you want the GPU or a clean-room
environment; use the loop above otherwise.

## Local equivalent of notebook 01

```
python pipeline/features/00_fetch_organism_genomes.py   # once, ~13 MB
```
then the construction/overfit checks from the notebook, which use only
`build_dataset(row_groups=...)` / `build_model` / `Trainer` — no Colab-specific
code in those cells.
