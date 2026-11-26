---
name: run-experiment
description: Launch soamp model experiments (CV runs on vast.ai, logged to W&B) from a natural-language request. Use whenever the user asks to run, launch, sweep or try experiments/architectures/hyperparameters/representations, e.g. from the phone.
---

# Run an experiment

All limits (cost, instance count, cells) are enforced **in code** by `scripts/exp/launch.py`; never try to
work around a refusal. You only translate the request into a request file and call the launcher.

## Steps

1. In the cloned repo run `pip install -q -e . --no-deps` (idempotent, ~2 s; the cloud setup script does not always run it in the repo).
2. Write `config/requests/<experiment>.yaml` from what the user asked. Rules:
   - `experiment`: a NEW lowercase slug (a-z, 0-9, `_`, `-`; 3-49 chars), e.g. `repr_sweep_v2`. It becomes the W&B group; never reuse a name unless the user wants a re-run (re-runs skip finished cells).
   - `dataset`: leave it out - the default is `dev_v2`, the current dataset (seven strains, MICs recomputed from concentration, NA rows inactive, 16,189 labelled pairs, test + 5 folds; `docs/dev_dataset_v2.md`). `dataset: dev` is the OLDER three-strain data, only if the user asks for it. `dataset: dev_v2_linear` is dev_v2 filtered to the linear peptides (12,717 pairs, same labels and the same test + 5-fold assignment, so a cell is directly comparable to the dev_v2 cell with the same settings; `docs/dev_dataset_v2_linear.md`). For both: peptide representations `rdkit`, `peptideclm`, `molgraph`; organism: only `vocab` exists (others are rejected with a message).
   - A `dev_v2` request has no comparison baseline: the default baseline run (`dev_sweep_small_wd1e-3`) belongs to `dev` and is not comparable, so do not set `baseline_run` unless the user names a dev_v2 run.
   - Cross-validation is the default and the only supported mode: do not write `cv:` unless asked; never `cv: false`.
   - `seeds: [42]` unless asked; `gpu_name` defaults to `A100_SXM4` (the class that has worked): only change it if the user asks for another GPU.
   - Architectures and their OWN hyperparameters (a key from another architecture is an error):
     - `baseline_classifier`: `learning_rate`, `weight_decay`, `batch_size`, `organism_embed_dim`, `hidden_dims`; plus `epochs`.
     - `attention_fusion_classifier`: `learning_rate`, `weight_decay`, `batch_size`, `projection_dim`, `num_attention_heads`, `num_attention_layers`, `hidden_dims`; plus `epochs`.
     - `hist_gradient_boosting_classifier` (tree model): `max_iter`, `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`, `max_features` - no `epochs`, not usable with `molgraph`.
     - `xgboost_classifier` (tree model, XGBoost on CPU): `n_estimators` (boosting rounds), `learning_rate`, `max_depth`, `min_child_weight`, `reg_lambda`, `reg_alpha`, `gamma`, `subsample`, `colsample_bytree` - no `epochs` (rounds are set by `n_estimators`), not usable with `molgraph`. Its keys are NOT the scikit-learn ones (`max_iter`, `max_leaf_nodes`, ...); mixing them is an error.
   - A list of values is a grid axis inside ONE architecture. `hidden_dims`: a flat list is one value, a list of lists is an axis.
   - Put architecture settings under that architecture's `hparams:`; `epochs` goes next to `name:`.
   - Example: see `config/requests/example_repr_v1.yaml`.
3. Run `python scripts/exp/launch.py config/requests/<experiment>.yaml --plan` and show the user the output
   (cells, estimated GPU-hours and cost). Exit 2 = invalid request: fix the file and retry; exit 1 = refused by a cap.
4. Only if the user asked to run it (not just to plan), run the same command with `--go` instead of `--plan`.
   Relay what it printed verbatim, including a refusal. Do not edit `config/runs/*.yaml`, do not call `vastai`
   to create instances yourself, and never retry a refused launch with different caps.
5. Tell the user: results appear in W&B project `soamp` (group = experiment name); the hourly watcher
   analyses it when every cell is finished and sends the deck; first results usually take ~15-30 min.

## Notes
- `--go` re-checks the caps against live vast.ai instances and the 24 h spend ledger, and launches only cells that have no finished W&B run, so repeating a request is safe.
- To re-launch the unfinished cells of an experiment (the watcher prints `RETRY <experiment>`): `python scripts/exp/launch.py --retry <experiment>`.
- Check what is running (read-only): `python scripts/exp/watch.py --dry-run`.
