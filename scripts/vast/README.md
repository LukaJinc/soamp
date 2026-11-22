# Running CV experiments on vast.ai

Why: use a bigger GPU, avoid Colab disconnects, and be able to turn the laptop
off. The instance destroys itself when the job ends (success, failure, timeout).

> Skeleton: `launch.sh` / `remote_main.sh` have not been run against vast.ai yet.
> Do a one-cell smoke test first and verify `CONTAINER_ID`/`CONTAINER_API_KEY`.

1. Write a run spec (copy `config/vast/example_grid.yaml`): which `train_cv`
   configs to run, and the GPU/disk/`max_hours` wanted.
2. Preview: `python scripts/vast/run_experiments.py --spec <spec> --dry-run`
3. Pick an offer, **most reliable first, not cheapest** (a dropped host loses the in-flight cell; there is no resume):
   `python scripts/vast/find_offers.py --spec <spec>` (verified, on-demand hosts; filters come from the spec's `instance:` block). Re-run at launch time.
4. `scripts/vast/launch.sh <offer_id> <spec>` (laptop on until it prints "started").
5. Turn the laptop off. Watch wandb (project `soamp`) or `vastai logs <id>`.
6. Check `vastai show instances` is empty. Manual kill: `vastai destroy instance <id>`.

Notes: a bigger GPU mainly speeds PeptideCLM/GIN cells; the tiny-MLP cells gain
little. There is no resume -- a rerun skips cells whose `reports/cv_results_<exp_id>.csv`
exists. Use on-demand, not interruptible, instances. The Colab notebooks in
`scripts/colab/` still work as an alternative.

## GPU check (read the log!)

`reports/vast_remote.log` on the instance must show `installed torch 2.13.0 from the cu126 index` and
`torch 2.13.0+cu126 cuda 12.6 NVIDIA A100...`, and the training log `device: cuda`. History: every run
before 2026-11-13 trained on the CPU because the default PyPI torch wheel (cu130) cannot run on the
hosts' CUDA-12.x drivers and PyTorch fell back silently. `remote_main.sh` now installs the CUDA wheel for
the driver from the PyTorch index *only* (an extra PyPI index wins in uv and brings back cu130) and aborts
with rc=11 if a GPU is present but unusable. Quick offline check of an index:
`echo torch==2.13.0 | uv pip compile - --python-platform x86_64-manylinux_2_28 --python-version 3.12 --index-url https://download.pytorch.org/whl/cu126`.
