#!/usr/bin/env bash
# Runs ON the vast.ai instance, detached (nohup) so it survives the SSH session
# and the laptop being off. Guarantees the instance is destroyed however the
# job ends: success, failure, timeout, or this script being killed (EXIT trap).
#
# usage: remote_main.sh <spec.yaml> <max_hours> <grace_seconds>
#
# SKELETON: unverified on a real instance. First smoke test must confirm
# CONTAINER_ID / CONTAINER_API_KEY exist on the chosen image; fallback is
# passing VAST_API_KEY + VAST_INSTANCE_ID via `--env` at create time.
set -uo pipefail

SPEC="${1:?spec path}"; MAX_HOURS="${2:?max_hours}"; GRACE="${3:-120}"
cd "$(dirname "$0")/../.."
LOG=reports/vast_remote.log
mkdir -p reports

# SSH sessions don't inherit the container env; PID 1 does. Load the vast
# identity vars from there if missing (verified on a real instance).
for v in CONTAINER_ID CONTAINER_API_KEY WANDB_API_KEY SOAMP_GIT_SHA; do
  if [ -z "${!v:-}" ]; then
    val=$(tr '\0' '\n' </proc/1/environ | sed -n "s/^$v=//p" | head -1)
    [ -n "$val" ] && export "$v=$val"
  fi
done

destroy() {
  echo "[remote_main] destroying instance in ${GRACE}s (rc=${RC:-unset})" | tee -a "$LOG"
  sleep "$GRACE"   # grace period: let wandb/log flushes finish
  pip install -q vastai >>"$LOG" 2>&1
  # `yes |`: destroy asks "[y/N]" and aborts without a tty (found in the smoke test).
  yes | vastai destroy instance "${CONTAINER_ID:?CONTAINER_ID unset}" \
    --api-key "${CONTAINER_API_KEY:?CONTAINER_API_KEY unset}" >>"$LOG" 2>&1
}
trap destroy EXIT

# Image python is 3.11 but the repo needs >=3.12: build a 3.12 venv with uv.
# torch is pinned to the repo's version. The default PyPI wheel is built for the newest CUDA (13.0 at
# the time of writing), which the hosts' drivers (CUDA 12.x) cannot run: torch.cuda.is_available()
# is then False and training silently falls back to the CPU (found when the GPU showed 0% use while
# every earlier "A100" run had been CPU-only). So install the wheel for the newest CUDA that the
# driver supports, and refuse to continue if a GPU is present but torch cannot use it.
TORCH_VER=$(sed -n 's/.*"torch==\([^"]*\)".*/\1/p' pyproject.toml | head -1)

install_torch() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    uv pip install -q "torch==${TORCH_VER}"; return
  fi
  local drv
  drv=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)   # e.g. 128
  echo "[remote_main] driver supports CUDA ${drv}" | tee -a "$LOG"
  for c in 128 126 124 121; do
    [ "$c" -le "${drv:-0}" ] || continue
    # ONLY the PyTorch index (it mirrors torch's dependencies): with --extra-index-url PyPI, uv searches
    # the extra index first and installs PyPI's CUDA-13 build (verified: resolves 2.13.0+cu130).
    # cu128 has no torch 2.13.0, so the loop falls through to cu126, which the CUDA-12.8 driver runs.
    if uv pip install -q "torch==${TORCH_VER}" --index-url "https://download.pytorch.org/whl/cu${c}"; then
      echo "[remote_main] installed torch ${TORCH_VER} from the cu${c} index" | tee -a "$LOG"; return 0
    fi
  done
  return 1
}

{
  pip install -q uv &&
  uv venv --python 3.12 /workspace/venv &&
  source /workspace/venv/bin/activate &&
  install_torch &&
  uv pip install -q -r scripts/colab/requirements-colab.txt &&
  uv pip install -q -e . --no-deps
} >>"$LOG" 2>&1 || { RC=10; exit "$RC"; }

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >>"$LOG" 2>&1; then
    echo "[remote_main] GPU present but torch.cuda.is_available() is False; aborting instead of training on the CPU" | tee -a "$LOG"
    RC=11; exit "$RC"
  fi
  python -c "import torch; print('[remote_main] torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))" | tee -a "$LOG"
fi

# Hard cap on the whole job; timeout exits 124 -> EXIT trap still destroys.
timeout "${MAX_HOURS}h" python scripts/vast/run_experiments.py --spec "$SPEC" >>"$LOG" 2>&1
RC=$?
echo "[remote_main] runner rc=$RC" | tee -a "$LOG"
exit "$RC"
