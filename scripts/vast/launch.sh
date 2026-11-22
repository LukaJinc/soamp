#!/usr/bin/env bash
# Runs on the LAPTOP: rent an instance, ship the working tree, start the job
# detached, print the instance id. After it returns the laptop can be turned off.
#
# usage: scripts/vast/launch.sh <offer_id> <spec.yaml> [existing_instance_id]
#   (3rd arg resumes at the rsync step on an already-created instance)
#   find an offer first:  vastai search offers 'gpu_name=RTX_4090 reliability>0.98 disk_space>=50 rentable=true' -o dph
#
# SKELETON: nothing here has been run against vast.ai. Verify each step the
# first time (esp. ssh-url parsing and create-instance output format).
set -euo pipefail

OFFER="${1:?offer id}"; SPEC="${2:?run spec yaml (repo-relative)}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"; VASTAI="$ROOT/.venv/bin/vastai"

# Instance params come from the spec (single source of truth), not flags.
read -r IMAGE DISK MAX_HOURS GRACE < <("$PY" - "$SPEC" <<'PYEOF'
import sys
from soamp.runs.spec import RunSpec
from soamp.utils.config import load_config
i = load_config(sys.argv[1], RunSpec).instance
print(i.image, i.disk_gb, i.max_hours, i.grace_seconds)
PYEOF
)

# WANDB_API_KEY from local .env (gitignored); never committed.
set -a; source "$ROOT/.env"; set +a
: "${WANDB_API_KEY:?set WANDB_API_KEY in .env}"

if [ -n "${3:-}" ]; then
  ID="$3"
else
  # .git is not shipped: record the laptop's commit (+ -dirty) for the run's metadata.
SOAMP_GIT_SHA="$(git rev-parse HEAD)$(git diff --quiet HEAD 2>/dev/null || echo -dirty)"

echo "creating instance from offer $OFFER ($IMAGE, ${DISK}GB)"
  OUT=$("$VASTAI" create instance "$OFFER" --image "$IMAGE" --disk "$DISK" \
        --ssh --direct --env "-e WANDB_API_KEY=$WANDB_API_KEY -e SOAMP_GIT_SHA=$SOAMP_GIT_SHA" --raw)
  ID=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["new_contract"])' "$OUT")
fi
echo "instance id: $ID   (manual kill: vastai destroy instance $ID)"

echo "waiting for instance to run..."
until [ "$("$VASTAI" show instance "$ID" --raw | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("actual_status"))')" = "running" ]; do
  sleep 10
done

URL=$("$VASTAI" ssh-url "$ID")                     # ssh://root@HOST:PORT
HOSTPORT=${URL#ssh://root@}; HOST=${HOSTPORT%:*}; PORT=${HOSTPORT##*:}
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new root@$HOST"

# "running" can precede sshd accepting connections (found on the first A100 launch).
echo "waiting for ssh..."
for _ in $(seq 1 60); do
  $SSH -n -o ConnectTimeout=5 true >/dev/null 2>&1 && break
  sleep 10
done
$SSH -n -o ConnectTimeout=5 true >/dev/null 2>&1 || { echo "ssh never came up; instance $ID still exists" >&2; exit 1; }

# Ship the working tree (uncommitted changes included, no GitHub needed).
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new" \
  --exclude .venv --exclude .git --exclude wandb --exclude artifacts \
  --exclude .cache --exclude '__pycache__' --exclude 'data/dbaasp_dev_*' \
  --exclude data/clustering --exclude reports/slide_figs \
  --exclude 'reports/cv_*' --exclude 'reports/train_cv_*' --exclude reports/vast_* \
  --exclude .env "$ROOT/" "root@$HOST:/workspace/soamp/"

# Start detached (setsid + all fds redirected, ssh -n) so ssh returns immediately and
# the job survives the session. Secrets are not put on the command line: remote_main.sh
# reads WANDB_API_KEY / SOAMP_GIT_SHA from the container env (PID 1) set at create time.
$SSH -n "cd /workspace/soamp && setsid nohup bash scripts/vast/remote_main.sh '$SPEC' '$MAX_HOURS' '$GRACE' </dev/null >/dev/null 2>&1 &"

echo "started. laptop can be turned off. monitor: wandb, or: vastai logs $ID"
echo "afterwards confirm nothing is billing: vastai show instances"
