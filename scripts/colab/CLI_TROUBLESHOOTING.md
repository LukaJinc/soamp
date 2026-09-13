# Driving `google-colab-cli` for this repo: problems hit and fixes

Notes from actually running the 2×2 featurization grid through Google's
`google-colab-cli` (the terminal tool that provisions real Colab runtimes —
see https://github.com/googlecolab/google-colab-cli), driven directly from a
local shell rather than through `01_smoke_overfit.ipynb`/
`02_run_experiments.ipynb`. Six distinct problems came up, in the order
they'd bite a fresh attempt. None of this touches the notebooks themselves —
they're a separate, working path (see `README.md`).

## 1. `pip install google-colab-cli` installs a broken, unrelated dependency

**Symptom:** every `colab exec`/`colab install` call crashes with
`AttributeError: module 'jupyter_kernel_client' has no attribute
'KernelClient'`.

**Cause:** the real `pyproject.toml` on GitHub pins
`jupyter-kernel-client` via `[tool.uv.sources]` to Google's own fork
(`github.com/googlecolab/jupyter-kernel-client`). That's a **`uv`-only**
mechanism — plain `pip` (whether from PyPI or `pip install
git+https://github.com/googlecolab/google-colab-cli.git`) has no concept of
`[tool.uv.sources]` and silently falls back to resolving `jupyter-kernel-client`
from PyPI, which is a *different, unrelated* package under the same name
(the real Jupyter org's client library) that exposes `JupyterKernelClient`,
not `KernelClient`.

**Fix:** install with `uv`, not `pip`, and from the GitHub source (the PyPI
release lagged a version behind and didn't yet have the `ColabKernelClient`
compatibility shim):
```
brew install uv   # or any other uv install method
uv tool install "git+https://github.com/googlecolab/google-colab-cli.git"
```
This installs `colab` into an isolated tool environment (`~/.local/bin/colab`
on macOS), separate from any project venv — appropriate, since this is a
general-purpose terminal tool, not a project dependency. Verify with
`colab --help` before doing anything else; if you see the same
`AttributeError` on first real command, the wrong `jupyter-kernel-client`
got resolved again.

## 2. `SSL: CERTIFICATE_VERIFY_FAILED` on `colab whoami` (and other calls)

**Symptom:**
```
[colab] whoami: tokeninfo request failed: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: unable to get local issuer certificate>
```

**Cause:** a generic macOS/python.org-installer issue — the interpreter's
default OpenSSL cert path doesn't point at a populated CA bundle. Unrelated
to `colab-cli` itself; affects any tool doing raw `urllib`/`ssl` calls under
this Python.

**Fix:** point `SSL_CERT_FILE` at `certifi`'s bundle for every `colab`
invocation:
```
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
```
Any project venv with `certifi` installed works as the source — it doesn't
need to be the same environment `colab` itself runs in.

## 3. Auth is a live, interactive browser flow — can't run inside a one-shot tool call

**Symptom:** `colab sessions` (or any first command) prints a Google OAuth
URL and blocks on `Enter the authorization code:` — needs a human to open
the URL, approve, and paste back a code.

**Not really a bug** — this is inherent to `--auth oauth2` (the CLI's
default). It has to be run in a terminal a person is actually watching, not
issued as a single non-interactive command whose stdin closes immediately.
Once done, the token is cached (`~/.config/colab-cli/token.json`); every
subsequent command is non-interactive. If scripting this for a fully
unattended/agent context, the skill file (`colab skill`) recommends
`--auth adc` (Application Default Credentials via `gcloud`) instead — still
a one-time human step, just via `gcloud auth application-default login`
rather than a code-paste.

## 4. `colab install -r requirements.txt` drops the connection on real installs

**Symptom:**
```
[colab] Installing packages on soamp-grid (preferring uv)...
RuntimeError: Connection was lost.
```
The VM survives (`colab status` still shows it IDLE right after) — only the
`colab install` command's own websocket connection dies, apparently on
longer-running installs (this repo's `requirements-colab.txt` pulls in
`torch`-adjacent packages, `transformers`, `rdkit`, etc.).

**Fix:** don't use `colab install` for anything nontrivial. Write a small
wrapper script and run it via `colab exec -f wrapper.py --timeout <seconds>`
instead, calling `subprocess.run([sys.executable, "-m", "pip", "install",
...])` directly — `colab exec` takes an explicit `--timeout` (default is
only **30 seconds**, another thing to override for anything but a trivial
call), so you control it instead of hitting whatever internal limit
`colab install` has.

## 5. `pip install -e .` succeeds, then `import soamp...` fails right after

**Symptom:** a wrapper script runs `pip install -e . --no-deps` (exit 0, no
errors), but the *next* `colab exec` call doing `from soamp.data.factory
import build_dataset` raises `ModuleNotFoundError: No module named
'soamp.data.factory'`.

**Cause:** per `colab skill`'s own "mental model" section, kernel state
*persists* across separate `colab exec` calls — each one reattaches to the
same long-lived Python process rather than starting fresh. A `pip install
-e .` run via a `subprocess.run` **inside that process** writes a new
editable-install path-finder registration, but Python only processes those
at *interpreter startup* (`site` module import) — the already-running kernel
process doesn't pick it up retroactively. A **freshly spawned** `python
pipeline/train.py` subprocess (a new process, not a call within the
persisted kernel) *would* pick it up immediately — this only bites code that
does `import soamp...` directly inside a `colab exec`'d script rather than
shelling out to a fresh process.

**Fix:** `colab restart-kernel -s <name>` right after any install step,
before the next `colab exec` that imports the package directly. Cheap (resets
the Python process, not the VM) and the safe default any time you install or
upgrade something mid-session.

## 6. `colab exec`/`colab status` occasionally hard-timeout on SSL handshake, but the kernel keeps running

**Symptom:** a long-running `colab exec` (e.g. running all four training
configs sequentially) or even a plain `colab status` dies with `TimeoutError:
_ssl.c:1001: The handshake operation timed out` / `ReadTimeoutError` against
`colab.research.google.com`, propagated up as a big traceback.

**This looks alarming but isn't necessarily a failure** — it can be a
transient client-side network hiccup unrelated to the actual VM or kernel.
**Don't blindly retry** the same script (risks double-running expensive
work and wasting billed GPU time) — instead:
1. `colab status -s <name>` — if it also times out, just retry it alone
   (cheap) until it answers.
2. Check `Status: IDLE` vs `BUSY` and `Last Execution` — `BUSY` means
   whatever you last submitted is still genuinely running server-side
   despite your client losing the connection; wait and re-check rather than
   resubmitting.
3. If `IDLE` and `Last Execution` names something *earlier* than what you
   just tried to run, the submission likely never completed — check for
   partial output (e.g. `colab exec` a one-liner listing the files a script
   was supposed to produce) before deciding whether to retry.

In this session's actual run, this happened exactly once, right as the grid
script started (`colab log` showed the exec had *begun* but never logged
completion); `colab status` confirmed `IDLE` with no new checkpoint files
produced, so it was safe to just resubmit the same script, which then ran
all four configs to completion in one clean call.

## Operational notes not tied to a specific bug

- **`colab exec -f FILE` sends FILE's *local* content to the remote kernel
  and executes it there — it does not reference a path already inside a
  cloned repo on the VM.** Every wrapper script pushed this way should do
  its real work via `subprocess.run([...], cwd="/content/soamp")` against
  the actual entrypoint scripts, rather than trying to `-f` a file that's
  only meaningful once cloned remotely. This matters concretely for this
  repo: several config modules resolve
  `REPO_ROOT = Path(__file__).resolve().parents[N]`, which breaks if a
  script's content is executed without `__file__` pointing at a real path
  inside the actual clone.
- **Secret handling for the GitHub clone and the wandb key:** neither this
  repo's `GITHUB_TOKEN` nor `WANDB_API_KEY` should ever appear in anything
  printed back to a terminal/log. The pattern used here: read the value
  from `.env` into a shell variable inside one `bash -c` (never echoed),
  write it into a short-lived wrapper script or a minimal `.env` file with
  `chmod 600`, push it via `colab exec -f`/`colab upload`, then delete the
  local scratch copy immediately after the remote call succeeds. Git
  operations that might echo a credential in their own error output (e.g. a
  failed clone) should have the token string replaced with `***` before
  printing.
- **`colab stop -s <name>` is not optional.** Confirm with `colab sessions`
  afterward — an empty list is the only real confirmation nothing is still
  billing.
