"""git SHA + seeding helpers, so neither is ever a bare literal in a script."""
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


class ReproducibilityError(RuntimeError):
    """Raised when the git SHA couldn't be determined -- fails loud
    rather than defaulting to 'unknown', since an untracked checkpoint
    isn't reproducible."""


def git_sha(repo_root: Path = REPO_ROOT) -> str:
    """`git rev-parse HEAD` run with cwd=repo_root. Raises
    ReproducibilityError on any failure (not a repo, git missing)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise ReproducibilityError(f"could not determine git SHA: {e}") from e
    return result.stdout.strip()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
