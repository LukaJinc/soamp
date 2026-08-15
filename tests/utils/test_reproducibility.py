import re

import torch

from soamp.utils.reproducibility import git_sha, seed_everything


def test_git_sha_returns_40_char_hex_string_in_this_repo():
    sha = git_sha()
    assert re.fullmatch(r"[0-9a-f]{40}", sha)


def test_seed_everything_makes_torch_rand_deterministic():
    seed_everything(123)
    a = torch.rand(5)
    seed_everything(123)
    b = torch.rand(5)
    assert torch.equal(a, b)
