import torch
from torch import nn

from soamp.engine.checkpointer import Checkpointer, CheckpointMetadata


def _metadata(iter=1):
    return CheckpointMetadata(
        exp_id="test_exp", iter=iter, git_sha="abc123", seed=42, resolved_config={"a": 1}
    )


def test_save_and_load_weights_round_trip(tmp_path):
    ckpt = Checkpointer(tmp_path, "test_exp")
    model = nn.Linear(3, 1)
    ckpt.save_weights(model, iter=1, metadata=_metadata(1))

    loaded_model = nn.Linear(3, 1)
    metadata = Checkpointer.load_weights(tmp_path / "test_exp_1.pth", loaded_model)
    assert torch.equal(model.weight, loaded_model.weight)
    assert metadata.git_sha == "abc123"


def test_save_and_load_full_state_round_trip_restores_optimizer(tmp_path):
    ckpt = Checkpointer(tmp_path, "test_exp")
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # take a step so optimizer state isn't just the initial empty state
    loss = model(torch.randn(1, 3)).sum()
    loss.backward()
    optimizer.step()

    ckpt.save_full_state(model, optimizer, iter=1, metadata=_metadata(1))

    loaded_model = nn.Linear(3, 1)
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-3)
    metadata = Checkpointer.load_full_state(
        tmp_path / "test_exp_checkpoint_1.pth", loaded_model, loaded_optimizer
    )
    assert torch.equal(model.weight, loaded_model.weight)
    assert metadata.seed == 42


def test_weights_filename_matches_naming_scheme(tmp_path):
    ckpt = Checkpointer(tmp_path, "exp_a")
    path = ckpt.save_weights(nn.Linear(2, 1), iter=7, metadata=_metadata(7))
    assert path.name == "exp_a_7.pth"


def test_best_filename_matches_naming_scheme(tmp_path):
    ckpt = Checkpointer(tmp_path, "exp_a")
    path = ckpt.save_weights(nn.Linear(2, 1), iter=7, metadata=_metadata(7), best=True)
    assert path.name == "exp_a_best.pth"


def test_full_state_filename_matches_naming_scheme(tmp_path):
    ckpt = Checkpointer(tmp_path, "exp_a")
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    path = ckpt.save_full_state(model, optimizer, iter=7, metadata=_metadata(7))
    assert path.name == "exp_a_checkpoint_7.pth"


def test_metadata_embedded_and_recovered(tmp_path):
    ckpt = Checkpointer(tmp_path, "exp_a")
    metadata_in = CheckpointMetadata(
        exp_id="exp_a", iter=3, git_sha="deadbeef", seed=99, resolved_config={"lr": 0.01}
    )
    ckpt.save_weights(nn.Linear(2, 1), iter=3, metadata=metadata_in)
    metadata_out = Checkpointer.load_weights(tmp_path / "exp_a_3.pth", nn.Linear(2, 1))
    assert metadata_out == metadata_in
