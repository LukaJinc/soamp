import torch
from torch.utils.data import DataLoader, TensorDataset

from soamp.engine.checkpointer import Checkpointer, CheckpointMetadata
from soamp.common.tracking import LocalFileTracker
from soamp.engine.trainer import Trainer
from soamp.model.baseline_mlp import BaselineClassifier
from soamp.utils.reproducibility import seed_everything


def test_trainer_checkpointer_tracker_end_to_end(tmp_path):
    seed_everything(0)
    n = 12
    dataset = TensorDataset(
        torch.randn(n, 5), torch.randint(0, 3, (n,)), torch.randint(0, 2, (n,)).float()
    )
    loader = DataLoader(dataset, batch_size=4)

    model = BaselineClassifier(peptide_feature_dim=5, organism_vocab_size=3, hidden_dims=[8])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    trainer = Trainer(model, optimizer, loss_fn)

    checkpointer = Checkpointer(tmp_path / "checkpoints", "integration_exp")
    tracker = LocalFileTracker(tmp_path / "runs", "integration_exp")
    tracker.log_config({"seed": 0})

    for epoch in range(1, 3):
        train_metrics = trainer.train_epoch(loader)
        assert train_metrics["loss"] == train_metrics["loss"]  # not NaN
        tracker.log_metrics({"train_loss": train_metrics["loss"]}, step=epoch)

        metadata = CheckpointMetadata(
            exp_id="integration_exp", iter=epoch, git_sha="testsha", seed=0, resolved_config={}
        )
        checkpointer.save_weights(model, epoch, metadata, best=(epoch == 2))
        checkpointer.save_full_state(model, optimizer, epoch, metadata)

    assert (tmp_path / "checkpoints" / "integration_exp_best.pth").exists()
    assert (tmp_path / "checkpoints" / "integration_exp_1.pth").exists()
    assert (tmp_path / "checkpoints" / "integration_exp_checkpoint_2.pth").exists()
    assert (tmp_path / "runs" / "integration_exp" / "metrics.jsonl").exists()
    assert (tmp_path / "runs" / "integration_exp" / "run_metadata.json").exists()
