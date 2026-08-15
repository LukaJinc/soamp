import torch
from torch.utils.data import DataLoader, TensorDataset

from soamp.engine.trainer import Trainer
from soamp.model.baseline_mlp import BaselineClassifier


def _loader(n=16):
    peptide_features = torch.randn(n, 13)
    organism_idx = torch.randint(0, 4, (n,))
    labels = torch.randint(0, 2, (n,)).float()
    dataset = TensorDataset(peptide_features, organism_idx, labels)
    return DataLoader(dataset, batch_size=4)


def _trainer():
    model = BaselineClassifier(peptide_feature_dim=13, organism_vocab_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    return Trainer(model, optimizer, loss_fn), model


def test_train_epoch_returns_loss_and_runs_without_error():
    trainer, _ = _trainer()
    result = trainer.train_epoch(_loader())
    assert "loss" in result
    assert result["loss"] == result["loss"]  # not NaN


def test_train_epoch_updates_model_parameters():
    trainer, model = _trainer()
    before = [p.clone() for p in model.parameters()]
    trainer.train_epoch(_loader())
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_evaluate_returns_arrays_matching_dataset_size():
    trainer, _ = _trainer()
    out = trainer.evaluate(_loader(n=16))
    assert out["logits"].shape == (16,)
    assert out["labels"].shape == (16,)
    assert out["organism_idx"].shape == (16,)


def test_evaluate_is_no_grad():
    trainer, model = _trainer()
    before = [p.clone() for p in model.parameters()]
    trainer.evaluate(_loader())
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))
