"""Engine/Trainer: runs the train/eval loop only -- forward, loss,
backward, optimizer step. No checkpoint I/O, no tracker calls, no config
loading (per CLAUDE.md sec 6 -- those are separate collaborators).
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """One grad-updating pass. Returns {'loss': mean_train_loss}."""
        self.model.train()
        total_loss, n_batches = 0.0, 0
        for peptide_features, organism_input, labels in loader:
            peptide_features = peptide_features.to(self.device)
            organism_input = organism_input.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(peptide_features, organism_input)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        return {"loss": total_loss / n_batches}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, "np.ndarray | float"]:
        """No-grad forward pass. Returns raw logits/labels/organism_input
        plus mean loss. organism_input is whatever the featurizer encodes --
        an (N,) index vector for output_kind="index", an (N, D) float matrix
        for "vector" -- so per-organism bucketing works off the eval rows'
        organism column instead (see engine/metrics.py). Metric *interpretation*
        (accuracy/F1/AUROC/per-organism) is orchestration's job
        (engine/metrics.py), kept out of Trainer per sec 6."""
        self.model.eval()
        all_logits, all_labels, all_organism_inputs = [], [], []
        total_loss, n_batches = 0.0, 0
        for peptide_features, organism_input, labels in loader:
            peptide_features = peptide_features.to(self.device)
            organism_input = organism_input.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(peptide_features, organism_input)
            loss = self.loss_fn(logits, labels)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_organism_inputs.append(organism_input.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

        return {
            "logits": np.concatenate(all_logits),
            "labels": np.concatenate(all_labels),
            "organism_input": np.concatenate(all_organism_inputs),
            "loss": total_loss / n_batches,
        }
