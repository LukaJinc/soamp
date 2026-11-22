import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from soamp.runs.budget import CostModel, Limits
from soamp.runs.launcher import LaunchRefused, launch
from soamp.runs.ledger import Ledger
from soamp.runs.plan import expand
from soamp.runs.request import ExperimentRequest
from soamp.runs.vast import Offer, VastClient, VastInstance

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
LIM = Limits(max_cost_per_request_usd=5, max_concurrent_instances=2, max_cost_per_24h_usd=20, max_cells=40,
             max_hours_per_instance=3, safety_margin=1.25)
MODEL = CostModel(minutes_per_100_epochs={"rdkit": 6.0}, instance_overhead_minutes=15, assumed_price_per_hour_usd=0.5)


class FakeClient(VastClient):
    def __init__(self, running=0, offers=2, price=0.5):
        self.created, self._running = [], running
        self._offers = [Offer(100 + i, "RTX_4090", price, 0.999, 900, "DE") for i in range(offers)]

    def search_offers(self, instance, limit=8):
        return self._offers[:limit]

    def list_instances(self, label_prefix="soamp-"):
        return [VastInstance(i, "soamp-x", "running", 0.5) for i in range(self._running)]

    def create_instance(self, offer_id, **kw):
        self.created.append((offer_id, kw))
        return 900 + len(self.created)


class FakeBackend:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def log_launch(self, m):
        self.rows.append(m.to_dict())

    def launches(self):
        return list(self.rows)


def _req():
    return ExperimentRequest(
        experiment="unit_exp", representations={"peptide": ["rdkit"], "organism": ["vocab"]},
        architectures=[{"name": "baseline_classifier", "hparams": {"weight_decay": [1e-3, 1e-4]}, "epochs": 5}],
    )


def _go(tmp_path: Path, client=None, backend=None, finished=frozenset(), limits=LIM):
    client = client or FakeClient()
    backend = backend or FakeBackend()
    published = []
    res = launch(
        _req(), client=client, ledger=Ledger(backend), limits=limits, cost_model=MODEL, finished=set(finished),
        publish_bundle=lambda extra: published.append(extra) or "ent/soamp-ops/code-unit_exp:v0", data_ref="ent/soamp-ops/soamp_dev_data:latest",
        entity="ent", wandb_api_key="SECRET", git_sha="abc", repo_root=tmp_path, now=NOW,
    )
    return res, client, backend, published


def test_launch_rents_labels_and_records(tmp_path):
    res, client, backend, published = _go(tmp_path)
    assert len(res.launched_cells) == 2 and len(client.created) == 1
    offer_id, kw = client.created[0]
    assert kw["label"] == "soamp-unit_exp" and kw["env"]["SOAMP_EXPERIMENT"] == "unit_exp" and kw["env"]["WANDB_API_KEY"] == "SECRET"
    assert "SECRET" not in kw["onstart_cmd"]
    assert backend.rows[0]["instance_ids"] == res.instance_ids and backend.rows[0]["git_sha"] == "abc"
    # generated configs + run spec were written and shipped
    assert any("config/vast/generated/unit_exp_0.yaml" in f for f in published[0])
    assert all((tmp_path / f).is_file() for f in published[0])


def test_finished_cells_skipped_and_all_finished_launches_nothing(tmp_path):
    ids = [c.exp_id for c in expand(_req())]
    res, client, backend, _ = _go(tmp_path, finished=ids[:1])
    assert res.skipped_finished == ids[:1] and len(res.launched_cells) == 1
    res2, client2, backend2, _ = _go(tmp_path, finished=ids)
    assert res2.launched_cells == [] and client2.created == [] and backend2.rows == []


def test_refuses_before_renting_when_caps_hit(tmp_path):
    for client, backend, limits in [
        (FakeClient(running=2), None, LIM),  # concurrency
        (None, FakeBackend([{"experiment": "o", "request_hash": "h", "created_at": NOW.isoformat(), "deadline": NOW.isoformat(),
                             "estimate_usd": 19.9, "git_sha": "", "cells": [], "instance_ids": []}]), LIM),  # 24h spend
        (FakeClient(price=500.0), None, LIM),  # per-request cost
    ]:
        c = client or FakeClient()
        with pytest.raises(LaunchRefused):
            _go(tmp_path, client=c, backend=backend, limits=limits)
        assert c.created == []


def test_refuses_without_offers(tmp_path):
    with pytest.raises(LaunchRefused, match="no vast.ai offers"):
        _go(tmp_path, client=FakeClient(offers=0))
