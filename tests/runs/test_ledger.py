from datetime import datetime, timedelta, timezone

from soamp.runs.ledger import Ledger, Manifest, deadline_for, spend_in_window

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


def _m(created: datetime, usd: float, deadline: datetime | None = None, exp="e") -> Manifest:
    return Manifest(exp, "h", created.isoformat(), (deadline or created + timedelta(hours=3)).isoformat(), usd, "sha", ["c1"])


class FakeBackend:
    def __init__(self):
        self.rows: list[dict] = []

    def log_launch(self, m: Manifest) -> None:
        self.rows.append(m.to_dict())

    def launches(self) -> list[dict]:
        return list(self.rows)


def test_spend_counts_only_last_24h():
    rows = [_m(NOW - timedelta(hours=1), 2.0).to_dict(), _m(NOW - timedelta(hours=23, minutes=59), 1.5).to_dict(),
            _m(NOW - timedelta(hours=25), 9.0).to_dict()]
    assert spend_in_window(rows, NOW) == 3.5


def test_ledger_round_trip_and_open_manifests():
    b = FakeBackend()
    led = Ledger(b)
    led.record_launch(_m(NOW - timedelta(hours=1), 1.0, deadline=NOW + timedelta(hours=1), exp="open"))
    led.record_launch(_m(NOW - timedelta(hours=5), 1.0, deadline=NOW - timedelta(hours=1), exp="expired"))
    assert led.spend_last_24h(NOW) == 2.0
    assert [m.experiment for m in led.open_manifests(NOW)] == ["open"]


def test_manifest_dict_round_trip_and_deadline():
    m = _m(NOW, 1.0)
    assert Manifest.from_dict(m.to_dict()) == m
    assert not m.is_past_deadline(NOW) and m.is_past_deadline(NOW + timedelta(hours=3))
    d = deadline_for(NOW, max_hours=2, grace_seconds=120, buffer_minutes=30)
    assert d == NOW + timedelta(hours=2, minutes=32)
