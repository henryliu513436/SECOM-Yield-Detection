"""lot_db：SQLite 存取層（lots / flags / approval_log）。"""

import math

import pytest

from lot_db import LotRecord, LotRepository, create_db


def _lots() -> list[LotRecord]:
    return [
        LotRecord("LOT-001", 1, "2008-10-01T10:00:00", (1.0, None, 3.5)),
        LotRecord("LOT-002", 2, "2008-10-02T11:30:00", (2.0, 0.5, None)),
    ]


@pytest.fixture
def repo(tmp_path) -> LotRepository:
    path = tmp_path / "secom.db"
    create_db(path, _lots())
    return LotRepository(path)


def test_get_lot_round_trips_features_including_missing(repo: LotRepository) -> None:
    lot = repo.get_lot("LOT-001")

    assert lot is not None
    assert lot.produced_at == "2008-10-01T10:00:00"
    assert lot.features[0] == 1.0
    assert lot.features[1] is None
    assert math.isclose(lot.features[2], 3.5)


def test_get_lot_returns_none_for_unknown_id(repo: LotRepository) -> None:
    assert repo.get_lot("LOT-999") is None


def test_list_lots_is_ordered_by_sequence(repo: LotRepository) -> None:
    assert [lot.lot_id for lot in repo.list_lots()] == ["LOT-001", "LOT-002"]
    assert repo.count() == 2


def test_add_flag_persists_and_is_listed(repo: LotRepository) -> None:
    flag = repo.add_flag("LOT-002", "SHAP 顯示 sensor_59 貢獻最大", flagged_at="2026-10-03T12:00:00+00:00")

    flags = repo.list_flags("LOT-002")
    assert flags == [flag]
    assert flag.reason == "SHAP 顯示 sensor_59 貢獻最大"
    assert repo.list_flags("LOT-001") == []
    assert repo.flagged_lot_ids() == {"LOT-002"}


def test_record_approval_keeps_audit_trail(repo: LotRepository) -> None:
    repo.record_approval("flag_lot", {"lot_id": "LOT-001"}, "rejected", decided_at="2026-10-03T12:00:00+00:00")
    repo.record_approval("flag_lot", {"lot_id": "LOT-002"}, "approved", decided_at="2026-10-03T12:01:00+00:00")

    log = repo.list_approvals()
    assert [(entry.tool, entry.arguments["lot_id"], entry.decision) for entry in log] == [
        ("flag_lot", "LOT-001", "rejected"),
        ("flag_lot", "LOT-002", "approved"),
    ]


def test_create_db_refuses_to_overwrite_existing_file(tmp_path) -> None:
    path = tmp_path / "secom.db"
    create_db(path, _lots())

    with pytest.raises(FileExistsError):
        create_db(path, _lots())


def test_repository_requires_existing_db(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="build_db.py"):
        LotRepository(tmp_path / "missing.db")


def test_records_from_test_frame_assigns_sequential_ids_and_drops_label() -> None:
    import pandas as pd

    from lot_db import format_lot_id, records_from_frame

    frame = pd.DataFrame(
        {
            "Time": pd.to_datetime(["2008-10-01 10:00:00", "2008-10-02 11:30:00"]),
            "0": [1.0, float("nan")],
            "1": [2.0, 3.0],
            "Pass/Fail": [-1, 1],
        }
    )

    records = records_from_frame(frame)

    assert [r.lot_id for r in records] == ["LOT-001", "LOT-002"]
    assert records[0].produced_at == "2008-10-01T10:00:00"
    assert records[1].features == (None, 3.0)  # 只有感測器欄，沒有標籤
    assert format_lot_id(47) == "LOT-047"
