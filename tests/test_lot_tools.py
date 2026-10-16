"""lot_tools：四個 agent 工具的邏輯與寫給模型看的錯誤訊息。

推論服務用假的 client 取代，不需要啟動 FastAPI。
"""

import pytest

from lot_db import LotRecord, LotRepository, create_db
from lot_tools import LotToolError, LotTools, ModelInfo, normalize_lot_id

THRESHOLD = 0.5
SCORES = {"LOT-001": 0.10, "LOT-002": 0.90, "LOT-003": 0.60, "LOT-004": 0.20}


class FakeClient:
    """依 lot_id 回固定分數，並記錄被呼叫幾次（用來驗證快取）。"""

    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, lot_id: str, features: tuple) -> dict:
        self.predict_calls += 1
        score = SCORES[lot_id]
        return {"lot_id": lot_id, "risk_score": score, "is_fail_predicted": score >= THRESHOLD, "threshold": THRESHOLD}

    def explain(self, lot_id: str, features: tuple) -> dict:
        return {
            "lot_id": lot_id,
            "base_value": 0.3,
            "risk_score": SCORES[lot_id],
            "top_features": [
                {"feature": "59", "shap_value": 0.4, "actual_value": 2.5},
                {"feature": "103", "shap_value": -0.1, "actual_value": 0.0},
            ],
        }


@pytest.fixture
def tools(tmp_path) -> LotTools:
    path = tmp_path / "secom.db"
    create_db(
        path,
        [LotRecord(f"LOT-00{i}", i, f"2008-10-0{i}T10:00:00", (float(i),)) for i in range(1, 5)],
    )
    return LotTools(
        repo=LotRepository(path),
        client=FakeClient(),
        model_info=ModelInfo(cv_roc_auc_mean=0.61, cv_pr_auc_ratio_mean=2.37),
        clock=lambda: "2026-10-03T12:00:00+00:00",
    )


@pytest.mark.parametrize("raw", ["LOT-002", "lot-2", "LOT-0002", "2", 2, " Lot 002 "])
def test_normalize_lot_id_accepts_common_spellings(raw) -> None:
    assert normalize_lot_id(raw) == "LOT-002"


def test_normalize_lot_id_rejects_garbage_with_format_hint() -> None:
    with pytest.raises(LotToolError, match="LOT-001"):
        normalize_lot_id("第二批")


def test_unknown_lot_error_tells_model_the_valid_range(tools: LotTools) -> None:
    with pytest.raises(LotToolError, match="找不到 LOT-999，現有批次範圍為 LOT-001 至 LOT-004"):
        tools.get_lot("LOT-999")


def test_list_lots_by_risk_sorts_descending_and_caches_scores(tools: LotTools) -> None:
    result = tools.list_lots(limit=2, order_by="risk")

    assert [lot["lot_id"] for lot in result["lots"]] == ["LOT-002", "LOT-003"]
    assert result["total_lots"] == 4
    assert result["lots"][0]["above_threshold"] is True

    tools.list_lots(limit=4, order_by="risk")
    assert tools._client.predict_calls == 4  # 第二次呼叫全部走快取


def test_list_lots_recent_returns_newest_first(tools: LotTools) -> None:
    result = tools.list_lots(limit=2, order_by="recent")

    assert [lot["lot_id"] for lot in result["lots"]] == ["LOT-004", "LOT-003"]


def test_list_lots_above_threshold_only(tools: LotTools) -> None:
    result = tools.list_lots(limit=10, order_by="risk", above_threshold_only=True)

    assert [lot["lot_id"] for lot in result["lots"]] == ["LOT-002", "LOT-003"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"limit": 0}, "limit 必須介於 1 到 20"), ({"order_by": "score"}, "order_by 只能是")],
)
def test_list_lots_rejects_bad_arguments_with_readable_message(tools: LotTools, kwargs, message) -> None:
    with pytest.raises(LotToolError, match=message):
        tools.list_lots(**kwargs)


def test_get_lot_reports_score_threshold_and_flags(tools: LotTools) -> None:
    tools.flag_lot("LOT-003", "風險分數超過閾值")

    result = tools.get_lot("lot-3")

    assert result["lot_id"] == "LOT-003"
    assert result["risk_score"] == 0.6
    assert result["threshold"] == THRESHOLD
    assert result["flags"] == [{"reason": "風險分數超過閾值", "flagged_at": "2026-10-03T12:00:00+00:00"}]


def test_explain_lot_names_sensors_and_states_model_limits(tools: LotTools) -> None:
    result = tools.explain_lot("LOT-002")

    first = result["top_contributors"][0]
    assert first["sensor"] == "sensor_59"
    assert first["effect"] == "推高風險"
    assert result["top_contributors"][1]["effect"] == "降低風險"
    assert "0.61" in result["reliability_note"]
    assert "匿名" in result["reliability_note"]


def test_flag_lot_writes_flag(tools: LotTools) -> None:
    result = tools.flag_lot("LOT-002", "SHAP 顯示 sensor_59 貢獻最大")

    assert result["status"] == "flagged"
    assert result["lot_id"] == "LOT-002"
    assert tools.list_lots(limit=1)["lots"][0]["already_flagged"] is True


def test_flag_lot_rejects_duplicate_with_existing_reason(tools: LotTools) -> None:
    tools.flag_lot("LOT-002", "第一次標記")

    with pytest.raises(LotToolError, match="已於 .* 標記過（原因：第一次標記）"):
        tools.flag_lot("LOT-002", "再標一次")


def test_flag_lot_requires_reason(tools: LotTools) -> None:
    with pytest.raises(LotToolError, match="reason"):
        tools.flag_lot("LOT-002", "   ")
