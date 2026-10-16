"""Agent 四個工具的邏輯：list_lots / get_lot / explain_lot / flag_lot。

這一層刻意不依賴 MCP：工具邏輯、參數驗證、錯誤訊息都在這裡，可以直接
單元測試；mcp_server.py 只負責把它們註冊成 MCP 工具。

錯誤一律丟 LotToolError，訊息是寫給模型看的——要說清楚哪裡錯、正確的
格式或範圍是什麼，模型才能自己修正下一次呼叫，而不是卡住或亂猜。

數字全部來自推論服務（FastAPI）與 metadata.json，這裡不做任何模型計算。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx

from lot_db import LotRecord, LotRepository, format_lot_id

DEFAULT_METADATA_PATH = Path(__file__).resolve().parents[1] / "models" / "metadata.json"
MAX_LIST_LIMIT = 20
MAX_REASON_LENGTH = 300
ORDER_BY_CHOICES = ("risk", "recent")
API_TIMEOUT_SECONDS = 10.0

_LOT_ID_PATTERN = re.compile(r"^\s*(?:lot[\s_-]*)?0*(\d+)\s*$", re.IGNORECASE)

SCORE_NOTE = "risk_score 是未校準的排序分數（愈高愈可疑），不是 Fail 的機率；above_threshold 代表超過模型的決策閾值。"


class LotToolError(Exception):
    """工具呼叫失敗；訊息會原封不動回給模型。"""


class InferenceClient(Protocol):
    def predict(self, lot_id: str, features: tuple) -> dict: ...

    def explain(self, lot_id: str, features: tuple) -> dict: ...


class HttpInferenceClient:
    """呼叫既有 FastAPI 的 /predict 與 /explain。"""

    def __init__(self, base_url: str, timeout: float = API_TIMEOUT_SECONDS) -> None:
        self._base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def _post(self, path: str, lot_id: str, features: tuple) -> dict:
        try:
            response = self._client.post(path, json={"lot_id": lot_id, "features": list(features)})
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LotToolError(
                f"推論服務（{self._base_url}）無法連線，請使用者確認 FastAPI 已啟動；這不是批次本身的問題。"
            ) from exc
        except httpx.HTTPError as exc:
            raise LotToolError(f"推論服務回傳錯誤：{exc}") from exc
        return response.json()

    def predict(self, lot_id: str, features: tuple) -> dict:
        return self._post("/predict", lot_id, features)

    def explain(self, lot_id: str, features: tuple) -> dict:
        return self._post("/explain", lot_id, features)


@dataclass(frozen=True)
class ModelInfo:
    """模型可信度的摘要數字，來自 train.py 寫進 metadata.json 的 cv_summary。"""

    cv_roc_auc_mean: float
    cv_pr_auc_ratio_mean: float


def load_model_info(path: Path = DEFAULT_METADATA_PATH) -> ModelInfo:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}，請先執行 `python src/train.py`")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if "cv_summary" not in metadata:
        raise ValueError(f"{path} 是舊版 metadata（沒有 cv_summary），請重新執行 `python src/train.py`")
    summary = metadata["cv_summary"]
    return ModelInfo(summary["xgb_roc_auc_mean"], summary["xgb_pr_auc_ratio_mean"])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_lot_id(raw: str | int) -> str:
    """接受 LOT-047、lot-47、47 等寫法，統一成 LOT-047。小模型常常會省略
    前綴或補零方式不同，這裡寬鬆接受，比讓它因格式問題失敗更實際。"""
    match = _LOT_ID_PATTERN.match(str(raw))
    if match is None:
        raise LotToolError(f"lot_id 格式錯誤：{raw!r}。請使用 LOT-001 這種格式，例如 LOT-047。")
    return format_lot_id(int(match.group(1)))


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise LotToolError(f"{name} 必須是 true 或 false，收到 {value!r}")


def _parse_limit(value: Any) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise LotToolError(f"limit 必須是整數，收到 {value!r}") from None
    if not 1 <= limit <= MAX_LIST_LIMIT:
        raise LotToolError(f"limit 必須介於 1 到 {MAX_LIST_LIMIT}，收到 {limit}")
    return limit


class LotTools:
    """四個工具的實作。風險分數第一次用到時向推論服務查詢並快取——同一
    個程序裡模型不會變，不需要每次 list_lots 都重打 471 次 /predict。"""

    def __init__(
        self,
        repo: LotRepository,
        client: InferenceClient,
        model_info: ModelInfo,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self._repo = repo
        self._client = client
        self._model_info = model_info
        self._clock = clock
        self._predictions: dict[str, dict] = {}

    def _require_lot(self, raw_lot_id: str | int) -> LotRecord:
        lot_id = normalize_lot_id(raw_lot_id)
        lot = self._repo.get_lot(lot_id)
        if lot is None:
            lots = self._repo.list_lots()
            raise LotToolError(
                f"找不到 {lot_id}，現有批次範圍為 {lots[0].lot_id} 至 {lots[-1].lot_id}"
            )
        return lot

    def _predict(self, lot: LotRecord) -> dict:
        if lot.lot_id not in self._predictions:
            self._predictions[lot.lot_id] = self._client.predict(lot.lot_id, lot.features)
        return self._predictions[lot.lot_id]

    def _reliability_note(self) -> str:
        return (
            f"模型可信度有限：時間序列交叉驗證 ROC-AUC 約 {self._model_info.cv_roc_auc_mean:.2f}"
            f"（0.5 等於隨機），PR-AUC 約為隨機基準的 {self._model_info.cv_pr_auc_ratio_mean:.1f} 倍。"
            "SHAP 說明的是「模型為什麼給這個分數」，不代表已確認的製程根因；"
            "感測器是匿名的，sensor 編號沒有已知的物理意義。"
        )

    def list_lots(self, limit: int = 5, order_by: str = "risk", above_threshold_only: bool = False) -> dict:
        limit = _parse_limit(limit)
        if order_by not in ORDER_BY_CHOICES:
            raise LotToolError(f"order_by 只能是 {' 或 '.join(ORDER_BY_CHOICES)}，收到 {order_by!r}")
        above_only = _parse_bool(above_threshold_only, "above_threshold_only")

        lots = self._repo.list_lots()
        if order_by == "recent":
            candidates = list(reversed(lots))
        else:
            candidates = sorted(lots, key=lambda lot: self._predict(lot)["risk_score"], reverse=True)
        if above_only:
            candidates = [lot for lot in candidates if self._predict(lot)["is_fail_predicted"]]

        flagged = self._repo.flagged_lot_ids()
        selected = candidates[:limit]
        rows = []
        for lot in selected:
            prediction = self._predict(lot)
            rows.append(
                {
                    "lot_id": lot.lot_id,
                    "produced_at": lot.produced_at,
                    "risk_score": round(prediction["risk_score"], 4),
                    "above_threshold": prediction["is_fail_predicted"],
                    "already_flagged": lot.lot_id in flagged,
                }
            )
        threshold = self._predict(selected[0])["threshold"] if selected else None
        return {
            "total_lots": len(lots),
            "returned": len(rows),
            "order_by": order_by,
            "threshold": threshold,
            "lots": rows,
            "note": SCORE_NOTE if rows else "沒有符合條件的批次。",
        }

    def get_lot(self, lot_id: str) -> dict:
        lot = self._require_lot(lot_id)
        prediction = self._predict(lot)
        return {
            "lot_id": lot.lot_id,
            "produced_at": lot.produced_at,
            "risk_score": round(prediction["risk_score"], 4),
            "above_threshold": prediction["is_fail_predicted"],
            "threshold": prediction["threshold"],
            "flags": [{"reason": f.reason, "flagged_at": f.flagged_at} for f in self._repo.list_flags(lot.lot_id)],
            "note": SCORE_NOTE,
        }

    def explain_lot(self, lot_id: str) -> dict:
        lot = self._require_lot(lot_id)
        explanation = self._client.explain(lot.lot_id, lot.features)
        return {
            "lot_id": lot.lot_id,
            "risk_score": round(explanation["risk_score"], 4),
            "base_value": round(explanation["base_value"], 4),
            "top_contributors": [
                {
                    "sensor": f"sensor_{item['feature']}",
                    "shap_value": round(item["shap_value"], 4),
                    "actual_value": item["actual_value"],
                    "effect": "推高風險" if item["shap_value"] > 0 else "降低風險",
                }
                for item in explanation["top_features"]
            ],
            "reliability_note": self._reliability_note(),
        }

    def flag_lot(self, lot_id: str, reason: str) -> dict:
        lot = self._require_lot(lot_id)
        reason = (reason or "").strip()
        if not reason:
            raise LotToolError("reason 不能是空的，請用一句話說明為什麼要標記這個批次（例如引用風險分數或 SHAP 結果）")
        if len(reason) > MAX_REASON_LENGTH:
            raise LotToolError(f"reason 太長（{len(reason)} 字），請精簡到 {MAX_REASON_LENGTH} 字以內")

        existing = self._repo.list_flags(lot.lot_id)
        if existing:
            first = existing[0]
            raise LotToolError(
                f"{lot.lot_id} 已於 {first.flagged_at} 標記過（原因：{first.reason}），不需重複標記"
            )
        flag = self._repo.add_flag(lot.lot_id, reason, flagged_at=self._clock())
        return {
            "status": "flagged",
            "flag_id": flag.id,
            "lot_id": flag.lot_id,
            "reason": flag.reason,
            "flagged_at": flag.flagged_at,
        }
