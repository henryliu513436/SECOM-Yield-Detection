"""MCP server：把 lot_tools 的四個工具註冊成 MCP 工具，走 stdio。

這一層只做註冊與錯誤轉換，邏輯全在 lot_tools.py。讀取工具標
read_only_hint=True；flag_lot 標 read_only_hint=False、destructive_hint=False
（新增紀錄、不覆寫既有資料），agent 端看到不是唯讀就會要求人工核准。

工具的 docstring 就是模型看到的 description——模型完全靠這段文字決定
要不要用、怎麼填參數，所以寫的是「什麼時候該用」與參數格式，而不只是
「它做什麼」。

stdout 是 MCP 協定通道，這個程序不能 print 任何東西到 stdout。

設定（環境變數）：
    SECOM_API_BASE_URL  推論服務位址，預設 http://127.0.0.1:8000
    SECOM_DB_PATH       SQLite 路徑，預設 data/secom.db
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field

# 平行匯入，這行讓本檔案以子程序啟動時也找得到 src/ 底下的模組。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from lot_db import DEFAULT_DB_PATH, LotRepository
from lot_tools import HttpInferenceClient, LotToolError, LotTools, load_model_info

# stderr 會直接出現在 agent 的 CLI 畫面上：每個 HTTP 請求、每次工具錯誤都
# 印一行會淹沒對話（工具錯誤 agent 自己會顯示），只保留真正的問題。
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.ERROR)

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)

LotIdParam = Annotated[str, Field(description="批次編號，格式為 LOT-001 至 LOT-471，例如 LOT-047")]

mcp = MCPServer(
    "secom-lots",
    instructions="SECOM 半導體產線測試期間（2008-09 ~ 2008-12）的 471 個批次，提供風險分數、SHAP 解釋與人工標記。",
)


@lru_cache(maxsize=1)
def _tools() -> LotTools:
    """第一次呼叫工具時才建立，避免 server 啟動時就因為 DB 或 API 沒準備好而掛掉。"""
    db_path = Path(os.environ.get("SECOM_DB_PATH", str(DEFAULT_DB_PATH)))
    base_url = os.environ.get("SECOM_API_BASE_URL", DEFAULT_API_BASE_URL)
    return LotTools(LotRepository(db_path), HttpInferenceClient(base_url), load_model_info())


def _call(method_name: str, **kwargs) -> dict:
    try:
        return getattr(_tools(), method_name)(**kwargs)
    except (LotToolError, FileNotFoundError, ValueError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(annotations=READ_ONLY)
def list_lots(
    limit: Annotated[int, Field(description="要回傳幾個批次，1 到 20")] = 5,
    order_by: Annotated[str, Field(description="排序方式：risk（風險分數由高到低）或 recent（生產時間由新到舊）")] = "risk",
    above_threshold_only: Annotated[bool, Field(description="true 時只列出超過模型決策閾值的批次")] = False,
) -> dict:
    """列出批次清單與各自的風險分數。當使用者想知道「哪些批次最可疑」「風險最高的是哪幾批」
    「最近生產的批次狀況如何」，或還不知道要查哪個批次時，先用這個工具找出對象，
    再對感興趣的批次呼叫 explain_lot。不回傳 SHAP 原因。"""
    return _call("list_lots", limit=limit, order_by=order_by, above_threshold_only=above_threshold_only)


@mcp.tool(annotations=READ_ONLY)
def get_lot(lot_id: LotIdParam) -> dict:
    """查詢單一批次的生產時間、風險分數、是否超過決策閾值，以及是否已被人工標記。
    使用者只問某個批次「風險多少」「正常嗎」「有沒有被標記過」時用這個；
    若還要知道原因，改用 explain_lot。"""
    return _call("get_lot", lot_id=lot_id)


@mcp.tool(annotations=READ_ONLY)
def explain_lot(lot_id: LotIdParam) -> dict:
    """用 SHAP 解釋單一批次的風險分數：回傳對分數影響最大的 5 個感測器、各自是推高還是降低風險，
    以及模型可信度說明。使用者問「為什麼」「是哪些感測器造成的」「主要原因」時用這個。
    感測器是匿名的，sensor 編號沒有已知的物理意義。"""
    return _call("explain_lot", lot_id=lot_id)


@mcp.tool(annotations=WRITE)
def flag_lot(
    lot_id: LotIdParam,
    reason: Annotated[str, Field(description="標記理由，一句話、300 字以內，應引用風險分數或 SHAP 結果")],
) -> dict:
    """把批次標記為需要人工複檢，寫入資料庫。這是寫入操作，執行前會暫停等待操作者核准。
    只在使用者明確要求標記、且你已經用其他工具取得依據時才呼叫。同一批次不能重複標記。"""
    return _call("flag_lot", lot_id=lot_id, reason=reason)


if __name__ == "__main__":
    mcp.run()
