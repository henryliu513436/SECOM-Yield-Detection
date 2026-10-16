"""SQLite 存取層：批次特徵（lots）、人工標記（flags）、核准稽核紀錄（approval_log）。

DB 只放測試集的批次、而且不存 Pass/Fail 標籤——agent 能查到的只有感測
器特徵與模型分數，不能「偷看答案」。訓練集也不放進來：模型在訓練集上
的分數是 in-sample，展示出來會讓人誤以為模型很準。

每個方法各自開關連線，不共用長連線：MCP server 與 agent 是兩個不同的
程序，都可能寫入同一個檔案，短連線最不容易留下鎖。
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# data.py 用平行匯入，這行讓本檔案不論從哪裡執行都找得到 src/ 底下的模組。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "secom.db"

_SCHEMA = """
CREATE TABLE lots (
    lot_id      TEXT PRIMARY KEY,
    seq         INTEGER NOT NULL UNIQUE,
    produced_at TEXT NOT NULL,
    features    TEXT NOT NULL
);
CREATE TABLE flags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id     TEXT NOT NULL REFERENCES lots(lot_id),
    reason     TEXT NOT NULL,
    flagged_at TEXT NOT NULL
);
CREATE TABLE approval_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    tool       TEXT NOT NULL,
    arguments  TEXT NOT NULL,
    decision   TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
    decided_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class LotRecord:
    lot_id: str
    seq: int
    produced_at: str
    features: tuple[float | None, ...]


@dataclass(frozen=True)
class FlagRecord:
    id: int
    lot_id: str
    reason: str
    flagged_at: str


@dataclass(frozen=True)
class ApprovalRecord:
    id: int
    tool: str
    arguments: dict[str, Any]
    decision: str
    decided_at: str


def format_lot_id(seq: int) -> str:
    """批次編號：依測試集時間順序從 1 開始，LOT-001 ~ LOT-471。"""
    return f"LOT-{seq:03d}"


def records_from_frame(frame: pd.DataFrame) -> list[LotRecord]:
    """把 load_raw 排序過的測試集轉成 LotRecord。只保留感測器欄（標籤由
    D.extract_features 丟掉），缺失值存成 None。"""
    features = D.extract_features(frame)
    times = frame[D.TIME_COLUMN]
    return [
        LotRecord(
            lot_id=format_lot_id(i + 1),
            seq=i + 1,
            produced_at=times.iloc[i].isoformat(),
            features=tuple(None if pd.isna(v) else float(v) for v in features.iloc[i]),
        )
        for i in range(len(frame))
    ]


def create_db(path: Path, lots: Iterable[LotRecord]) -> None:
    """建立新的 DB 並寫入批次。檔案已存在就拒絕——裡面可能有人工標記與
    稽核紀錄，不能被重建 DB 這個動作悄悄洗掉。"""
    if path.exists():
        raise FileExistsError(f"{path} 已存在；若確定要重建，請先手動刪除（會一併刪掉標記與稽核紀錄）")
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript(_SCHEMA)
        conn.executemany(
            "INSERT INTO lots (lot_id, seq, produced_at, features) VALUES (?, ?, ?, ?)",
            [(lot.lot_id, lot.seq, lot.produced_at, json.dumps(list(lot.features))) for lot in lots],
        )


def _row_to_lot(row: tuple) -> LotRecord:
    lot_id, seq, produced_at, features = row
    return LotRecord(lot_id, seq, produced_at, tuple(json.loads(features)))


class LotRepository:
    """lots / flags / approval_log 的讀寫介面。"""

    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {path}，請先執行 `python scripts/build_db.py`")
        self._path = path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with closing(self._connect()) as conn:
            return conn.execute(sql, params).fetchall()

    def _insert(self, sql: str, params: tuple) -> int:
        with closing(self._connect()) as conn, conn:
            return int(conn.execute(sql, params).lastrowid)

    def count(self) -> int:
        return int(self._query("SELECT COUNT(*) FROM lots")[0][0])

    def list_lots(self) -> list[LotRecord]:
        rows = self._query("SELECT lot_id, seq, produced_at, features FROM lots ORDER BY seq")
        return [_row_to_lot(row) for row in rows]

    def get_lot(self, lot_id: str) -> LotRecord | None:
        rows = self._query("SELECT lot_id, seq, produced_at, features FROM lots WHERE lot_id = ?", (lot_id,))
        return _row_to_lot(rows[0]) if rows else None

    def list_flags(self, lot_id: str) -> list[FlagRecord]:
        rows = self._query(
            "SELECT id, lot_id, reason, flagged_at FROM flags WHERE lot_id = ? ORDER BY id", (lot_id,)
        )
        return [FlagRecord(*row) for row in rows]

    def flagged_lot_ids(self) -> set[str]:
        return {row[0] for row in self._query("SELECT DISTINCT lot_id FROM flags")}

    def add_flag(self, lot_id: str, reason: str, flagged_at: str) -> FlagRecord:
        flag_id = self._insert(
            "INSERT INTO flags (lot_id, reason, flagged_at) VALUES (?, ?, ?)", (lot_id, reason, flagged_at)
        )
        return FlagRecord(flag_id, lot_id, reason, flagged_at)

    def record_approval(self, tool: str, arguments: dict[str, Any], decision: str, decided_at: str) -> None:
        self._insert(
            "INSERT INTO approval_log (tool, arguments, decision, decided_at) VALUES (?, ?, ?, ?)",
            (tool, json.dumps(arguments, ensure_ascii=False), decision, decided_at),
        )

    def list_approvals(self) -> list[ApprovalRecord]:
        rows = self._query("SELECT id, tool, arguments, decision, decided_at FROM approval_log ORDER BY id")
        return [ApprovalRecord(id_, tool, json.loads(args), decision, at) for id_, tool, args, decision, at in rows]
