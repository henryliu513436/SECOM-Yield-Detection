"""把 SECOM 測試集（時間切分後的最後 30%，471 筆）寫進 data/secom.db，
讓 agent 的工具能依 lot_id 查詢批次。

SECOM 原始資料沒有批次編號，這裡依時間順序合成 LOT-001 ~ LOT-471。只
寫感測器特徵與生產時間，不寫 Pass/Fail 標籤，也不寫訓練集——理由見
src/lot_db.py 開頭。

用法：
    python scripts/build_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# data.py/lot_db.py 用平行匯入，這行讓本檔案不論從專案根目錄或從 scripts/
# 本身執行都找得到 src/ 底下的模組。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data as D
from lot_db import DEFAULT_DB_PATH, create_db, records_from_frame


def main() -> None:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"
    split = D.split_by_time(D.load_raw(csv_path))
    records = records_from_frame(split.test)

    create_db(DEFAULT_DB_PATH, records)
    print(f"已寫入 {DEFAULT_DB_PATH}：{len(records)} 筆批次（{records[0].lot_id} ~ {records[-1].lot_id}）")
    print(f"生產時間範圍：{records[0].produced_at} ~ {records[-1].produced_at}")


if __name__ == "__main__":
    main()
