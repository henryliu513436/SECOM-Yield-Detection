"""展示腳本：從測試集隨機挑幾筆，呼叫正在跑的 API 並印出結果。

只用測試集的**特徵值**當作寫實的示範輸入，不讀取、不顯示、也不拿
Pass/Fail 真實標籤做任何比對——測試集的最終評估是另一件事、還沒被
授權執行，這支腳本純粹是「展示 API 長什麼樣子」，不是評估。

執行前必須先在另一個終端機啟動服務：
    uvicorn src.api:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd

# data.py 用平行匯入，這行讓本檔案不論從哪裡執行都找得到 src/ 底下的模組。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data as D

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
N_DEMO_LOTS = 3
RANDOM_SEED = 42


def pick_demo_rows(n: int = N_DEMO_LOTS) -> pd.DataFrame:
    """從測試集隨機挑 n 筆特徵，只回傳特徵、不帶標籤。"""
    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"
    raw = D.load_raw(csv_path)
    split = D.split_by_time(raw)
    X_test = D.extract_features(split.test)
    return X_test.sample(n=n, random_state=RANDOM_SEED)


def build_payload(row: pd.Series, lot_id: str) -> dict:
    """把一列特徵轉成 API 要的 JSON 格式，缺失值轉成 null。"""
    features = [None if pd.isna(v) else float(v) for v in row]
    return {"lot_id": lot_id, "features": features}


def call_endpoint(client: httpx.Client, base_url: str, path: str, payload: dict) -> dict:
    response = client.post(f"{base_url}{path}", json=payload)
    response.raise_for_status()
    return response.json()


def format_report(lot_id: str, predict_result: dict, explain_result: dict) -> str:
    lines = [
        f"批次 {lot_id}：",
        f"  risk_score = {predict_result['risk_score']:.4f}"
        f"（閾值 {predict_result['threshold']:.4f}，"
        f"判定 {'FAIL' if predict_result['is_fail_predicted'] else 'PASS'}）",
        f"  base_value = {explain_result['base_value']:.4f}",
        "  top 貢獻特徵：",
    ]
    for f in explain_result["top_features"]:
        lines.append(
            f"    {f['feature']:>6}: shap={f['shap_value']:+.4f}  actual={f['actual_value']:.4f}"
        )
    return "\n".join(lines)


def main(base_url: str = DEFAULT_BASE_URL) -> None:
    demo_rows = pick_demo_rows()

    try:
        with httpx.Client(timeout=10.0) as client:
            for index, row in demo_rows.iterrows():
                lot_id = f"test-{index}"
                payload = build_payload(row, lot_id)

                predict_result = call_endpoint(client, base_url, "/predict", payload)
                explain_result = call_endpoint(client, base_url, "/explain", payload)

                print(format_report(lot_id, predict_result, explain_result))
                print()
    except httpx.ConnectError:
        print(f"連不上 {base_url}——請先在另一個終端機執行：uvicorn src.api:app --reload")
        sys.exit(1)


if __name__ == "__main__":
    main()
