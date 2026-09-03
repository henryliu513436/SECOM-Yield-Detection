"""LLM agent 層：使用者提問驅動，LLM 自主決定呼叫工具的順序，最後產出
文字報告。

LLM 不做任何計算或異常判斷，只做工具調度與結果翻譯——risk_score、
base_value、SHAP 貢獻這些數字全部來自 get_risk / get_contributors 兩個
工具實際呼叫本專案 FastAPI 的回傳值，不是模型自己算出來的。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
import pandas as pd

# data.py 用平行匯入，這行讓本檔案不論從哪裡執行都找得到 src/ 底下的模組。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e4b"
API_BASE_URL = "http://127.0.0.1:8000"
MAX_TOOL_ROUNDS = 5
OLLAMA_TIMEOUT_SECONDS = 180.0

SYSTEM_PROMPT = (
    "你是 SECOM 產線良率異常偵測系統的助理。你不能自己計算風險分數、"
    "SHAP 貢獻度，或判斷某個批次是否異常——這些數字只能透過 get_risk 跟"
    "get_contributors 這兩個工具取得。回答使用者之前，先用工具拿到需要"
    "的數字，再根據工具回傳的結果寫報告；絕對不要自己編造或估計數字。"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_risk",
            "description": "取得指定批次（lot）的風險分數與模型判定結果",
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_index": {
                        "type": "integer",
                        "description": "批次在測試集裡的索引，從 0 開始",
                    }
                },
                "required": ["lot_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contributors",
            "description": "取得指定批次的 SHAP 解釋：base_value、風險分數、top-5 貢獻特徵",
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_index": {
                        "type": "integer",
                        "description": "批次在測試集裡的索引，從 0 開始",
                    }
                },
                "required": ["lot_index"],
            },
        },
    },
]


def load_test_features() -> pd.DataFrame:
    """只讀測試集的特徵，不讀標籤——這裡的用途是給工具查詢用的批次資料。"""
    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"
    raw = D.load_raw(csv_path)
    split = D.split_by_time(raw)
    return D.extract_features(split.test).reset_index(drop=True)


def _build_payload(row: pd.Series, lot_index: int) -> dict:
    features = [None if pd.isna(v) else float(v) for v in row]
    return {"lot_id": f"test-{lot_index}", "features": features}


def _validate_lot_index(lot_index: int, test_features: pd.DataFrame) -> int:
    """擋掉超出範圍的索引，尤其是負數。

    pandas 的 .iloc 對負數索引是「從尾端數」的合法語意（.iloc[-1] 是最
    後一筆），不會報錯——如果不擋，lot_index=-1 這種值會靜默回傳錯的
    批次，而不是明確失敗。工具的 JSON schema 只宣告 type: integer，沒
    有 minimum，所以這裡的檢查是唯一的防線。
    """
    lot_index = int(lot_index)
    if not 0 <= lot_index < len(test_features):
        raise ValueError(f"lot_index 超出範圍：{lot_index}（有效範圍 0~{len(test_features) - 1}）")
    return lot_index


def get_risk(lot_index: int, test_features: pd.DataFrame, base_url: str = API_BASE_URL) -> dict:
    """呼叫 POST /predict，回傳風險分數與判定結果。"""
    lot_index = _validate_lot_index(lot_index, test_features)
    row = test_features.iloc[lot_index]
    payload = _build_payload(row, lot_index)
    response = httpx.post(f"{base_url}/predict", json=payload, timeout=10.0)
    response.raise_for_status()
    return response.json()


def get_contributors(lot_index: int, test_features: pd.DataFrame, base_url: str = API_BASE_URL) -> dict:
    """呼叫 POST /explain，回傳 SHAP 解釋。"""
    lot_index = _validate_lot_index(lot_index, test_features)
    row = test_features.iloc[lot_index]
    payload = _build_payload(row, lot_index)
    response = httpx.post(f"{base_url}/explain", json=payload, timeout=10.0)
    response.raise_for_status()
    return response.json()


@dataclass
class AgentResult:
    """一次完整互動的結果：最終報告文字，以及過程中呼叫了哪些工具。"""

    final_report: str
    tool_calls_made: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    rounds: int = 0


def _make_tool_functions(
    test_features: pd.DataFrame, base_url: str
) -> dict[str, Callable[[int], dict]]:
    return {
        "get_risk": lambda lot_index: get_risk(lot_index, test_features, base_url),
        "get_contributors": lambda lot_index: get_contributors(lot_index, test_features, base_url),
    }


def _call_ollama(messages: list[dict]) -> dict:
    """故意不加 think=False。

    實測過：關掉思考雖然能把單次回應從數十秒壓到約 1 秒，但 20 次測試
    的成功率會從 85% 崩到 30%——模型會因為少了推理空間，對「使用者說
    『批次47』該不該直接對應 lot_index=47」這種簡單映射開始猶豫、反過
    來問使用者要索引，而不是直接呼叫工具。寧可慢，也要保留思考。
    """
    response = httpx.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "messages": messages, "tools": TOOLS, "stream": False},
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["message"]


def run_agent(
    question: str, test_features: pd.DataFrame, base_url: str = API_BASE_URL
) -> AgentResult:
    """跑一次完整的工具呼叫迴圈：LLM 決定呼叫哪些工具、呼叫幾次，直到它
    產出最終文字回答，或超過 MAX_TOOL_ROUNDS 為止。
    """
    tool_funcs = _make_tool_functions(test_features, base_url)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    result = AgentResult(final_report="")

    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        result.rounds = round_num
        try:
            message = _call_ollama(messages)
        except httpx.HTTPError as exc:
            # Ollama 逾時/連線失敗算這次互動失敗，不能讓整個評估流程（或
            # 其他使用者的請求）被一次慢回應拖垮。
            result.tool_errors.append(f"ollama 呼叫失敗：{exc}")
            result.final_report = f"（呼叫模型失敗：{exc}）"
            return result
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            result.final_report = message.get("content", "")
            return result

        for call in tool_calls:
            try:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
            except (KeyError, TypeError) as exc:
                # Ollama 回傳的 tool_calls 結構不完整（跟逾時是同一類「不
                # 能讓一次失敗拖垮整個迴圈」的問題），記一筆錯誤、跳過這
                # 個工具呼叫，不要讓 KeyError 往外炸掉整個 run_agent。
                result.tool_errors.append(f"工具呼叫格式異常：{exc}")
                messages.append(
                    {"role": "tool", "content": json.dumps({"error": str(exc)}, ensure_ascii=False)}
                )
                continue

            result.tool_calls_made.append(name)

            if name not in tool_funcs:
                tool_result = {"error": f"未知工具：{name}"}
                result.tool_errors.append(f"{name}: 未知工具")
            else:
                try:
                    tool_result = tool_funcs[name](**args)
                except Exception as exc:  # noqa: BLE001 - 工具失敗要回報給 LLM，不是讓整個流程崩潰
                    tool_result = {"error": str(exc)}
                    result.tool_errors.append(f"{name}({args}): {exc}")

            messages.append(
                {"role": "tool", "name": name, "content": json.dumps(tool_result, ensure_ascii=False)}
            )

    result.final_report = "（超過最大工具呼叫輪數，未能產出最終報告）"
    return result


def interactive_main(base_url: str = API_BASE_URL) -> None:
    test_features = load_test_features()
    print(f"SECOM 風險評估助理（Ollama: {OLLAMA_MODEL}）。輸入問題，或輸入 exit 離開。")
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        result = run_agent(question, test_features, base_url)
        print(result.final_report)
        print()


EVAL_QUESTIONS = [
    (0, "批次 0 的風險評分是多少？"),
    (5, "批次 5 有沒有異常風險？"),
    (12, "幫我看一下第 12 個批次，風險高不高？"),
    (20, "批次 20 為什麼風險比較高，是哪些感測器造成的？"),
    (33, "第 33 筆的主要異常貢獻特徵有哪些？"),
    (47, "批次 47 的風險分數跟主要原因都告訴我。"),
    (58, "查一下 lot 58，完整報告。"),
    (66, "批次 66 正常嗎？"),
    (79, "幫我分析批次 79：風險分數、判定結果、還有前幾大貢獻因素。"),
    (85, "第 85 個批次可疑嗎？請說明理由。"),
    (99, "批次 99 的 SHAP 貢獻特徵是什麼？"),
    (110, "看一下批次 110 是不是有問題。"),
    (123, "批次 123 風險評分？"),
    (140, "幫我完整評估批次 140，包含風險分數與原因分析。"),
    (155, "第 155 筆資料的異常程度如何？"),
    (168, "批次 168 有沒有被判定為異常？為什麼？"),
    (200, "查詢批次 200 的風險與貢獻因素。"),
    (250, "批次 250 的狀況怎麼樣？"),
    (300, "幫我看批次 300，完整分析一下。"),
    (350, "批次 350 風險分數是多少，主要受哪些特徵影響？"),
]


@dataclass
class EvaluationSummary:
    """20 次實測的彙總結果，成功率記進 README。"""

    n_trials: int
    n_success: int
    failures: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_trials if self.n_trials else float("nan")


def _trial_succeeded(result: AgentResult) -> bool:
    """成功的定義：至少呼叫過一次工具、工具沒有出錯、且產出了非空的最終報告。

    LLM 不做計算，所以「完全沒呼叫工具就回答」一定算失敗——那代表它在
    編數字，不是在調度工具。
    """
    if not result.tool_calls_made:
        return False
    if result.tool_errors:
        return False
    if not result.final_report or result.final_report.startswith("（超過最大工具呼叫輪數"):
        return False
    return True


def evaluate_tool_calling(
    test_features: pd.DataFrame, base_url: str = API_BASE_URL
) -> EvaluationSummary:
    """實測 EVAL_QUESTIONS 這 20 個問題，記錄 tool calling 的成功率。"""
    summary = EvaluationSummary(n_trials=len(EVAL_QUESTIONS), n_success=0)

    for lot_index, question in EVAL_QUESTIONS:
        result = run_agent(question, test_features, base_url)
        if _trial_succeeded(result):
            summary.n_success += 1
            status = "成功"
        else:
            status = "失敗"
            summary.failures.append(
                f"lot={lot_index} q={question!r} tools={result.tool_calls_made} "
                f"errors={result.tool_errors} report={result.final_report[:60]!r}"
            )
        print(f"[{status}] lot={lot_index} tools={result.tool_calls_made} rounds={result.rounds}")

    return summary


if __name__ == "__main__":
    if "--evaluate" in sys.argv:
        summary = evaluate_tool_calling(load_test_features())
        print()
        print(f"成功率：{summary.n_success}/{summary.n_trials} = {summary.success_rate:.0%}")
        for failure in summary.failures:
            print(f"  失敗案例：{failure}")
    else:
        interactive_main()
