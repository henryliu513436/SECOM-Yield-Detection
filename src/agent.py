"""LLM agent 層：使用者提問驅動，LLM 依前一步結果自主決定下一步要呼叫
哪個工具，直到它認為資訊足夠才回答。

- 工具來自 MCP server（src/mcp_server.py，stdio 子程序），這裡透過
  mcp_bridge 把它們轉成 Ollama 的 tools 格式並派發呼叫。
- 會寫入資料的工具（MCP annotations 沒標 read_only_hint=True 的）執行前
  一律經過 approval.ApprovalGate 人工核准。
- 對話記憶：每次 run_agent 回傳更新後的 history，下一題帶進去即可追問。

LLM 不做任何計算或異常判斷，只做工具調度與結果翻譯——風險分數、SHAP
貢獻這些數字全部來自工具實際呼叫 FastAPI 的回傳值。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# 平行匯入，這行讓本檔案不論從哪裡執行都找得到 src/ 底下的模組。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from approval import ApprovalGate, rejection_message
from mcp_bridge import ToolExecutor, ToolSpec, to_ollama_tool

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
# 多步題到後面幾輪時，context 已經包含清單 + SHAP 結果，thinking 模式單次
# 回應實測會超過 180 秒（eval 第 18 題兩次都逾時），所以放寬到 300 秒。
OLLAMA_TIMEOUT_SECONDS = 300.0
OLLAMA_MAX_ATTEMPTS = 2
# Ollama 預設的 context 很短，超過時會無聲截斷前文；工具定義 + 對話記憶
# + SHAP 回傳很容易超過，所以明確指定。
OLLAMA_NUM_CTX = 16384
MAX_TOOL_ROUNDS = 8
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = (
    "你是 SECOM 半導體產線的良率診斷助理，用繁體中文回答。\n"
    "規則：\n"
    "1. 你不能自己計算風險分數或 SHAP 貢獻，也不能判斷批次是否異常——所有數字都必須來自工具回傳，"
    "絕對不要編造或估計數字。\n"
    "2. 必要時可以連續呼叫多個工具：例如先用 list_lots 找出可疑批次，再用 explain_lot 查原因。"
    "下一步要查什麼，請根據上一步的結果決定。\n"
    "3. 感測器是匿名的，sensor 編號沒有已知的物理意義，不要猜測它代表溫度、壓力等。\n"
    "4. 模型可信度有限，風險分數只能用來排序。回報時要說明這是模型的判斷，不是確定的製程根因。\n"
    "5. 只有在使用者要求、且有工具結果作為依據時，才呼叫 flag_lot 標記批次。"
    "若使用者拒絕執行，不要重試，改為回報分析結果，並明確寫出「標記已被拒絕、沒有執行」；"
    "只有工具實際回傳成功時，才可以說已經標記。\n"
    "6. 工具回傳錯誤時，閱讀錯誤訊息並修正參數後再試。"
)

LlmCall = Callable[[list[dict], list[dict]], dict]
EventSink = Callable[[str], None]


@dataclass(frozen=True)
class AgentResult:
    """一次提問的完整結果。history 是不含 system prompt 的對話紀錄，下一題
    傳回 run_agent 就能延續對話。"""

    final_report: str
    tool_calls_made: tuple[str, ...] = ()
    tool_errors: tuple[str, ...] = ()
    rejected_calls: tuple[str, ...] = ()
    rounds: int = 0
    history: tuple[dict, ...] = ()
    # 每個寫入操作實際的結果，由程式產生。小模型被拒絕後仍可能在報告裡
    # 寫「已成功標記」（eval 實測發生過），所以寫入結果不能只靠模型轉述。
    write_notes: tuple[str, ...] = ()

    def display_report(self) -> str:
        """給人看的版本：模型的報告 + 程式產生的寫入紀錄。"""
        if not self.write_notes:
            return self.final_report
        notes = "\n".join(self.write_notes)
        return f"{self.final_report}\n\n【系統紀錄：寫入操作的實際結果（由程式產生，以此為準）】\n{notes}"


@dataclass(frozen=True)
class _Dispatched:
    """單一 tool_call 的處理結果。write_note 只有寫入工具才有。"""

    name: str
    content: str
    error: str | None = None
    rejected: bool = False
    write_note: str | None = None


def trim_history(history: tuple[dict, ...], max_turns: int = MAX_HISTORY_TURNS) -> tuple[dict, ...]:
    """只保留最近 max_turns 個「使用者提問」開始的完整回合。

    必須以 user 訊息為切點：如果從中間切，可能留下沒有對應 tool_calls
    的 tool 訊息，對話結構就壞了。
    """
    user_indices = [i for i, message in enumerate(history) if message.get("role") == "user"]
    if len(user_indices) <= max_turns:
        return history
    return history[user_indices[-max_turns]:]


def _call_ollama(messages: list[dict], tools: list[dict]) -> dict:
    """故意不加 think=False。

    實測過：關掉思考雖然能把單次回應從數十秒壓到約 1 秒，但 20 次測試
    的成功率會從 85% 崩到 30%——模型少了推理空間，會對簡單的批次編號
    映射開始猶豫、反過來問使用者，而不是直接呼叫工具。寧可慢，也要保留
    思考。

    逾時重試 OLLAMA_MAX_ATTEMPTS 次：本地單一推論槽偶爾會在請求當下被
    佔用，逾時多半是暫態的排隊問題，重打一次往往就過了。
    """
    last_exc: httpx.HTTPError | None = None
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": {"num_ctx": OLLAMA_NUM_CTX},
    }
    for attempt in range(OLLAMA_MAX_ATTEMPTS):
        try:
            response = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()["message"]
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt + 1 < OLLAMA_MAX_ATTEMPTS:
                print(f"  [重試] ollama 呼叫逾時/失敗（第 {attempt + 1} 次），重試中...", file=sys.stderr)
    raise last_exc


def _parse_arguments(raw: Any) -> dict:
    """Ollama 原生 API 給 dict；若模型吐的是 JSON 字串也接受。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw) if raw.strip() else {}
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"工具參數必須是 JSON 物件，收到 {raw!r}")


def _for_history(message: dict) -> dict:
    """存進記憶前拿掉 thinking：推理過程很長，帶進下一輪只會擠掉 context。"""
    return {key: value for key, value in message.items() if key != "thinking"}


async def _dispatch(
    call: dict, specs: dict[str, ToolSpec], executor: ToolExecutor, gate: ApprovalGate, emit: EventSink
) -> _Dispatched:
    """執行單一 tool_call。任何失敗都轉成文字回給模型，不讓單一呼叫的問題
    炸掉整個迴圈。"""
    try:
        name = call["function"]["name"]
        arguments = _parse_arguments(call["function"].get("arguments", {}))
    except (KeyError, TypeError, ValueError) as exc:
        return _Dispatched("?", json.dumps({"error": f"工具呼叫格式異常：{exc}"}, ensure_ascii=False), f"格式異常：{exc}")

    if name not in specs:
        available = "、".join(specs)
        return _Dispatched(name, f"未知工具：{name}。可用的工具只有：{available}", f"{name}: 未知工具")

    call_text = f"{name}({json.dumps(arguments, ensure_ascii=False)})"
    emit(f"→ 呼叫 {call_text}")
    is_write = not specs[name].read_only
    if is_write and not gate.request(name, arguments):
        emit("✗ 操作者拒絕")
        note = f"✗ {call_text}：操作者拒絕，沒有執行，資料庫未變更"
        return _Dispatched(name, rejection_message(name, arguments), rejected=True, write_note=note)

    outcome = await executor.call_tool(name, arguments)
    if outcome.is_error:
        emit(f"✗ 工具回報錯誤：{outcome.text}")
        note = f"✗ {call_text}：已核准但執行失敗（{outcome.text}），資料庫未變更" if is_write else None
        return _Dispatched(name, outcome.text, f"{name}({arguments}): {outcome.text}", write_note=note)
    note = f"✓ {call_text}：已核准並寫入資料庫" if is_write else None
    return _Dispatched(name, outcome.text, write_note=note)


async def run_agent(
    question: str,
    executor: ToolExecutor,
    gate: ApprovalGate,
    history: tuple[dict, ...] = (),
    call_llm: LlmCall = _call_ollama,
    max_rounds: int = MAX_TOOL_ROUNDS,
    emit: EventSink = lambda _s: None,
) -> AgentResult:
    """跑一次完整的多步工具呼叫迴圈：LLM 決定呼叫哪些工具、呼叫幾次，直到
    它產出最終文字回答，或超過 max_rounds 為止。"""
    specs = {spec.name: spec for spec in await executor.list_tools()}
    tools = [to_ollama_tool(spec) for spec in specs.values()]
    turn: list[dict] = [{"role": "user", "content": question}]
    calls_made: list[str] = []
    errors: list[str] = []
    rejected: list[str] = []
    write_notes: list[str] = []

    def result(report: str, rounds: int, keep_turn: bool = True) -> AgentResult:
        new_history = trim_history(history + tuple(turn)) if keep_turn else history
        return AgentResult(
            report, tuple(calls_made), tuple(errors), tuple(rejected), rounds, new_history, tuple(write_notes)
        )

    for round_num in range(1, max_rounds + 1):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, *turn]
        try:
            message = await asyncio.to_thread(call_llm, messages, tools)
        except httpx.HTTPError as exc:
            # 失敗的這一題不寫進記憶，避免下一題帶著半截對話。
            errors.append(f"ollama 呼叫失敗：{exc}")
            return result(f"（呼叫模型失敗：{exc}）", round_num, keep_turn=False)
        turn.append(_for_history(message))

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return result(message.get("content", ""), round_num)

        for call in tool_calls:
            outcome = await _dispatch(call, specs, executor, gate, emit)
            calls_made.append(outcome.name)
            if outcome.error is not None:
                errors.append(outcome.error)
            if outcome.rejected:
                rejected.append(outcome.name)
            if outcome.write_note is not None:
                write_notes.append(outcome.write_note)
            turn.append({"role": "tool", "tool_name": outcome.name, "content": outcome.content})

    return result("（超過最大工具呼叫輪數，未能產出最終報告）", max_rounds)


def _server_env() -> dict[str, str]:
    """傳給 MCP server 子程序的設定；沒設定的就用 mcp_server.py 的預設值。"""
    keys = ("SECOM_API_BASE_URL", "SECOM_DB_PATH")
    return {key: os.environ[key] for key in keys if key in os.environ}


async def interactive_main() -> None:
    from lot_db import LotRepository
    from lot_tools import utc_now
    from mcp_bridge import connect_mcp_server

    repo = LotRepository(Path(os.environ["SECOM_DB_PATH"])) if "SECOM_DB_PATH" in os.environ else LotRepository()
    gate = ApprovalGate(
        ask=input,
        recorder=lambda tool, args, decision: repo.record_approval(tool, args, decision, decided_at=utc_now()),
    )
    history: tuple[dict, ...] = ()

    async with connect_mcp_server(_server_env()) as executor:
        print(f"SECOM 良率診斷助理（Ollama: {OLLAMA_MODEL}）。輸入問題；/reset 清除對話記憶；exit 離開。")
        while True:
            try:
                question = (await asyncio.to_thread(input, "> ")).strip()
            except EOFError:
                break
            if question.lower() in {"exit", "quit"}:
                break
            if question == "/reset":
                history = ()
                print("（已清除對話記憶）")
                continue
            if not question:
                continue
            result = await run_agent(question, executor, gate, history=history, emit=print)
            history = result.history
            print(result.display_report())
            print()


if __name__ == "__main__":
    if "--evaluate" in sys.argv:
        from agent_eval import main as evaluate_main

        asyncio.run(evaluate_main())
    else:
        asyncio.run(interactive_main())
