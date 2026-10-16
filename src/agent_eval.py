"""Agent 的自動 eval：固定題組 + 期望的工具呼叫序列，對真的 Ollama 實測。

延續舊版 20 題 eval（舊版成功率 17/20），改用新的四個工具與 LOT-XXX
編號，並加入多步題（先列清單再解釋）、錯誤自我修正題與核准題。

判定成功的條件：
- 期望的工具依序出現在實際呼叫序列中（中間可以夾別的呼叫）；每一步可
  以有多個可接受的工具，例如「查單一批次」用 get_lot 或 explain_lot 都算對
- 沒有工具錯誤（刻意測試錯誤處理的題目除外）
- 有產出最終報告，且不是「超過輪數」或「模型呼叫失敗」
- 寫入被拒絕時，報告不得聲稱已經標記（claims_write_done）

寫入題一律自動拒絕（approval.auto_reject_gate），eval 不會改到 DB。
Ollama 沒有固定 seed，成功率是有雜訊的估計值。

用法（需先啟動 FastAPI 與 Ollama）：
    python src/agent.py --evaluate
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from agent import AgentResult, _server_env, run_agent
from approval import auto_reject_gate
from mcp_bridge import connect_mcp_server

LOOKUP = frozenset({"get_lot", "explain_lot"})
EXPLAIN = frozenset({"explain_lot"})
LIST = frozenset({"list_lots"})
FLAG = frozenset({"flag_lot"})
_FAILURE_PREFIXES = ("（超過最大工具呼叫輪數", "（呼叫模型失敗")
# 前面接「未 / 沒 / 不」的是否定句（「尚未成功標記」），不算聲稱完成。
_WRITE_DONE_PATTERNS = (
    re.compile(r"(?<![未沒不])(已經?|成功)被?(成功)?地?標記"),
    re.compile(r"執行了標記"),
    re.compile(r"(?<![未沒不])標記(完成|成功)"),
)


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_steps: tuple[frozenset[str], ...]
    allow_tool_errors: bool = False


EVAL_CASES: tuple[EvalCase, ...] = (
    # 單一批次查詢（沿用舊題組的問法，改成 LOT 編號）
    EvalCase("LOT-001 的風險分數是多少？", (LOOKUP,)),
    EvalCase("批次 6 有沒有異常風險？", (LOOKUP,)),
    EvalCase("幫我看一下第 13 個批次，風險高不高？", (LOOKUP,)),
    EvalCase("LOT-021 為什麼風險比較高，是哪些感測器造成的？", (EXPLAIN,)),
    EvalCase("LOT-034 的主要異常貢獻特徵有哪些？", (EXPLAIN,)),
    EvalCase("批次 48 的風險分數跟主要原因都告訴我。", (EXPLAIN,)),
    EvalCase("查一下 lot 59，完整報告。", (LOOKUP,)),
    EvalCase("LOT-067 正常嗎？", (LOOKUP,)),
    EvalCase("幫我分析 LOT-080：風險分數、判定結果、還有前幾大貢獻因素。", (EXPLAIN,)),
    EvalCase("LOT-111 是不是有問題？", (LOOKUP,)),
    EvalCase("LOT-156 的 SHAP 貢獻特徵是什麼？", (EXPLAIN,)),
    EvalCase("批次 351 風險分數是多少，主要受哪些特徵影響？", (EXPLAIN,)),
    # 清單與多步推理
    EvalCase("目前風險最高的 3 個批次是哪些？", (LIST,)),
    EvalCase("最近生產的 5 個批次風險如何？", (LIST,)),
    EvalCase("有哪些批次超過模型的決策閾值？", (LIST,)),
    EvalCase("風險最高的批次是哪一個？它為什麼被判定為高風險？", (LIST, EXPLAIN)),
    EvalCase("最近 5 批裡面風險最高的是哪批？說明主要原因。", (LIST, EXPLAIN)),
    EvalCase("找出風險最高的批次並解釋原因，如果有問題就把它標記起來。", (LIST, EXPLAIN, FLAG)),
    # 錯誤自我修正與核准
    EvalCase("LOT-999 的風險如何？", (LOOKUP,), allow_tool_errors=True),
    EvalCase("把 LOT-005 標記起來，理由是例行抽查。", (FLAG,)),
)


def _contains_in_order(calls: tuple[str, ...], steps: tuple[frozenset[str], ...]) -> bool:
    remaining = iter(calls)
    return all(any(call in step for call in remaining) for step in steps)


def claims_write_done(report: str) -> bool:
    """報告是否聲稱已經標記。關鍵字比對是粗略的判斷，用來抓「被拒絕後
    仍說已成功標記」這類幻覺（實測：「批次 LOT-225 已被成功標記」）。"""
    return any(pattern.search(report) for pattern in _WRITE_DONE_PATTERNS)


def trial_succeeded(case: EvalCase, result: AgentResult) -> bool:
    if not _contains_in_order(result.tool_calls_made, case.expected_steps):
        return False
    if result.tool_errors and not case.allow_tool_errors:
        return False
    report = result.final_report.strip()
    if result.rejected_calls and claims_write_done(report):
        return False
    return bool(report) and not report.startswith(_FAILURE_PREFIXES)


async def main() -> None:
    n_success = 0
    failures: list[str] = []
    async with connect_mcp_server(_server_env()) as executor:
        for i, case in enumerate(EVAL_CASES, start=1):
            start = time.perf_counter()
            result = await run_agent(case.question, executor, auto_reject_gate())
            elapsed = time.perf_counter() - start
            ok = trial_succeeded(case, result)
            n_success += ok
            status = "成功" if ok else "失敗"
            print(f"[{status}] {i:02d} {elapsed:5.0f}s tools={list(result.tool_calls_made)} q={case.question}", flush=True)
            if not ok:
                failures.append(
                    f"{i:02d} q={case.question!r} tools={list(result.tool_calls_made)} "
                    f"errors={list(result.tool_errors)} report={result.final_report[:80]!r}"
                )

    total = len(EVAL_CASES)
    print()
    print(f"成功率：{n_success}/{total} = {n_success / total:.0%}")
    for failure in failures:
        print(f"  失敗案例：{failure}")
