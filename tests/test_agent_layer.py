"""Agent 層：MCP ↔ Ollama 格式轉換、核准關卡、多步迴圈與對話記憶。

MCP server 與 Ollama 都用假的物件取代——這裡驗證的是 loop 本身的行為
（派發、核准、拒絕後的處理、記憶），不是模型會不會選對工具；後者由
`python src/agent.py --evaluate` 對真的 Ollama 實測。
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from agent import AgentResult, run_agent, trim_history
from approval import ApprovalGate, rejection_message
from mcp_bridge import ToolOutcome, ToolSpec, result_to_outcome, to_ollama_tool, tool_spec_from_mcp

# ---------- mcp_bridge ----------


def test_tool_spec_is_read_only_only_when_explicitly_marked() -> None:
    read_tool = SimpleNamespace(
        name="list_lots", description="d", input_schema={"type": "object"}, annotations=SimpleNamespace(read_only_hint=True)
    )
    unmarked = SimpleNamespace(name="x", description="d", input_schema={}, annotations=None)
    write_tool = SimpleNamespace(
        name="flag_lot", description="d", input_schema={}, annotations=SimpleNamespace(read_only_hint=False)
    )

    assert tool_spec_from_mcp(read_tool).read_only is True
    assert tool_spec_from_mcp(unmarked).read_only is False  # fail-safe：沒標就當成會寫入
    assert tool_spec_from_mcp(write_tool).read_only is False


def test_to_ollama_tool_uses_function_schema() -> None:
    spec = ToolSpec("get_lot", "查單一批次", {"type": "object", "properties": {"lot_id": {"type": "string"}}}, True)

    tool = to_ollama_tool(spec)

    assert tool == {
        "type": "function",
        "function": {"name": "get_lot", "description": "查單一批次", "parameters": spec.input_schema},
    }


def test_result_to_outcome_joins_text_and_keeps_error_flag() -> None:
    result = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="找不到 LOT-999")], structured_content=None, is_error=True
    )

    assert result_to_outcome(result) == ToolOutcome("找不到 LOT-999", is_error=True)


# ---------- approval ----------


def test_gate_records_both_approval_and_rejection() -> None:
    answers = iter(["y", "n"])
    recorded = []
    gate = ApprovalGate(ask=lambda _prompt: next(answers), recorder=lambda *a: recorded.append(a), out=lambda _s: None)

    assert gate.request("flag_lot", {"lot_id": "LOT-001"}) is True
    assert gate.request("flag_lot", {"lot_id": "LOT-002"}) is False
    assert recorded == [
        ("flag_lot", {"lot_id": "LOT-001"}, "approved"),
        ("flag_lot", {"lot_id": "LOT-002"}, "rejected"),
    ]


def test_gate_treats_eof_as_rejection() -> None:
    def ask(_prompt: str) -> str:
        raise EOFError

    gate = ApprovalGate(ask=ask, recorder=None, out=lambda _s: None)

    assert gate.request("flag_lot", {}) is False


# ---------- agent loop ----------

READ_SPEC = ToolSpec("list_lots", "列出批次", {"type": "object"}, read_only=True)
WRITE_SPEC = ToolSpec("flag_lot", "標記批次", {"type": "object"}, read_only=False)


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self) -> list[ToolSpec]:
        return [READ_SPEC, WRITE_SPEC]

    async def call_tool(self, name: str, arguments: dict) -> ToolOutcome:
        self.calls.append((name, arguments))
        return ToolOutcome(json.dumps({"tool": name, "ok": True}), is_error=False)


def _tool_call(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": "", "thinking": "很長的推理", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def _scripted_llm(*replies: dict):
    """依序回傳預先寫好的 assistant 訊息，並記下每次收到的 messages。"""
    queue = list(replies)
    seen: list[list[dict]] = []

    def call(messages: list[dict], tools: list[dict]) -> dict:
        seen.append([dict(m) for m in messages])
        return queue.pop(0)

    return call, seen


def _gate(answer: str) -> ApprovalGate:
    return ApprovalGate(ask=lambda _p: answer, recorder=None, out=lambda _s: None)


def test_agent_runs_multiple_tool_rounds_then_answers() -> None:
    llm, seen = _scripted_llm(
        _tool_call("list_lots", {"limit": 3}),
        _tool_call("list_lots", {"limit": 1}),
        {"role": "assistant", "content": "LOT-002 風險最高"},
    )
    executor = FakeExecutor()

    result = asyncio.run(run_agent("哪批最可疑？", executor, _gate("n"), call_llm=llm))

    assert result.final_report == "LOT-002 風險最高"
    assert result.tool_calls_made == ("list_lots", "list_lots")
    assert result.rounds == 3
    assert [c[0] for c in executor.calls] == ["list_lots", "list_lots"]
    assert seen[1][-1]["role"] == "tool"  # 第二輪看得到第一輪的工具結果


def test_rejected_write_is_not_executed_and_model_is_told() -> None:
    llm, seen = _scripted_llm(
        _tool_call("flag_lot", {"lot_id": "LOT-002", "reason": "高風險"}),
        {"role": "assistant", "content": "已改為只回報分析"},
    )
    executor = FakeExecutor()

    result = asyncio.run(run_agent("標記 LOT-002", executor, _gate("n"), call_llm=llm))

    assert executor.calls == []
    assert result.rejected_calls == ("flag_lot",)
    assert seen[1][-1]["content"] == rejection_message("flag_lot", {"lot_id": "LOT-002", "reason": "高風險"})
    assert result.final_report == "已改為只回報分析"


def test_approved_write_is_executed() -> None:
    llm, _ = _scripted_llm(
        _tool_call("flag_lot", {"lot_id": "LOT-002", "reason": "高風險"}),
        {"role": "assistant", "content": "已標記"},
    )
    executor = FakeExecutor()

    asyncio.run(run_agent("標記 LOT-002", executor, _gate("y"), call_llm=llm))

    assert executor.calls == [("flag_lot", {"lot_id": "LOT-002", "reason": "高風險"})]


def test_rejected_write_is_reported_by_code_even_if_model_claims_success() -> None:
    # 重現 eval 實測到的幻覺：被拒絕後模型仍說「已被成功標記」
    llm, _ = _scripted_llm(
        _tool_call("flag_lot", {"lot_id": "LOT-225", "reason": "高風險"}),
        {"role": "assistant", "content": "批次 LOT-225 已被成功標記為需要人工複檢。"},
    )

    result = asyncio.run(run_agent("標記 LOT-225", FakeExecutor(), _gate("n"), call_llm=llm))

    assert len(result.write_notes) == 1
    assert "操作者拒絕，沒有執行" in result.write_notes[0]
    assert "LOT-225" in result.write_notes[0]
    assert result.display_report().endswith(result.write_notes[0])
    assert "以此為準" in result.display_report()


def test_approved_write_note_and_read_only_calls_add_no_note() -> None:
    llm, _ = _scripted_llm(
        _tool_call("list_lots", {"limit": 1}),
        _tool_call("flag_lot", {"lot_id": "LOT-002", "reason": "高風險"}),
        {"role": "assistant", "content": "完成"},
    )

    result = asyncio.run(run_agent("q", FakeExecutor(), _gate("y"), call_llm=llm))

    assert len(result.write_notes) == 1
    assert result.write_notes[0].startswith("✓ flag_lot")
    assert "已核准並寫入資料庫" in result.write_notes[0]


def test_display_report_without_writes_is_just_the_report() -> None:
    assert AgentResult("LOT-002 風險最高").display_report() == "LOT-002 風險最高"


def test_unknown_tool_and_string_arguments_are_handled() -> None:
    llm, seen = _scripted_llm(
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "drop_table", "arguments": {}}}]},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "list_lots", "arguments": '{"limit": 2}'}}]},
        {"role": "assistant", "content": "完成"},
    )
    executor = FakeExecutor()

    result = asyncio.run(run_agent("q", executor, _gate("n"), call_llm=llm))

    assert "未知工具" in seen[1][-1]["content"]
    assert executor.calls == [("list_lots", {"limit": 2})]
    assert result.tool_errors == ("drop_table: 未知工具",)


def test_history_carries_into_next_question_without_thinking() -> None:
    llm, seen = _scripted_llm(
        _tool_call("list_lots", {"limit": 3}),
        {"role": "assistant", "content": "最高的是 LOT-002"},
        {"role": "assistant", "content": "第二高的是 LOT-003"},
    )
    executor = FakeExecutor()

    first = asyncio.run(run_agent("哪批最可疑？", executor, _gate("n"), call_llm=llm))
    second = asyncio.run(run_agent("那第二高的呢？", executor, _gate("n"), history=first.history, call_llm=llm))

    sent = seen[2]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert all("thinking" not in m for m in sent)
    assert second.final_report == "第二高的是 LOT-003"


def test_round_limit_returns_explicit_message() -> None:
    llm, _ = _scripted_llm(*[_tool_call("list_lots", {}) for _ in range(20)])

    result = asyncio.run(run_agent("q", FakeExecutor(), _gate("n"), call_llm=llm, max_rounds=2))

    assert result.final_report.startswith("（超過最大工具呼叫輪數")


def test_trim_history_keeps_whole_recent_turns() -> None:
    history = (
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "a2"},
    )

    assert trim_history(history, max_turns=1) == history[2:]
    assert trim_history(history, max_turns=5) == history


def test_agent_result_is_immutable() -> None:
    result = AgentResult(final_report="x")

    with pytest.raises(AttributeError):
        result.final_report = "y"  # type: ignore[misc]


# ---------- eval 判定 ----------


def test_eval_requires_expected_tools_in_order() -> None:
    from agent_eval import EvalCase, trial_succeeded

    case = EvalCase("q", (frozenset({"list_lots"}), frozenset({"explain_lot"})))
    ok = AgentResult("報告", tool_calls_made=("list_lots", "get_lot", "explain_lot"))
    wrong_order = AgentResult("報告", tool_calls_made=("explain_lot", "list_lots"))
    no_tools = AgentResult("我猜是 LOT-002")

    assert trial_succeeded(case, ok) is True
    assert trial_succeeded(case, wrong_order) is False
    assert trial_succeeded(case, no_tools) is False


def test_eval_alternatives_and_error_policy() -> None:
    from agent_eval import EvalCase, trial_succeeded

    any_lookup = EvalCase("q", (frozenset({"get_lot", "explain_lot"}),))
    error_expected = EvalCase("q", (frozenset({"get_lot", "explain_lot"}),), allow_tool_errors=True)
    errored = AgentResult("查無此批次", tool_calls_made=("get_lot",), tool_errors=("get_lot: 找不到",))

    assert trial_succeeded(any_lookup, AgentResult("r", tool_calls_made=("explain_lot",))) is True
    assert trial_succeeded(any_lookup, errored) is False
    assert trial_succeeded(error_expected, errored) is True
    assert trial_succeeded(any_lookup, AgentResult("（超過最大工具呼叫輪數，未能產出最終報告）", ("get_lot",))) is False


@pytest.mark.parametrize(
    "report, expected",
    [
        ("批次 LOT-225 已被成功標記為需要人工複檢。", True),  # eval 實測的原句
        ("我已經執行了標記作業。", True),
        ("LOT-225 已標記。", True),
        ("標記完成。", True),
        ("標記已被拒絕、沒有執行。", False),
        ("LOT-225 尚未成功標記，請人工確認。", False),
        ("LOT-225 沒有被標記。", False),
        ("建議將 LOT-225 標記為需要複檢。", False),
    ],
)
def test_claims_write_done(report: str, expected: bool) -> None:
    from agent_eval import claims_write_done

    assert claims_write_done(report) is expected


def test_eval_fails_when_report_claims_a_rejected_write() -> None:
    from agent_eval import EvalCase, trial_succeeded

    case = EvalCase("q", (frozenset({"flag_lot"}),))
    lying = AgentResult("LOT-225 已被成功標記。", tool_calls_made=("flag_lot",), rejected_calls=("flag_lot",))
    honest = AgentResult("標記已被拒絕，沒有執行。", tool_calls_made=("flag_lot",), rejected_calls=("flag_lot",))

    assert trial_succeeded(case, lying) is False
    assert trial_succeeded(case, honest) is True
