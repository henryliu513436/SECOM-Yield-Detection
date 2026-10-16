"""人工核准關卡：會寫入資料的工具在執行前一律暫停，等操作者 y/n。

被拒絕時不拋例外，而是回一段文字給模型（rejection_message），讓它重新
規劃——例如改成只回報分析結果。每一次核准或拒絕都交給 recorder 寫進
approval_log，「可稽核」要有實際紀錄。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

Recorder = Callable[[str, dict[str, Any], str], None]


def rejection_message(tool_name: str, arguments: dict[str, Any]) -> str:
    args = json.dumps(arguments, ensure_ascii=False)
    return (
        f"使用者拒絕執行 {tool_name}({args})，這個操作沒有被執行，資料庫沒有任何改變。"
        "不要再次嘗試同一個操作。請直接向使用者回報你目前的分析結果與建議，"
        "並在報告中明確說明「這個操作已被拒絕、沒有執行」；絕對不可以寫成已經完成。"
    )


class ApprovalGate:
    """ask/out 可注入：CLI 用 input/print，eval 用自動拒絕，測試用假函式。"""

    def __init__(
        self,
        ask: Callable[[str], str] = input,
        recorder: Recorder | None = None,
        out: Callable[[str], None] = print,
    ) -> None:
        self._ask = ask
        self._recorder = recorder
        self._out = out

    def request(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        self._out(f"⚠ Agent 要執行寫入操作：{tool_name}({json.dumps(arguments, ensure_ascii=False)})")
        try:
            answer = self._ask("核准？ [y/N] ")
        except EOFError:
            answer = ""  # 輸入被關閉（Ctrl-D、非互動環境）一律視為拒絕
        approved = answer.strip().lower() == "y"
        if self._recorder is not None:
            self._recorder(tool_name, arguments, "approved" if approved else "rejected")
        return approved


def auto_reject_gate() -> ApprovalGate:
    """eval 用：不問人、一律拒絕，也不寫 audit log（不是真的人工決定）。"""
    return ApprovalGate(ask=lambda _prompt: "n", recorder=None, out=lambda _s: None)
