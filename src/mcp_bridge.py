"""MCP ↔ Ollama 的格式轉換，以及透過 MCP 執行工具的 executor。

Ollama 不是 MCP client：它只吃 OpenAI 風格的 `tools` 清單、回傳 JSON
形式的 tool_calls，由呼叫端自己派發。所以 agent 要當中間人——從 MCP
server 拿工具清單 → 轉成 Ollama 格式 → 解析 tool_calls → 透過 MCP 執行。

轉換函式只讀物件屬性、不 import mcp，測試時可以用簡單的假物件；只有
connect_mcp_server 真正需要 mcp 套件，所以在函式內延遲匯入。
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

MCP_SERVER_SCRIPT = Path(__file__).resolve().parent / "mcp_server.py"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool


@dataclass(frozen=True)
class ToolOutcome:
    text: str
    is_error: bool


class ToolExecutor(Protocol):
    async def list_tools(self) -> list[ToolSpec]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolOutcome: ...


def tool_spec_from_mcp(tool: Any) -> ToolSpec:
    """MCP Tool → ToolSpec。

    只有明確標記 read_only_hint=True 的工具才算唯讀；沒標記的一律當成
    可能寫入、需要核准。MCP 規範本身的預設也是「不保證唯讀」，fail-safe
    比較安全：新增工具時忘了標，最壞是多問一次，而不是讓 AI 自動寫入。
    """
    annotations = getattr(tool, "annotations", None)
    read_only = getattr(annotations, "read_only_hint", None) is True if annotations is not None else False
    return ToolSpec(
        name=tool.name,
        description=tool.description or "",
        input_schema=dict(tool.input_schema or {}),
        read_only=read_only,
    )


def to_ollama_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": spec.name, "description": spec.description, "parameters": spec.input_schema},
    }


def result_to_outcome(result: Any) -> ToolOutcome:
    """MCP CallToolResult → 要回給模型的文字。優先用 text content，沒有
    文字時才退回 structured_content 的 JSON。"""
    texts = [block.text for block in (result.content or []) if getattr(block, "type", None) == "text"]
    text = "\n".join(texts)
    if not text and getattr(result, "structured_content", None) is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False)
    return ToolOutcome(text=text, is_error=bool(getattr(result, "is_error", False)))


class McpToolExecutor:
    """包住一個已連線的 mcp.Client。"""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def list_tools(self) -> list[ToolSpec]:
        result = await self._client.list_tools()
        return [tool_spec_from_mcp(tool) for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        return result_to_outcome(await self._client.call_tool(name, arguments))


@asynccontextmanager
async def connect_mcp_server(env: dict[str, str] | None = None) -> AsyncIterator[McpToolExecutor]:
    """以 stdio 子程序啟動 src/mcp_server.py 並連線。子程序用同一個 Python
    直譯器，確保跟 agent 在同一個環境裡。"""
    from mcp import Client, StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_SCRIPT)],
        env=env,
        cwd=str(MCP_SERVER_SCRIPT.parents[1]),
    )
    async with Client(params) as client:
        yield McpToolExecutor(client)
