"""
tools/base.py — Tool System Foundation

How a Valkyrie acts on the world. Not just text in → text out.
Tools give the bot hands.

Design principles:
  - Simpler than nanobot's Tool ABC (no JSON Schema validation overhead)
  - Workspace-sandboxed by default (bot can't escape its directory)
  - Async execution (matches weave's async breath cycle)
  - OpenAI function-calling compatible schema output

Zero external dependencies.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


# ── data types ───────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """A tool call requested by the LLM."""
    id: str                     # unique call ID (from LLM)
    name: str                   # tool name
    arguments: dict[str, Any]   # parsed arguments


@dataclass
class ToolResult:
    """Result of executing a tool."""
    call_id: str                # matches ToolCall.id
    name: str                   # tool name
    output: str                 # result text (always string for LLM)
    success: bool = True
    duration_ms: float = 0.0


@dataclass
class LLMResponse:
    """Extended LLM response that may contain tool calls.

    Replaces the old str-only return from complete().
    """
    content: str | None = None          # text response (None if only tool calls)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def text(self) -> str:
        """Get text content, empty string if None."""
        return self.content or ""


# ── tool protocol ────────────────────────────────────────────────────

class Tool(Protocol):
    """What a tool looks like. Implement this."""

    @property
    def name(self) -> str:
        """Tool name (used in function calls)."""
        ...

    @property
    def description(self) -> str:
        """What the tool does (shown to LLM)."""
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for parameters."""
        ...

    async def execute(self, **kwargs: Any) -> str:
        """Run the tool. Returns result as string."""
        ...


# ── tool base class (optional convenience) ───────────────────────────

class BaseTool:
    """Convenience base class for tools. Not required — Tool protocol works too."""

    _name: str = ""
    _description: str = ""
    _parameters: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    def to_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ── tool registry ────────────────────────────────────────────────────

class ToolRegistry:
    """Manages available tools and executes them by name.

    Usage:
        registry = ToolRegistry()
        registry.register(ReadFileTool(workspace))
        registry.register(WriteFileTool(workspace))

        # get schemas for LLM
        schemas = registry.get_schemas()

        # execute a tool call
        result = await registry.execute(tool_call)
    """

    def __init__(self):
        self._tools: dict[str, Any] = {}  # name → tool instance

    def register(self, tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        log.debug("Tool registered: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool."""
        self._tools.pop(name, None)

    def get(self, name: str):
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get all tool schemas in OpenAI function-calling format."""
        schemas = []
        for tool in self._tools.values():
            if hasattr(tool, "to_schema"):
                schemas.append(tool.to_schema())
            else:
                # build schema from protocol properties
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                })
        return schemas

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        tool = self._tools.get(call.name)
        if not tool:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                output=f"Error: unknown tool '{call.name}'. Available: {', '.join(self.names)}",
                success=False,
            )

        start = time.monotonic()
        try:
            output = await tool.execute(**call.arguments)
            duration = (time.monotonic() - start) * 1000
            log.info("Tool %s executed in %.0fms", call.name, duration)
            return ToolResult(
                call_id=call.id,
                name=call.name,
                output=output,
                success=True,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            log.warning("Tool %s failed: %s", call.name, e)
            return ToolResult(
                call_id=call.id,
                name=call.name,
                output=f"Error executing {call.name}: {e}",
                success=False,
                duration_ms=duration,
            )

    def describe_for_prompt(self) -> str:
        """Human-readable tool list for system prompt injection."""
        if not self._tools:
            return ""
        lines = ["You have the following tools available:"]
        for tool in self._tools.values():
            lines.append(f"  - **{tool.name}**: {tool.description}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools