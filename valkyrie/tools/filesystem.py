"""
tools/filesystem.py — Filesystem Tools

The bot's hands. Read, write, edit files within its workspace.
Sandboxed by default — can't escape the workspace directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from valkyrie.tools.base import BaseTool


def _safe_resolve(path: str, workspace: Path) -> Path:
    """Resolve path and enforce workspace boundary."""
    # handle relative and absolute paths
    p = Path(path)
    if not p.is_absolute():
        p = workspace / p
    resolved = p.resolve()

    # enforce sandbox
    ws_resolved = workspace.resolve()
    if not str(resolved).startswith(str(ws_resolved)):
        raise PermissionError(
            f"Access denied: {path} is outside workspace ({workspace})"
        )
    return resolved


class ReadFileTool(BaseTool):
    """Read file contents."""

    _name = "read_file"
    _description = "Read the contents of a file. Path is relative to workspace."
    _parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to read (relative to workspace)",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Path):
        self._workspace = workspace

    async def execute(self, path: str, **kw: Any) -> str:
        fp = _safe_resolve(path, self._workspace)
        if not fp.exists():
            return f"Error: file not found: {path}"
        if not fp.is_file():
            return f"Error: not a file: {path}"
        try:
            content = fp.read_text(encoding="utf-8")
            if len(content) > 50_000:
                return content[:50_000] + f"\n\n... [truncated, {len(content)} total chars]"
            return content
        except UnicodeDecodeError:
            return f"Error: {path} is a binary file, cannot read as text"


class WriteFileTool(BaseTool):
    """Write content to a file."""

    _name = "write_file"
    _description = "Write content to a file. Creates parent directories if needed. Path is relative to workspace."
    _parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to write (relative to workspace)",
            },
            "content": {
                "type": "string",
                "description": "Content to write",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Path):
        self._workspace = workspace

    async def execute(self, path: str, content: str, **kw: Any) -> str:
        fp = _safe_resolve(path, self._workspace)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"


class EditFileTool(BaseTool):
    """Edit a file by replacing text."""

    _name = "edit_file"
    _description = "Edit a file by finding and replacing text. The old_text must appear exactly once."
    _parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to edit (relative to workspace)",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to find",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text",
            },
        },
        "required": ["path", "old_text", "new_text"],
    }

    def __init__(self, workspace: Path):
        self._workspace = workspace

    async def execute(self, path: str, old_text: str, new_text: str, **kw: Any) -> str:
        fp = _safe_resolve(path, self._workspace)
        if not fp.exists():
            return f"Error: file not found: {path}"

        content = fp.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count == 0:
            return "Error: old_text not found in file"
        if count > 1:
            return f"Error: old_text appears {count} times. Provide more context to make it unique."

        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"


class ListDirTool(BaseTool):
    """List directory contents."""

    _name = "list_dir"
    _description = "List contents of a directory. Path is relative to workspace."
    _parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path (relative to workspace). Use '.' for workspace root.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Path):
        self._workspace = workspace

    async def execute(self, path: str = ".", **kw: Any) -> str:
        dp = _safe_resolve(path, self._workspace)
        if not dp.exists():
            return f"Error: directory not found: {path}"
        if not dp.is_dir():
            return f"Error: not a directory: {path}"

        items = []
        for item in sorted(dp.iterdir()):
            if item.name.startswith("."):
                continue
            prefix = "dir " if item.is_dir() else "file"
            size = ""
            if item.is_file():
                s = item.stat().st_size
                if s < 1024:
                    size = f" ({s}B)"
                else:
                    size = f" ({s // 1024}KB)"
            items.append(f"  {prefix}  {item.name}{size}")

        if not items:
            return f"Directory {path} is empty"

        return f"{path}/\n" + "\n".join(items)