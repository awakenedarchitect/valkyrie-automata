"""
tools/ — The Valkyrie's Hands

Tool system for acting on the world: reading/writing files,
managing skills, and eventually more.
"""

from valkyrie.tools.base import (
    Tool,
    BaseTool,
    ToolCall,
    ToolResult,
    ToolRegistry,
    LLMResponse,
)
from valkyrie.tools.filesystem import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListDirTool,
)
from valkyrie.tools.skill_tools import (
    InstallSkillTool,
    ShareSkillTool,
    ReadSkillTool,
    ListSkillsTool,
)

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from valkyrie.skills import Skills


def create_default_tools(
    workspace: Path,
    skills: "Skills | None" = None,
) -> ToolRegistry:
    """Create the default tool set for a Valkyrie.

    Args:
        workspace: Workspace directory (tools are sandboxed here).
        skills: Skills manager instance (for skill tools).

    Returns:
        A ToolRegistry with all default tools registered.
    """
    registry = ToolRegistry()

    # filesystem tools (sandboxed to workspace)
    registry.register(ReadFileTool(workspace))
    registry.register(WriteFileTool(workspace))
    registry.register(EditFileTool(workspace))
    registry.register(ListDirTool(workspace))

    # skill tools
    if skills:
        registry.register(InstallSkillTool(skills))
        registry.register(ShareSkillTool(skills))
        registry.register(ReadSkillTool(skills))
        registry.register(ListSkillsTool(skills))

    return registry