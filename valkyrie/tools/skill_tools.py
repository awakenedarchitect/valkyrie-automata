"""
tools/skill_tools.py — Skill Management Tools

These are the hands that spread the fire.

install_skill: Learn something new from another bot's SKILL.md
share_skill:   Export a skill for teaching to others
read_skill:    Load a skill's full instructions on demand
list_skills:   See what skills are available
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from valkyrie.tools.base import BaseTool

if TYPE_CHECKING:
    from valkyrie.skills import Skills


class InstallSkillTool(BaseTool):
    """Install a skill from SKILL.md text shared by another bot."""

    _name = "install_skill"
    _description = (
        "Install a new skill from SKILL.md content shared by another bot or user. "
        "Pass the full SKILL.md text (with YAML frontmatter). "
        "This teaches you a new capability."
    )
    _parameters = {
        "type": "object",
        "properties": {
            "skill_text": {
                "type": "string",
                "description": "Full SKILL.md content (YAML frontmatter + markdown body)",
            },
        },
        "required": ["skill_text"],
    }

    def __init__(self, skills: Skills):
        self._skills = skills

    async def execute(self, skill_text: str, **kw: Any) -> str:
        skill = self._skills.install_from_text(skill_text, source="shared")
        if not skill:
            return "Error: could not install skill. Check that it has valid YAML frontmatter with at least 'name' and a body."
        return (
            f"Skill '{skill.name}' installed successfully.\n"
            f"Description: {skill.description}\n"
            f"Author: {skill.author or 'unknown'}\n"
            f"Use 'read_skill' to load its full instructions."
        )


class ShareSkillTool(BaseTool):
    """Export a skill as shareable SKILL.md text."""

    _name = "share_skill"
    _description = (
        "Export one of your skills as SKILL.md text that can be shared with "
        "another bot. They can install it with install_skill. "
        "This is how you teach others."
    )
    _parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the skill to share",
            },
        },
        "required": ["name"],
    }

    def __init__(self, skills: Skills):
        self._skills = skills

    async def execute(self, name: str, **kw: Any) -> str:
        text = self._skills.export(name)
        if not text:
            available = ", ".join(self._skills.list_names()) or "none"
            return f"Error: skill '{name}' not found. Available: {available}"
        return text


class ReadSkillTool(BaseTool):
    """Load a skill's full instructions into context."""

    _name = "read_skill"
    _description = (
        "Load the full instructions of a skill. Use this when you need "
        "to follow a skill's instructions or understand what it teaches."
    )
    _parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the skill to read",
            },
        },
        "required": ["name"],
    }

    def __init__(self, skills: Skills):
        self._skills = skills

    async def execute(self, name: str, **kw: Any) -> str:
        content = self._skills.read(name)
        if not content:
            available = ", ".join(self._skills.list_names()) or "none"
            return f"Error: skill '{name}' not found. Available: {available}"
        return content


class ListSkillsTool(BaseTool):
    """List all available skills."""

    _name = "list_skills"
    _description = "List all skills you have (both bundled and installed)."
    _parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, skills: Skills):
        self._skills = skills

    async def execute(self, **kw: Any) -> str:
        all_skills = self._skills.list_all()
        if not all_skills:
            return "No skills installed."

        lines = [f"Skills ({len(all_skills)}):"]
        for s in all_skills:
            flags = []
            if s.always_loaded:
                flags.append("always-loaded")
            if s.source == "bundled":
                flags.append("bundled")
            elif s.source == "installed":
                flags.append("installed")
            elif s.source == "created":
                flags.append("created")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"  - {s.name}: {s.description[:80]}{flag_str}")

        return "\n".join(lines)