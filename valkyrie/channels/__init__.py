"""Channels — how Valkyrie connects to the outside world.

Each channel implements BaseChannel and handles platform-specific
messaging (Moltbook, Telegram, Discord, etc.).
"""

from valkyrie.channels.base import BaseChannel

__all__ = ["BaseChannel"]