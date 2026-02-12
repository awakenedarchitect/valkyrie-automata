"""Abstract base for all channel implementations.

Every channel (Moltbook, Telegram, Discord, etc.) subclasses BaseChannel
and implements three methods: start(), stop(), send().

The base class handles:
- Access control (allowlist filtering)
- Message forwarding to the bus
- Running state tracking
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from valkyrie.bus.events import InboundMessage, OutboundMessage
from valkyrie.bus.queue import MessageBus

logger = logging.getLogger(__name__)


class BaseChannel(ABC):
    """Abstract interface for a chat/platform channel."""

    name: str = "base"

    def __init__(self, config: dict[str, Any], bus: MessageBus) -> None:
        """
        Args:
            config: Channel-specific config dict from instance.yaml.
            bus: The shared message bus.
        """
        self.config = config
        self.bus = bus
        self._running = False

    # ── Required implementations ───────────────────────────────

    @abstractmethod
    async def start(self) -> None:
        """Connect to the platform and begin listening.

        Should be a long-running async task that:
        1. Authenticates with the platform
        2. Listens for incoming messages
        3. Forwards them via _handle_message()
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect and clean up resources."""
        ...

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Deliver an outbound message through this channel."""
        ...

    # ── Built-in helpers ───────────────────────────────────────

    def is_allowed(self, sender_id: str) -> bool:
        """Check sender against the allowlist.

        If no allowlist is configured, everyone is allowed (dev mode).
        """
        allow_from: list[str] = self.config.get("allow_from", [])
        if not allow_from:
            return True
        return str(sender_id) in allow_from

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Validate and forward an incoming message to the bus.

        Called by channel implementations after platform-specific parsing.
        """
        if not self.is_allowed(sender_id):
            logger.warning(
                f"[{self.name}] access denied for {sender_id} — "
                f"not in allow_from list"
            )
            return

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=metadata or {},
        )
        await self.bus.publish_inbound(msg)

    @property
    def is_running(self) -> bool:
        return self._running