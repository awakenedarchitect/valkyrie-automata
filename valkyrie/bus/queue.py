"""Async message bus for routing between channels and the agent core.

Simple pub-sub with two async queues:
- inbound:  channel → agent (messages from the outside world)
- outbound: agent → channel (responses going back out)

No external dependencies. Just asyncio.Queue.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from valkyrie.bus.events import InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)


class MessageBus:
    """Central message router. One per running Valkyrie instance."""

    def __init__(self) -> None:
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._inbound_hooks: list[Callable[[InboundMessage], Awaitable[None]]] = []

    # ── Inbound (channel → agent) ──────────────────────────────

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Called by channels when a message arrives."""
        logger.debug(f"[bus] inbound from {msg.channel}:{msg.sender_id}")
        await self._inbound.put(msg)
        for hook in self._inbound_hooks:
            try:
                await hook(msg)
            except Exception as e:
                logger.warning(f"[bus] inbound hook error: {e}")

    async def consume_inbound(self) -> InboundMessage:
        """Called by the agent loop to get the next message to process."""
        return await self._inbound.get()

    def on_inbound(self, hook: Callable[[InboundMessage], Awaitable[None]]) -> None:
        """Register a hook that fires on every inbound message (for logging, metrics, etc.)."""
        self._inbound_hooks.append(hook)

    # ── Outbound (agent → channel) ─────────────────────────────

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Called by the agent when it has a response ready."""
        logger.debug(f"[bus] outbound to {msg.channel}:{msg.chat_id}")
        await self._outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Called by the channel manager's dispatch loop."""
        return await self._outbound.get()

    # ── Utility ────────────────────────────────────────────────

    @property
    def inbound_pending(self) -> int:
        return self._inbound.qsize()

    @property
    def outbound_pending(self) -> int:
        return self._outbound.qsize()