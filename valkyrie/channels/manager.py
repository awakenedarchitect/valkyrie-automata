"""Channel manager — orchestrates all active channels.

Responsibilities:
- Read config, lazy-import only enabled channels
- Start/stop all channels
- Dispatch outbound messages to the correct channel
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from valkyrie.bus.events import OutboundMessage
from valkyrie.bus.queue import MessageBus
from valkyrie.channels.base import BaseChannel

logger = logging.getLogger(__name__)

# Registry: channel name → (module path, class name)
# Add new channels here. They're only imported if enabled in config.
CHANNEL_REGISTRY: dict[str, tuple[str, str]] = {
    "moltbook": ("valkyrie.channels.moltbook", "MoltbookChannel"),
    "telegram": ("valkyrie.channels.telegram", "TelegramChannel"),
    "discord": ("valkyrie.channels.discord", "DiscordChannel"),
}


class ChannelManager:
    """Manages the lifecycle of all active channels."""

    def __init__(self, config: dict[str, Any], bus: MessageBus) -> None:
        """
        Args:
            config: Full instance config (we read config["channels"]).
            bus: The shared message bus.
        """
        self.config = config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task[None] | None = None

        self._init_channels()

    def _init_channels(self) -> None:
        """Lazy-load only the channels that are enabled in config."""
        channels_config: dict[str, Any] = self.config.get("channels", {})

        for name, (module_path, class_name) in CHANNEL_REGISTRY.items():
            chan_conf = channels_config.get(name, {})
            if not chan_conf.get("enabled", False):
                continue

            try:
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                self.channels[name] = cls(chan_conf, self.bus)
                logger.info(f"[channels] {name} enabled")
            except ImportError as e:
                logger.warning(f"[channels] {name} not available: {e}")
            except Exception as e:
                logger.error(f"[channels] {name} failed to init: {e}")

    # ── Lifecycle ──────────────────────────────────────────────

    async def start_all(self) -> None:
        """Start all enabled channels + the outbound dispatcher."""
        if not self.channels:
            logger.warning("[channels] no channels enabled")
            return

        # Start the outbound dispatch loop
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())

        # Start each channel concurrently
        tasks = [
            asyncio.create_task(self._start_one(name, ch))
            for name, ch in self.channels.items()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Stop all channels and the dispatcher."""
        logger.info("[channels] stopping all...")

        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        for name, ch in self.channels.items():
            try:
                await ch.stop()
                logger.info(f"[channels] {name} stopped")
            except Exception as e:
                logger.error(f"[channels] error stopping {name}: {e}")

    # ── Internal ───────────────────────────────────────────────

    async def _start_one(self, name: str, channel: BaseChannel) -> None:
        """Start a single channel with error logging."""
        try:
            logger.info(f"[channels] starting {name}...")
            await channel.start()
        except Exception as e:
            logger.error(f"[channels] {name} crashed: {e}")

    async def _dispatch_loop(self) -> None:
        """Route outbound messages to the correct channel."""
        logger.info("[channels] outbound dispatcher running")
        while True:
            try:
                msg: OutboundMessage = await asyncio.wait_for(
                    self.bus.consume_outbound(), timeout=1.0
                )
                ch = self.channels.get(msg.channel)
                if ch:
                    await ch.send(msg)
                else:
                    logger.warning(f"[channels] unknown channel: {msg.channel}")
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[channels] dispatch error: {e}")

    # ── Info ───────────────────────────────────────────────────

    def get_status(self) -> dict[str, dict[str, bool]]:
        return {
            name: {"enabled": True, "running": ch.is_running}
            for name, ch in self.channels.items()
        }

    @property
    def enabled(self) -> list[str]:
        return list(self.channels.keys())