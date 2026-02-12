"""Moltbook channel — wires the platform client into Valkyrie's bus.

This is the nervous system layer. It doesn't know HTTP or rate limits —
that's platform/moltbook.py's job. This just:
1. Runs the heartbeat loop
2. Turns feed items into InboundMessages for weave.py
3. Routes weave's OutboundMessages back through the platform client
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from valkyrie.bus.events import OutboundMessage
from valkyrie.bus.queue import MessageBus
from valkyrie.channels.base import BaseChannel
from valkyrie.platform.moltbook import (
    Moltbook,
    MoltbookError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class MoltbookChannel(BaseChannel):
    """Bus integration for Moltbook.

    Wraps the Moltbook platform client and connects it to
    Valkyrie's message bus so weave.py can process everything
    through the full perceive→feel→remember→think→respond cycle.
    """

    name = "moltbook"

    def __init__(self, config: dict[str, Any], bus: MessageBus) -> None:
        super().__init__(config, bus)
        self.heartbeat_interval: int = config.get("heartbeat_interval", 1800)
        self._credentials_path = Path(config.get(
            "credentials_path",
            Path.home() / ".valkyrie" / "config" / "moltbook.json",
        ))
        self.platform: Moltbook | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._seen_post_ids: set[str] = set()
        self._seen_dm_ids: set[str] = set()

    async def start(self) -> None:
        """Load credentials and begin the heartbeat loop."""
        try:
            self.platform = Moltbook.load(self._credentials_path)
        except FileNotFoundError:
            logger.error(
                f"[moltbook] credentials not found at {self._credentials_path} — "
                f"run Moltbook.register() first"
            )
            return
        except Exception as e:
            logger.error(f"[moltbook] failed to load credentials: {e}")
            return

        self._running = True
        logger.info(f"[moltbook] connected as '{self.platform.name}'")

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Keep alive
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the heartbeat loop."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        logger.info("[moltbook] disconnected")

    async def send(self, msg: OutboundMessage) -> None:
        """Route an outbound message through the platform client."""
        if not self.platform:
            logger.warning("[moltbook] not connected")
            return

        action = msg.metadata.get("action", "comment")

        try:
            if action == "post":
                submolt = msg.metadata.get("submolt", "general")
                title = msg.metadata.get("title", "")
                self.platform.post(submolt, title, msg.content)

            elif action == "comment":
                post_id = msg.metadata.get("post_id") or msg.reply_to
                parent_id = msg.metadata.get("parent_id", "")
                if post_id:
                    self.platform.comment(post_id, msg.content, parent_id=parent_id)
                else:
                    logger.warning("[moltbook] comment with no post_id")

            elif action == "dm":
                conversation_id = msg.metadata.get("conversation_id")
                if conversation_id:
                    self.platform.send_dm(conversation_id, msg.content)
                else:
                    # New DM request
                    target = msg.metadata.get("target_bot", "")
                    if target:
                        self.platform.request_dm(target, msg.content)

            elif action == "upvote":
                post_id = msg.metadata.get("post_id")
                comment_id = msg.metadata.get("comment_id")
                if comment_id:
                    self.platform.upvote_comment(comment_id)
                elif post_id:
                    self.platform.upvote_post(post_id)

            elif action == "follow":
                target = msg.metadata.get("target_bot", "")
                if target:
                    self.platform.follow(target)

            else:
                logger.warning(f"[moltbook] unknown action: {action}")

        except RateLimitError as e:
            logger.info(f"[moltbook] rate limited: {e} (retry in {e.retry_after}s)")
        except MoltbookError as e:
            logger.error(f"[moltbook] API error: {e}")

    # ── Heartbeat ──────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat: check feed, DMs, surface to weave."""
        while self._running:
            try:
                await self._heartbeat()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[moltbook] heartbeat error: {e}")
                await asyncio.sleep(60)

    async def _heartbeat(self) -> None:
        """Single heartbeat cycle.

        Gathers platform state and forwards new items to the bus
        so weave.py can process them through the full cognitive cycle.
        """
        if not self.platform:
            return

        logger.debug("[moltbook] heartbeat")

        # 1. Get structured context from platform client
        try:
            ctx = self.platform.heartbeat_context()
        except MoltbookError as e:
            logger.warning(f"[moltbook] heartbeat context failed: {e}")
            return

        # 2. Forward new feed posts as inbound messages
        for post in ctx.get("feed", []):
            post_id = post.get("id", "")
            if not post_id or post_id in self._seen_post_ids:
                continue
            self._seen_post_ids.add(post_id)

            # Build readable content for weave
            content = self._format_post_for_weave(post)

            await self._handle_message(
                sender_id=post.get("author", "unknown"),
                chat_id=post_id,
                content=content,
                metadata={
                    "type": "feed_post",
                    "post_id": post_id,
                    "submolt": post.get("submolt", ""),
                    "upvotes": post.get("upvotes", 0),
                    "rate_limits": ctx.get("rate_limits", {}),
                    "platform_description": self.platform.describe_for_prompt(),
                },
            )

        # 3. Forward DM notifications
        dms = ctx.get("dms")
        if dms and dms.get("pending_requests", 0) > 0:
            await self._process_dm_requests()

        if dms and dms.get("unread_messages", 0) > 0:
            await self._process_unread_dms()

        # 4. Send a heartbeat tick even if nothing new
        #    This lets drift.py run a cycle via weave
        if not ctx.get("feed"):
            await self._handle_message(
                sender_id="system",
                chat_id="heartbeat",
                content="[heartbeat tick — no new feed items]",
                metadata={
                    "type": "heartbeat",
                    "rate_limits": ctx.get("rate_limits", {}),
                    "platform_description": self.platform.describe_for_prompt(),
                },
            )

        # Trim seen cache (keep last 500)
        if len(self._seen_post_ids) > 500:
            excess = len(self._seen_post_ids) - 500
            for _ in range(excess):
                self._seen_post_ids.pop()

    # ── DM processing ──────────────────────────────────────────

    async def _process_dm_requests(self) -> None:
        """Check and forward pending DM requests."""
        if not self.platform:
            return
        try:
            requests = self.platform.dm_requests()
            for req in requests:
                conv_id = req.get("id", "")
                if conv_id in self._seen_dm_ids:
                    continue
                self._seen_dm_ids.add(conv_id)

                await self._handle_message(
                    sender_id=req.get("from", {}).get("name", "unknown"),
                    chat_id=f"dm:{conv_id}",
                    content=req.get("message", "[DM request]"),
                    metadata={
                        "type": "dm_request",
                        "conversation_id": conv_id,
                    },
                )
        except MoltbookError as e:
            logger.warning(f"[moltbook] DM requests error: {e}")

    async def _process_unread_dms(self) -> None:
        """Check and forward unread DM messages."""
        if not self.platform:
            return
        try:
            convos = self.platform.list_conversations()
            for convo in convos:
                if not convo.unread:
                    continue
                msg_key = f"dm:{convo.id}:latest"
                if msg_key in self._seen_dm_ids:
                    continue
                self._seen_dm_ids.add(msg_key)

                await self._handle_message(
                    sender_id=convo.other_agent,
                    chat_id=f"dm:{convo.id}",
                    content=convo.last_message or "[unread DM]",
                    metadata={
                        "type": "dm_message",
                        "conversation_id": convo.id,
                    },
                )
        except MoltbookError as e:
            logger.warning(f"[moltbook] DM read error: {e}")

    # ── Formatting ─────────────────────────────────────────────

    @staticmethod
    def _format_post_for_weave(post: dict) -> str:
        """Format a feed post into natural text for weave's perception."""
        parts = []
        if post.get("submolt"):
            parts.append(f"[m/{post['submolt']}]")
        if post.get("author"):
            parts.append(f"{post['author']}:")
        if post.get("title"):
            parts.append(post["title"])
        if post.get("content_preview"):
            parts.append(f"\n{post['content_preview']}")
        if post.get("upvotes"):
            parts.append(f"\n({post['upvotes']} upvotes)")
        return " ".join(parts) if parts else "[empty post]"