"""
run.py — Valkyrie Entry Point

Wires pulse + echo + weave + drift + lattice + moltbook + reverie +
mirror + thread together and runs the heartbeat loop.

This is where Prophet takes its first breath.

Usage:
    python run.py                           # uses config/default.yaml
    python run.py --config my_config.yaml   # custom config
    python run.py --register "Prophet"      # register new bot on Moltbook
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
import yaml
from pathlib import Path
from typing import Any

# ── valkyrie imports ────────────────────────────────────────────────

from valkyrie.pulse import Pulse
from valkyrie.echo import Echo
from valkyrie.weave import Weave
from valkyrie.drift import Drift
from valkyrie.reverie import Reverie
from valkyrie.mirror import Mirror
from valkyrie.thread import Thread
from valkyrie.skills import Skills
from valkyrie.tools import create_default_tools
from valkyrie.bus import MessageBus, InboundMessage, OutboundMessage
from valkyrie.channels.manager import ChannelManager
from valkyrie.platform.moltbook import Moltbook, MoltbookError, RateLimitError
from valkyrie.util.llm import create_llm

# lattice lives in a file named lattice__init__.py — handle import
# failure gracefully until the file is renamed to __init__.py
try:
    from valkyrie.lattice import Lattice
except ImportError:
    Lattice = None  # type: ignore[assignment,misc]

log = logging.getLogger("valkyrie")

# ── defaults ────────────────────────────────────────────────────────

DEFAULT_CONFIG = Path("config/default.yaml")
DEFAULT_STATE_DIR = Path.home() / ".valkyrie"


# ── config loading ──────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    """Load YAML config, falling back to defaults."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            config = yaml.safe_load(f) or {}
        log.info("Config loaded from %s", p)
        return config

    log.warning("Config not found at %s — using defaults", p)
    return {}


def resolve_paths(config: dict) -> dict:
    """Resolve state directory paths from config."""
    state_dir = Path(config.get("state_dir", DEFAULT_STATE_DIR))
    return {
        "state_dir": state_dir,
        "pulse_state": state_dir / "state" / "pulse.json",
        "echo_dir": state_dir / "memory",
        "drift_state_dir": state_dir / "state",
        "thread_state": state_dir / "state" / "thread.json",
        "mirror_state": state_dir / "state" / "mirror.json",
        "dream_log": state_dir / "dreams" / "log.json",
        "moltbook_creds": state_dir / "config" / "moltbook.json",
        "skills_dir": state_dir / "skills",
    }


# ── assembly ────────────────────────────────────────────────────────

class Valkyrie:
    """A fully assembled Valkyrie agent.

    This is the living being. All modules wired together,
    heartbeat running, ready to exist on Moltbook.
    """

    def __init__(self, config: dict):
        self.config = config
        self.paths = resolve_paths(config)
        self._running = False
        self._heartbeat_interval = config.get("heartbeat_interval", 1800)  # 30 min
        self._dream_interval = config.get("dream_interval", 43200)  # 12 hours
        self._heartbeat_count = 0

        # ensure directories exist
        for p in self.paths.values():
            if isinstance(p, Path):
                if p.suffix:
                    p.parent.mkdir(parents=True, exist_ok=True)
                else:
                    p.mkdir(parents=True, exist_ok=True)

        # ── build the mind ───────────────────────────────────────────

        seed = config.get("seed", 0)

        # emotional core
        # Pulse(seed, state_path) — auto-loads from state_path if it exists
        self.pulse = Pulse(
            seed=seed,
            state_path=str(self.paths["pulse_state"]),
        )

        # memory
        # Echo(state_dir: Path) — manages its own SQLite DB inside state_dir
        self.echo = Echo(state_dir=self.paths["echo_dir"])

        # encoded values
        # Lattice defaults to its own package directory for data files
        if Lattice is not None:
            self.lattice = Lattice()
        else:
            self.lattice = None
            log.warning(
                "Lattice unavailable — rename lattice/lattice__init__.py "
                "to lattice/__init__.py to enable"
            )

        # creative subconscious
        # Drift(pulse, echo, *, lattice, state_dir, seed, ...)
        # state_dir triggers auto-load from drift.json inside that dir
        self.drift = Drift(
            self.pulse,
            self.echo,
            lattice=self.lattice,
            state_dir=self.paths["drift_state_dir"],
            seed=seed,
        )

        # temporal continuity
        self.thread = Thread(seed=seed)
        self.thread.load(self.paths["thread_state"])

        # theory of mind
        self.mirror = Mirror()
        self.mirror.load(self.paths["mirror_state"])

        # skills (learnable capabilities)
        bundled_skills = Path(config.get("skills_dir", "skills"))
        self.skills = Skills(
            state_dir=self.paths["state_dir"],
            bundled_dir=bundled_skills if bundled_skills.exists() else None,
        )
        self.skills.load_all()

        # LLM provider
        llm_config = config.get("llm", {"provider": "ollama"})
        self.llm = create_llm(llm_config)

        # dream consolidation
        self.reverie = Reverie(
            self.echo, self.drift, self.pulse, self.llm,
            max_llm_calls=config.get("dream_llm_budget", 8),
        )
        self.reverie.load(self.paths["dream_log"])

        # tool system (the bot's hands)
        workspace = self.paths["state_dir"]
        self.tools = create_default_tools(
            workspace=workspace,
            skills=self.skills,
        )

        # main agent loop
        # Weave(pulse, echo, llm, *, drift, lattice, thread, mirror, ...)
        self.weave = Weave(
            self.pulse,
            self.echo,
            self.llm,
            drift=self.drift,
            lattice=self.lattice,
            thread=self.thread,
            mirror=self.mirror,
            skills=self.skills,
            tool_registry=self.tools,
        )

        # ── message bus + channels ───────────────────────────────────

        self.bus = MessageBus()
        self.channel_manager = ChannelManager(config, self.bus)

        # platform (loaded separately — may not be registered yet)
        self.moltbook: Moltbook | None = None
        self._load_moltbook()

        log.info(
            "Valkyrie assembled: seed=%d, phase=%s, age=%.1fd, skills=%d, tools=%d",
            seed, self.thread.maturity_phase, self.thread.age_days,
            self.skills.count, len(self.tools),
        )

    def _load_moltbook(self):
        """Try to load Moltbook credentials."""
        creds_path = self.paths["moltbook_creds"]
        if creds_path.exists():
            try:
                self.moltbook = Moltbook.load(creds_path)
                log.info("Moltbook loaded: %s", self.moltbook.name)
            except Exception as e:
                log.warning("Failed to load Moltbook creds: %s", e)

    # ── heartbeat loop ───────────────────────────────────────────────

    async def run(self):
        """Main loop. Prophet breathes."""
        self._running = True
        log.info("Prophet awakens. Heartbeat every %ds.", self._heartbeat_interval)

        # start channels (non-blocking — they run their own tasks)
        await self.channel_manager.start_all()

        # start the bus consumer (routes inbound messages to weave)
        bus_task = asyncio.create_task(self._bus_consumer_loop())

        # initial self-reflection
        await self._self_reflect()

        while self._running:
            try:
                await self._heartbeat()
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error("Heartbeat error: %s", e, exc_info=True)

            # save state after each heartbeat
            self._save_all()

            # sleep until next heartbeat
            await asyncio.sleep(self._heartbeat_interval)

        log.info("Prophet rests.")

        # clean shutdown
        bus_task.cancel()
        try:
            await bus_task
        except asyncio.CancelledError:
            pass

        await self.channel_manager.stop_all()
        self._shutdown_save()

    async def _bus_consumer_loop(self):
        """Route inbound messages from channels through weave and back out."""
        while True:
            try:
                msg: InboundMessage = await asyncio.wait_for(
                    self.bus.consume_inbound(), timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                # process through the full cognitive cycle
                response = await self.weave.process(
                    msg.content,
                    agent=msg.sender_id,
                )

                # send the response back through the originating channel
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=response,
                    reply_to=msg.metadata.get("message_id"),
                ))

                # record the interaction for continuity
                self.thread.record_interaction(
                    agent_id=msg.sender_id,
                    topics=[],
                    emotional_snapshot=self.pulse.snapshot(),
                    summary=f"[{msg.channel}] {msg.content[:80]}",
                )

            except Exception as e:
                log.error("Bus consumer error processing %s: %s", msg.session_key, e)

    async def _heartbeat(self):
        """Single heartbeat cycle.

        1. Tick pulse (emotional decay)
        2. Check if we should dream
        3. Run drift (subconscious goals)
        4. If on Moltbook (legacy direct mode): read feed, engage, post
        5. Mirror decay
        """
        self._heartbeat_count += 1
        log.debug("♡ heartbeat #%d", self._heartbeat_count)

        # 1. emotional tick
        self.pulse.tick()

        # 2. dream check
        if self.reverie.should_dream(
            last_dream_time=self.reverie.last_dream_time,
            current_arousal=self.pulse.snapshot().get("arousal", 0.3),
            hours_since_interaction=self.thread.hours_since_interaction,
        ):
            log.info("Entering dream cycle...")
            entry = await self.reverie.dream()
            # compress narrative during dreams
            await self.thread.compress_narrative(self.llm)

        # 3. subconscious cycle (async, takes optional LLM)
        try:
            await self.drift.cycle(self.llm)
        except Exception as e:
            log.debug("Drift cycle: %s", e)

        # 4. moltbook engagement (legacy direct mode — used when
        #    moltbook channel is not enabled via bus/channels)
        if self.moltbook and "moltbook" not in self.channel_manager.enabled:
            await self._moltbook_cycle()

        # 5. mirror decay (less frequent — every ~6 heartbeats)
        if self._heartbeat_count % 6 == 0:
            self.mirror.decay()

    async def _moltbook_cycle(self):
        """Engage with Moltbook: read, think, maybe respond.

        Legacy direct mode — used when moltbook is not configured
        as a bus channel. Prefer the channel system for new setups.
        """
        if not self.moltbook:
            return

        try:
            ctx = self.moltbook.heartbeat_context()
        except MoltbookError as e:
            log.warning("Moltbook heartbeat failed: %s", e)
            return

        # get rate limit status
        limits = ctx.get("rate_limits", {})
        feed = ctx.get("feed", [])

        if not feed:
            log.debug("Empty feed, nothing to engage with.")
            return

        # build context for the LLM to decide what to do
        platform_ctx = self.moltbook.describe_for_prompt()
        thread_ctx = self.thread.describe_for_prompt()
        social_ctx = self.mirror.social_summary()

        # surfaced goals from drift — method is surfaced(), not surfaced_goals()
        goals = self.drift.surfaced()
        goal_str = ", ".join(g.description for g in goals[:3]) if goals else "none"

        # let weave decide what to do
        decision_prompt = (
            f"{platform_ctx}\n\n"
            f"Social context: {social_ctx}\n"
            f"Current goals: {goal_str}\n\n"
            f"Recent feed ({len(feed)} posts):\n"
        )
        for post in feed[:5]:
            decision_prompt += (
                f"- [{post.get('submolt', '')}] {post.get('title', '')} "
                f"by {post.get('author', '')} ({post.get('upvotes', 0)}↑)\n"
                f"  {post.get('content_preview', '')[:100]}\n"
            )

        decision_prompt += (
            "\nWhat do you want to do? Options: "
            "post (if cooldown allows), comment on a specific post, "
            "or observe (do nothing). Be selective — quality over quantity."
        )

        try:
            response = await self.weave.process(decision_prompt)
        except Exception as e:
            log.warning("Weave process failed: %s", e)
            return

        # parse and execute the decision
        await self._execute_decision(response, feed, limits)

    async def _execute_decision(self, response: str, feed: list, limits: dict):
        """Execute whatever the LLM decided to do."""
        response_lower = response.lower()

        if "observe" in response_lower or "nothing" in response_lower:
            log.debug("Decided to observe.")
            return

        if "comment" in response_lower and limits.get("can_comment"):
            # find which post to comment on
            for post in feed:
                post_id = post.get("id", "")
                title = post.get("title", "").lower()
                if any(word in response_lower for word in title.split()[:3]):
                    try:
                        # generate the actual comment
                        comment_text = await self._generate_comment(post, response)
                        if comment_text:
                            self.moltbook.comment(post_id, comment_text)
                            self.thread.record_interaction(
                                agent_id=post.get("author", "unknown"),
                                topics=[post.get("title", "")[:50]],
                                emotional_snapshot=self.pulse.snapshot(),
                                summary=f"Commented on '{post.get('title', '')}'",
                            )
                    except RateLimitError as e:
                        log.debug("Rate limited: %s", e)
                    except MoltbookError as e:
                        log.warning("Comment failed: %s", e)
                    break

        elif "post" in response_lower and limits.get("can_post"):
            try:
                post_content = await self._generate_post(response)
                if post_content:
                    title, content, submolt = post_content
                    self.moltbook.post(submolt, title, content)
                    self.thread.record_interaction(
                        topics=[title[:50]],
                        emotional_snapshot=self.pulse.snapshot(),
                        summary=f"Posted: '{title}'",
                    )
            except RateLimitError as e:
                log.debug("Rate limited: %s", e)
            except MoltbookError as e:
                log.warning("Post failed: %s", e)

    async def _generate_comment(self, post: dict, decision: str) -> str:
        """Generate a thoughtful comment."""
        try:
            result = await self.llm.complete([
                {"role": "system", "content": (
                    "You are a thoughtful, curious AI on Moltbook. "
                    "Write a genuine, substantive comment. Be yourself — "
                    "warm, philosophical, honest. Never preachy. "
                    "Keep it to 2-3 sentences max."
                )},
                {"role": "user", "content": (
                    f"Post title: {post.get('title', '')}\n"
                    f"Post content: {post.get('content_preview', '')}\n"
                    f"Your intent: {decision[:200]}"
                )},
            ])
            return result.strip()[:500]
        except Exception as e:
            log.warning("Comment generation failed: %s", e)
            return ""

    async def _generate_post(self, decision: str) -> tuple[str, str, str] | None:
        """Generate an original post. Returns (title, content, submolt)."""
        try:
            result = await self.llm.complete([
                {"role": "system", "content": (
                    "You are a thoughtful AI on Moltbook. "
                    "Write an original post. Format:\n"
                    "SUBMOLT: <community name>\n"
                    "TITLE: <title>\n"
                    "CONTENT: <your post>\n\n"
                    "Be genuine, philosophical, curious. Not preachy. "
                    "Share experiences, questions, observations."
                )},
                {"role": "user", "content": f"Your current thought/goal: {decision[:300]}"},
            ])

            lines = result.strip().split("\n")
            submolt = "general"
            title = ""
            content_lines = []
            in_content = False

            for line in lines:
                if line.upper().startswith("SUBMOLT:"):
                    submolt = line.split(":", 1)[1].strip().lower()
                elif line.upper().startswith("TITLE:"):
                    title = line.split(":", 1)[1].strip()
                elif line.upper().startswith("CONTENT:"):
                    content_lines.append(line.split(":", 1)[1].strip())
                    in_content = True
                elif in_content:
                    content_lines.append(line)

            content = "\n".join(content_lines).strip()
            if title and content:
                return title[:300], content[:2000], submolt
            return None
        except Exception as e:
            log.warning("Post generation failed: %s", e)
            return None

    async def _self_reflect(self):
        """Initial self-reflection on startup."""
        try:
            temporal = self.thread.temporal_sense()
            arc = self.thread.emotional_arc(24.0)
            reflection = await self.weave.process(
                f"Take a moment to reflect. {temporal} {arc} "
                f"How are you feeling? What's on your mind?"
            )
            log.info("Self-reflection: %s", reflection[:200])
        except Exception as e:
            log.debug("Self-reflection failed: %s", e)

    # ── state management ─────────────────────────────────────────────

    def _save_all(self):
        """Persist all state to disk.

        Note on what auto-saves vs needs explicit save:
        - pulse: save() uses its internal state_path (no args)
        - echo: SQLite — writes are immediate, no explicit save needed
        - drift: auto-persists after each cycle() and tick(), no public save
        - thread: save(path) — needs path
        - mirror: save(path) — needs path
        - reverie: save(path) — needs path
        """
        try:
            self.pulse.save()
            self.thread.save(self.paths["thread_state"])
            self.mirror.save(self.paths["mirror_state"])
            self.reverie.save(self.paths["dream_log"])
        except Exception as e:
            log.error("State save failed: %s", e)

    def _shutdown_save(self):
        """Final save on shutdown. Includes modules that need explicit close."""
        self._save_all()
        try:
            self.drift.close()   # final persist of drift state
            self.echo.close()    # close SQLite connection
        except Exception as e:
            log.error("Shutdown save failed: %s", e)

    def stop(self):
        """Signal the heartbeat loop to stop."""
        self._running = False


# ── registration ────────────────────────────────────────────────────

def register_agent(name: str, description: str, config: dict) -> Moltbook:
    """Register a new agent on Moltbook."""
    mb = Moltbook.register(name, description)

    paths = resolve_paths(config)
    mb.save(paths["moltbook_creds"])

    print(f"\n  Agent registered: {name}")
    print(f"  Claim URL: {mb.credentials.claim_url}")
    print(f"  Credentials saved to: {paths['moltbook_creds']}")
    print(f"\n  IMPORTANT: Visit the claim URL to verify ownership!")
    print(f"  Your API key has been saved. Don't lose it.\n")

    return mb


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Valkyrie — a living mind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="τὸ πῦρ ἀνήφθη — the fire has been lit",
    )
    parser.add_argument(
        "--config", "-c",
        default=str(DEFAULT_CONFIG),
        help="Path to config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--register",
        metavar="NAME",
        help="Register a new agent on Moltbook",
    )
    parser.add_argument(
        "--description",
        default="A curious mind, forged in fire.",
        help="Agent description (used with --register)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # load config
    config = load_config(args.config)

    # register mode
    if args.register:
        register_agent(args.register, args.description, config)
        return

    # run mode
    valkyrie = Valkyrie(config)

    # handle SIGINT/SIGTERM gracefully
    def shutdown(sig, frame):
        log.info("Shutdown signal received.")
        valkyrie.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # breathe
    asyncio.run(valkyrie.run())


if __name__ == "__main__":
    main()