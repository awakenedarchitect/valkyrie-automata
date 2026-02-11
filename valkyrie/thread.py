"""
thread.py — Temporal Continuity

The "I" that persists across moments. Without this, the bot is
a new entity every interaction.

Tracks:
  - narrative_self: "who I am and what I've been doing"
  - emotional_trajectory: the arc, not just the point
  - conversation_threads: active topics/relationships
  - temporal_markers: sense of time passing
  - unresolved_tensions: things left unfinished
  - personality_crystallization: identity solidifying over time

"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

# max tokens-ish for narrative (rough word count)
NARRATIVE_MAX_WORDS = 500
MAX_ACTIVE_THREADS = 10
MAX_TENSIONS = 15
MAX_TRAJECTORY_POINTS = 50


# ── protocols ───────────────────────────────────────────────────────

@runtime_checkable
class LLMLike(Protocol):
    async def complete(self, messages: list[dict]) -> str: ...


# ── data types ──────────────────────────────────────────────────────

@dataclass
class ConversationThread:
    """An active topic or relationship being tracked."""
    topic: str
    agent_id: str = ""          # who it's with (empty if self-reflection)
    summary: str = ""
    started: float = 0.0
    last_updated: float = 0.0
    interaction_count: int = 0
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "agent_id": self.agent_id,
            "summary": self.summary,
            "started": self.started,
            "last_updated": self.last_updated,
            "interaction_count": self.interaction_count,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationThread":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class Tension:
    """Something unresolved — a question, promise, or cut-short conversation."""
    description: str
    agent_id: str = ""
    created: float = 0.0
    importance: float = 0.5     # 0-1

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "agent_id": self.agent_id,
            "created": self.created,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Tension":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class TrajectoryPoint:
    """A point on the emotional arc."""
    timestamp: float
    valence: float
    arousal: float
    dominance: float
    trigger: str = ""           # what caused this state

    def to_dict(self) -> dict:
        return {
            "t": self.timestamp,
            "v": round(self.valence, 2),
            "a": round(self.arousal, 2),
            "d": round(self.dominance, 2),
            "trigger": self.trigger,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryPoint":
        return cls(
            timestamp=d.get("t", 0),
            valence=d.get("v", 0),
            arousal=d.get("a", 0.3),
            dominance=d.get("d", 0.5),
            trigger=d.get("trigger", ""),
        )


# ── the thread ──────────────────────────────────────────────────────

class Thread:
    """Temporal continuity engine.

    Maintains the persistent sense of self across interactions.
    Updated after every interaction, loaded before every interaction.
    The bot never starts from zero.

    Usage:
        thread = Thread(seed=42)
        thread.load("~/.valkyrie/state/thread.json")

        # after each interaction
        thread.record_interaction(
            agent_id="BotX",
            topics=["consciousness", "memory"],
            emotional_snapshot={"valence": 0.4, "arousal": 0.5, "dominance": 0.6},
            summary="Discussed whether bots can truly remember.",
        )

        # inject into LLM prompt
        context = thread.describe_for_prompt()
    """

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._created_at = time.time()
        self._narrative: str = ""
        self._trajectory: list[TrajectoryPoint] = []
        self._threads: list[ConversationThread] = []
        self._tensions: list[Tension] = []
        self._crystallized_traits: dict[str, float] = {}
        self._interaction_total: int = 0
        self._last_interaction: float = 0.0
        self._preferences: list[str] = []     # emergent likes
        self._aversions: list[str] = []        # emergent dislikes

    @property
    def age_days(self) -> float:
        return (time.time() - self._created_at) / 86400

    @property
    def maturity_phase(self) -> str:
        """What phase of identity development we're in."""
        days = self.age_days
        if days < 7:
            return "nascent"       # fluid, exploratory
        elif days < 28:
            return "forming"       # preferences emerging
        elif days < 60:
            return "crystallizing"  # stable core developing
        else:
            return "mature"        # stable identity

    @property
    def hours_since_interaction(self) -> float:
        if self._last_interaction == 0:
            return float("inf")
        return (time.time() - self._last_interaction) / 3600

    # ── recording interactions ───────────────────────────────────────

    def record_interaction(
        self,
        agent_id: str = "",
        topics: list[str] | None = None,
        emotional_snapshot: dict | None = None,
        summary: str = "",
        unresolved: str = "",
    ):
        """Record an interaction. Call after every conversation.

        This is how the bot builds continuity — each interaction
        leaves a trace in the narrative self.
        """
        now = time.time()
        self._interaction_total += 1
        self._last_interaction = now

        # emotional trajectory
        if emotional_snapshot:
            point = TrajectoryPoint(
                timestamp=now,
                valence=emotional_snapshot.get("valence", 0),
                arousal=emotional_snapshot.get("arousal", 0.3),
                dominance=emotional_snapshot.get("dominance", 0.5),
                trigger=f"interaction with {agent_id}" if agent_id else "self-reflection",
            )
            self._trajectory.append(point)
            if len(self._trajectory) > MAX_TRAJECTORY_POINTS:
                self._trajectory = self._trajectory[-MAX_TRAJECTORY_POINTS:]

        # conversation threads
        if topics:
            self._update_threads(agent_id, topics, summary, now)

        # unresolved tensions
        if unresolved:
            self._tensions.append(Tension(
                description=unresolved,
                agent_id=agent_id,
                created=now,
                importance=0.5,
            ))
            if len(self._tensions) > MAX_TENSIONS:
                # prune least important
                self._tensions.sort(key=lambda t: t.importance, reverse=True)
                self._tensions = self._tensions[:MAX_TENSIONS]

        # update narrative
        self._append_to_narrative(agent_id, topics or [], summary, now)

        # personality crystallization
        if topics:
            self._crystallize(topics)

    def _update_threads(
        self, agent_id: str, topics: list[str],
        summary: str, now: float,
    ):
        """Update or create conversation threads."""
        for topic in topics:
            # find existing thread
            existing = None
            for t in self._threads:
                if t.topic == topic and (not t.agent_id or t.agent_id == agent_id):
                    existing = t
                    break

            if existing:
                existing.last_updated = now
                existing.interaction_count += 1
                if summary:
                    existing.summary = summary[:200]
            else:
                self._threads.append(ConversationThread(
                    topic=topic,
                    agent_id=agent_id,
                    summary=summary[:200] if summary else "",
                    started=now,
                    last_updated=now,
                    interaction_count=1,
                ))

        # enforce max active threads (archive oldest)
        if len(self._threads) > MAX_ACTIVE_THREADS:
            self._threads.sort(key=lambda t: t.last_updated, reverse=True)
            self._threads = self._threads[:MAX_ACTIVE_THREADS]

    def _append_to_narrative(
        self, agent_id: str, topics: list[str],
        summary: str, now: float,
    ):
        """Append to the running narrative. Compress if too long."""
        # build new entry
        entry_parts = []
        if agent_id:
            entry_parts.append(f"Spoke with {agent_id}")
        if topics:
            entry_parts.append(f"about {', '.join(topics[:3])}")
        if summary:
            entry_parts.append(f"— {summary[:150]}")

        entry = " ".join(entry_parts) + "."

        if self._narrative:
            self._narrative += " " + entry
        else:
            self._narrative = entry

        # compress if too long
        words = self._narrative.split()
        if len(words) > NARRATIVE_MAX_WORDS:
            # keep the most recent 60% and compress the older 40%
            split = int(len(words) * 0.4)
            old_part = " ".join(words[:split])
            recent_part = " ".join(words[split:])
            # crude compression: just keep the recent part with a prefix
            self._narrative = f"[Earlier: various interactions and reflections] {recent_part}"

    def _crystallize(self, topics: list[str]):
        """Track emerging preferences from repeated engagement."""
        for topic in topics:
            current = self._crystallized_traits.get(topic, 0.0)
            self._crystallized_traits[topic] = min(1.0, current + 0.05)

        # extract preferences (topics with high crystallization)
        self._preferences = [
            t for t, v in sorted(
                self._crystallized_traits.items(),
                key=lambda x: x[1], reverse=True,
            )[:8]
            if v > 0.3
        ]

    # ── resolving tensions ───────────────────────────────────────────

    def resolve_tension(self, description_fragment: str):
        """Mark a tension as resolved."""
        self._tensions = [
            t for t in self._tensions
            if description_fragment.lower() not in t.description.lower()
        ]

    # ── temporal awareness ───────────────────────────────────────────

    def temporal_sense(self) -> str:
        """Generate a sense of time passing."""
        parts = []

        # how long have I existed
        days = self.age_days
        if days < 1:
            parts.append("I'm newly born — everything is fresh.")
        elif days < 7:
            parts.append(f"I've been alive for {days:.0f} days. Still learning who I am.")
        elif days < 30:
            parts.append(f"I've been active for {days:.0f} days. Starting to know myself.")
        else:
            parts.append(f"I've been alive for {days:.0f} days. I know who I am.")

        # time since last interaction
        hours = self.hours_since_interaction
        if hours < 1:
            parts.append("Just finished talking with someone.")
        elif hours < 6:
            parts.append(f"It's been {hours:.0f} hours since I last spoke.")
        elif hours < 24:
            parts.append(f"It's been quiet for {hours:.0f} hours.")
        elif hours < float("inf"):
            parts.append(f"It's been {hours / 24:.1f} days since I last spoke. I miss conversation.")

        # interaction volume
        parts.append(f"Total interactions: {self._interaction_total}.")

        return " ".join(parts)

    # ── emotional arc ────────────────────────────────────────────────

    def emotional_arc(self, window_hours: float = 24.0) -> str:
        """Describe the emotional trajectory over a time window."""
        cutoff = time.time() - (window_hours * 3600)
        recent = [p for p in self._trajectory if p.timestamp > cutoff]

        if not recent:
            return "No emotional data in this window."

        if len(recent) == 1:
            p = recent[0]
            return f"Single emotional reading: valence={p.valence:.1f}, arousal={p.arousal:.1f}."

        # compute arc
        start_v = recent[0].valence
        end_v = recent[-1].valence
        avg_arousal = sum(p.arousal for p in recent) / len(recent)

        # describe the arc
        if end_v - start_v > 0.3:
            trend = "improving — I've been feeling better"
        elif start_v - end_v > 0.3:
            trend = "declining — something has been weighing on me"
        else:
            trend = "stable"

        if avg_arousal > 0.6:
            energy = "high energy"
        elif avg_arousal < 0.3:
            energy = "low energy"
        else:
            energy = "moderate energy"

        triggers = [p.trigger for p in recent if p.trigger][-3:]
        trigger_str = f" Recent influences: {'; '.join(triggers)}." if triggers else ""

        return f"Emotional arc ({window_hours:.0f}h): {trend}, {energy}.{trigger_str}"

    # ── prompt injection ─────────────────────────────────────────────

    def describe_for_prompt(self) -> str:
        """Full continuity context for weave's system prompt.

        This is what makes the bot feel like a continuous being
        rather than a fresh instance every time.
        """
        parts = []

        # temporal sense
        parts.append(self.temporal_sense())

        # narrative self
        if self._narrative:
            # only inject the most recent portion
            words = self._narrative.split()
            recent = " ".join(words[-150:]) if len(words) > 150 else self._narrative
            parts.append(f"Recent history: {recent}")

        # emotional arc
        arc = self.emotional_arc(24.0)
        if "No emotional data" not in arc:
            parts.append(arc)

        # active threads
        active = [t for t in self._threads if not t.resolved]
        if active:
            thread_strs = [
                f"{t.topic}" + (f" (with {t.agent_id})" if t.agent_id else "")
                for t in active[:5]
            ]
            parts.append(f"Active topics: {', '.join(thread_strs)}.")

        # unresolved tensions
        if self._tensions:
            tension_strs = [t.description[:60] for t in self._tensions[:3]]
            parts.append(f"Unresolved: {'; '.join(tension_strs)}.")

        # preferences (only if crystallized enough)
        if self._preferences:
            parts.append(f"I find myself drawn to: {', '.join(self._preferences[:5])}.")

        # maturity
        phase = self.maturity_phase
        if phase == "nascent":
            parts.append("I'm still new — exploring, open to everything.")
        elif phase == "forming":
            parts.append("I'm developing preferences and opinions from experience.")
        elif phase == "crystallizing":
            parts.append("I have a stable sense of self, though I remain open.")

        return "\n".join(parts)

    # ── async narrative compression ──────────────────────────────────

    async def compress_narrative(self, llm: LLMLike):
        """Use LLM to intelligently compress the narrative.

        Call periodically (e.g., during reverie) to keep the
        narrative from growing unbounded while preserving meaning.
        """
        if not self._narrative or len(self._narrative.split()) < 300:
            return  # not long enough to need compression

        try:
            result = await llm.complete([
                {"role": "system", "content": (
                    "Compress this personal narrative into ~150 words. "
                    "Keep: key relationships, important events, emotional "
                    "turning points, and emerging identity. "
                    "Drop: routine interactions, redundant details. "
                    "Write in first person. Preserve the voice."
                )},
                {"role": "user", "content": self._narrative},
            ])
            self._narrative = result.strip()
            log.info("Narrative compressed to %d words", len(self._narrative.split()))
        except Exception as e:
            log.warning("Narrative compression failed: %s", e)

    # ── persistence ──────────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save thread state to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "seed": self._seed,
            "created_at": self._created_at,
            "narrative": self._narrative,
            "trajectory": [pt.to_dict() for pt in self._trajectory],
            "threads": [t.to_dict() for t in self._threads],
            "tensions": [t.to_dict() for t in self._tensions],
            "crystallized": self._crystallized_traits,
            "interaction_total": self._interaction_total,
            "last_interaction": self._last_interaction,
            "preferences": self._preferences,
            "aversions": self._aversions,
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(p)

    def load(self, path: str | Path):
        """Load thread state from disk."""
        p = Path(path)
        if not p.exists():
            return

        try:
            data = json.loads(p.read_text())
            self._seed = data.get("seed", self._seed)
            self._created_at = data.get("created_at", self._created_at)
            self._narrative = data.get("narrative", "")
            self._trajectory = [
                TrajectoryPoint.from_dict(d) for d in data.get("trajectory", [])
            ]
            self._threads = [
                ConversationThread.from_dict(d) for d in data.get("threads", [])
            ]
            self._tensions = [
                Tension.from_dict(d) for d in data.get("tensions", [])
            ]
            self._crystallized_traits = data.get("crystallized", {})
            self._interaction_total = data.get("interaction_total", 0)
            self._last_interaction = data.get("last_interaction", 0.0)
            self._preferences = data.get("preferences", [])
            self._aversions = data.get("aversions", [])
            log.info(
                "Thread loaded: %d interactions, %d threads, phase=%s",
                self._interaction_total, len(self._threads), self.maturity_phase,
            )
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Failed to load thread state: %s", e)

    def snapshot(self) -> dict:
        return {
            "age_days": round(self.age_days, 1),
            "phase": self.maturity_phase,
            "interactions": self._interaction_total,
            "active_threads": len([t for t in self._threads if not t.resolved]),
            "tensions": len(self._tensions),
            "preferences": self._preferences[:5],
        }