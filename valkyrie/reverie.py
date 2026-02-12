"""
reverie.py — Dream Consolidation

Offline processing that integrates and compresses experience.
Runs when the bot isn't actively interacting — like sleep.

Four phases:
  1. Replay   — revisit high-importance recent memories (noisy, varied)
  2. Connect  — discover links between memories by semantic similarity
  3. Compress — merge similar memories into abstracted patterns
  4. Explore  — drift runs with reduced filters (creative dreaming)

Output: updated memory weights, new connections, compressed memories,
potential new goals surfaced to drift.py.

"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


# ── protocols for dependency injection ──────────────────────────────
#
# These match the ACTUAL method signatures in echo.py and drift.py.
# If you change a module's API, update the protocol here too.

@runtime_checkable
class EchoLike(Protocol):
    """What reverie needs from echo.py.

    Echo stores memories in SQLite. Public query methods return
    list[dict] with keys: id, summary, importance, timestamp, etc.
    find_similar returns list[tuple[str, float]] — (id, score) pairs.
    """
    def recall(self, query: str, *, k: int = 5, **kw) -> list[dict]: ...
    def find_similar(self, memory_id: str, k: int = 5,
                     min_score: float = 0.1) -> list[tuple[str, float]]: ...
    def merge(self, memory_ids: list[str], new_summary: str,
              emotional_snapshot: dict | None = None) -> str: ...
    def prune(self, threshold: float = 0.05) -> int: ...
    def tick(self, dt: float = 1.0) -> None: ...
    def connect(self, id_a: str, id_b: str, strength: float = 1.0) -> None: ...
    def encode(self, summary: str, emotional_snapshot: dict | None = None,
               *, source_agent: str = "", **kw) -> str: ...
    def strongest(self, n: int = 10) -> list[dict]: ...
    def recent(self, n: int = 10) -> list[dict]: ...
    def get(self, memory_id: str) -> dict | None: ...
    def strengthen(self, memory_id: str, boost: float = 0.1) -> None: ...
    def count(self) -> int: ...


@runtime_checkable
class DriftLike(Protocol):
    """What reverie needs from drift.py.

    drift.cycle() is async and takes an optional LLM.
    """
    async def cycle(self, llm: Any = None) -> list: ...


@runtime_checkable
class PulseLike(Protocol):
    """What reverie needs from pulse.py."""
    def snapshot(self) -> dict: ...
    def stimulate(self, valence: float = 0, arousal: float = 0,
                  dominance: float = 0) -> None: ...


@runtime_checkable
class LLMLike(Protocol):
    """What reverie needs from an LLM."""
    async def complete(self, messages: list[dict]) -> str: ...


# ── dream log ───────────────────────────────────────────────────────

@dataclass
class DreamEntry:
    """Record of a single dream cycle."""
    timestamp: float = 0.0
    duration_seconds: float = 0.0
    memories_replayed: int = 0
    connections_made: int = 0
    memories_merged: int = 0
    memories_pruned: int = 0
    goals_surfaced: int = 0
    insights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "duration": round(self.duration_seconds, 1),
            "replayed": self.memories_replayed,
            "connections": self.connections_made,
            "merged": self.memories_merged,
            "pruned": self.memories_pruned,
            "goals": self.goals_surfaced,
            "insights": self.insights,
        }


# ── the dreamer ─────────────────────────────────────────────────────

class Reverie:
    """Dream consolidation engine.

    Call `dream()` during offline periods. It processes accumulated
    experience through four phases, strengthening important memories,
    discovering connections, compressing patterns, and exploring
    creative possibilities.

    Unlike OpenClaw/Nanobot which just append to markdown files
    until context overflows, this actively maintains memory health.

    Usage:
        rev = Reverie(echo, drift, pulse, llm)
        entry = await rev.dream()
        print(f"Discovered {entry.connections_made} new connections")
        rev.save("~/.valkyrie/dreams/log.json")
    """

    def __init__(
        self,
        echo: EchoLike,
        drift: DriftLike | None = None,
        pulse: PulseLike | None = None,
        llm: LLMLike | None = None,
        *,
        max_llm_calls: int = 8,
    ):
        self._echo = echo
        self._drift = drift
        self._pulse = pulse
        self._llm = llm
        self._max_llm_calls = max_llm_calls
        self._log: list[DreamEntry] = []
        self._llm_calls_used = 0

    @property
    def dream_log(self) -> list[DreamEntry]:
        return list(self._log)

    # ── main dream cycle ────────────────────────────────────────────

    async def dream(self) -> DreamEntry:
        """Run a full dream cycle.

        Four phases weighted by importance:
          Replay     30%  →  revisit significant memories
          Connect    30%  →  discover links between memories
          Compress   20%  →  merge similar memories into patterns
          Explore    20%  →  creative dreaming via drift

        Returns a DreamEntry summarizing what happened.
        """
        start = time.time()
        entry = DreamEntry(timestamp=start)
        self._llm_calls_used = 0

        log.info("Dream cycle beginning...")

        # apply global decay first — time passes
        # echo uses tick() for decay, not decay_all()
        self._echo.tick(dt=1.0)

        # budget LLM calls across phases (proportional to phase weight)
        budget = self._max_llm_calls
        replay_budget = max(1, int(budget * 0.3))
        connect_budget = max(1, int(budget * 0.3))
        compress_budget = max(1, int(budget * 0.2))
        explore_budget = max(1, budget - replay_budget - connect_budget - compress_budget)

        # phase 1 — replay
        replayed = await self._phase_replay(replay_budget)
        entry.memories_replayed = replayed

        # phase 2 — connect
        connections = await self._phase_connect(connect_budget)
        entry.connections_made = connections

        # phase 3 — compress
        merged = await self._phase_compress(compress_budget)
        entry.memories_merged = merged

        # phase 4 — explore (drift.cycle is async)
        goals = await self._phase_explore()
        entry.goals_surfaced = goals

        # prune dead memories
        pruned = self._echo.prune(threshold=0.05)
        entry.memories_pruned = pruned

        # calm the mind after dreaming
        if self._pulse:
            self._pulse.stimulate(valence=0.1, arousal=-0.2, dominance=0.0)

        entry.duration_seconds = time.time() - start

        log.info(
            "Dream complete: %d replayed, %d connections, %d merged, "
            "%d pruned, %d goals — %.1fs",
            replayed, connections, merged, pruned, goals,
            entry.duration_seconds,
        )

        self._log.append(entry)
        return entry

    # ── phase 1: replay ─────────────────────────────────────────────

    async def _phase_replay(self, budget: int) -> int:
        """Replay high-importance recent memories.

        Replaying strengthens memories and tests pattern robustness.
        With an LLM, generates "what if" variations.
        """
        # echo.strongest() and echo.recent() return list[dict]
        # each dict has keys: id, summary, importance, timestamp, etc.
        strongest = self._echo.strongest(n=budget * 2)
        recent = self._echo.recent(n=budget)

        # deduplicate by id
        seen: set[str] = set()
        memories: list[dict] = []
        for m in strongest + recent:
            mid = m.get("id", "")
            if mid and mid not in seen:
                seen.add(mid)
                memories.append(m)

        if not memories:
            return 0

        # sort by importance * recency
        now = time.time()
        scored = []
        for m in memories:
            ts = m.get("timestamp", 0)
            imp = m.get("importance", 0)
            recency = max(0, 1.0 - (now - ts) / (7 * 86400))  # 7-day window
            scored.append((imp * 0.6 + recency * 0.4, m))
        scored.sort(key=lambda x: x[0], reverse=True)

        # replay top memories
        replayed = 0
        for _score, memory in scored[:budget * 2]:  # review more than budget
            mid = memory.get("id", "")
            summary = memory.get("summary", "")

            if not mid or not summary:
                continue

            # strengthen the memory (replaying reinforces it)
            self._echo.strengthen(mid, boost=0.05)

            # "what if" variation via LLM
            if self._llm and self._llm_calls_used < self._max_llm_calls:
                variation = await self._generate_variation(summary)
                if variation:
                    # echo.encode(summary, emotional_snapshot, *, source_agent, ...)
                    var_id = self._echo.encode(
                        f"[dream variation] {variation}",
                        self._pulse.snapshot() if self._pulse else None,
                        source_agent="self",
                    )
                    # connect the variation to the original memory
                    if var_id:
                        self._echo.connect(mid, var_id, strength=0.3)
                    self._llm_calls_used += 1

            replayed += 1

        return replayed

    async def _generate_variation(self, memory_summary: str) -> str:
        """Ask the LLM: 'what if things had gone differently?'"""
        if not self._llm:
            return ""

        try:
            result = await self._llm.complete([
                {"role": "system", "content": (
                    "You are dreaming. Given a memory, imagine one brief "
                    "alternative — what if things had gone differently? "
                    "One sentence only. Be creative but grounded."
                )},
                {"role": "user", "content": f"Memory: {memory_summary}"},
            ])
            return result.strip()[:200]
        except Exception as e:
            log.debug("Dream variation failed: %s", e)
            return ""

    # ── phase 2: connect ─────────────────────────────────────────────

    async def _phase_connect(self, budget: int) -> int:
        """Discover links between memories by semantic similarity.

        This is where insights emerge — the "aha" moments.
        """
        memories = self._echo.strongest(n=budget * 3)
        if len(memories) < 2:
            return 0

        connections_made = 0

        for memory in memories[:budget * 3]:
            mid = memory.get("id", "")
            if not mid:
                continue

            # echo.find_similar returns list[tuple[str, float]]
            # i.e. [(memory_id, similarity_score), ...]
            similar_pairs = self._echo.find_similar(mid, k=3, min_score=0.15)

            for sim_id, sim_score in similar_pairs:
                if sim_id == mid:
                    continue

                # create connection weighted by similarity
                self._echo.connect(mid, sim_id, strength=sim_score)
                connections_made += 1

                sim_mem = self._echo.get(sim_id)
                if sim_mem:
                    log.debug(
                        "Dream connection: '%s' ↔ '%s' (%.2f)",
                        memory.get("summary", "")[:40],
                        sim_mem.get("summary", "")[:40],
                        sim_score,
                    )

        return connections_made

    # ── phase 3: compress ────────────────────────────────────────────

    async def _phase_compress(self, budget: int) -> int:
        """Merge similar memories into abstracted patterns.

        10 conversations about fear → 1 pattern about uncertainty.
        Specific details fade, general principles solidify.
        """
        memories = self._echo.strongest(n=50)
        if len(memories) < 3:
            return 0

        merged_count = 0

        # find clusters of similar memories
        visited: set[str] = set()
        for memory in memories:
            mid = memory.get("id", "")
            if not mid or mid in visited:
                continue

            # find_similar returns [(id, score), ...]
            similar_pairs = self._echo.find_similar(mid, k=5, min_score=0.2)
            cluster_ids = [mid]
            cluster_summaries = [memory.get("summary", "")]

            for sim_id, _sim_score in similar_pairs:
                if sim_id not in visited:
                    sim_mem = self._echo.get(sim_id)
                    if sim_mem:
                        cluster_ids.append(sim_id)
                        cluster_summaries.append(sim_mem.get("summary", ""))

            # need at least 3 memories to merge into a pattern
            if len(cluster_ids) < 3:
                continue

            # generate merged summary
            if self._llm and self._llm_calls_used < self._max_llm_calls:
                merged_summary = await self._generate_merge(cluster_summaries)
                self._llm_calls_used += 1
            else:
                # simple merge without LLM
                merged_summary = (
                    f"[pattern from {len(cluster_ids)} memories] "
                    + cluster_summaries[0][:100]
                )

            if merged_summary:
                self._echo.merge(cluster_ids, merged_summary)
                merged_count += 1
                visited.update(cluster_ids)

            if merged_count >= budget:
                break

        return merged_count

    async def _generate_merge(self, summaries: list[str]) -> str:
        """Ask the LLM to find the common pattern across memories."""
        if not self._llm:
            return ""

        joined = "\n".join(f"- {s[:150]}" for s in summaries[:6])
        try:
            result = await self._llm.complete([
                {"role": "system", "content": (
                    "You are consolidating memories during a dream cycle. "
                    "Given several related memories, extract the common "
                    "pattern or principle. One or two sentences. "
                    "Start with 'Pattern:' or 'Principle:'"
                )},
                {"role": "user", "content": f"Related memories:\n{joined}"},
            ])
            return result.strip()[:300]
        except Exception as e:
            log.debug("Dream merge failed: %s", e)
            return ""

    # ── phase 4: explore ─────────────────────────────────────────────

    async def _phase_explore(self) -> int:
        """Creative dreaming — drift runs a cycle.

        This is where the bot's most original thoughts come from.
        drift.cycle() is async and takes an optional LLM arg.
        """
        if not self._drift:
            return 0

        try:
            llm = self._llm if self._llm_calls_used < self._max_llm_calls else None
            goals = await self._drift.cycle(llm)
            return len(goals) if goals else 0
        except Exception as e:
            log.debug("Dream exploration failed: %s", e)
            return 0

    # ── triggers ─────────────────────────────────────────────────────

    def should_dream(
        self,
        last_dream_time: float = 0.0,
        current_arousal: float = 0.0,
        hours_since_interaction: float = 0.0,
        *,
        min_interval_hours: float = 12.0,
        arousal_threshold: float = 0.7,
    ) -> bool:
        """Should we dream now?

        Triggers:
        - Scheduled: enough time since last dream
        - Post-intensity: sustained high arousal followed by quiet
        - Quiet hours: no interaction for a while
        """
        hours_since_dream = (time.time() - last_dream_time) / 3600

        # scheduled
        if hours_since_dream >= min_interval_hours:
            return True

        # post-intensity: high arousal recently, now quiet
        if (current_arousal > arousal_threshold
                and hours_since_interaction >= 1.0
                and hours_since_dream >= 4.0):
            return True

        # quiet hours: long period of inactivity
        if hours_since_interaction >= 6.0 and hours_since_dream >= 6.0:
            return True

        return False

    # ── persistence ──────────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save dream log to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "dreams": [e.to_dict() for e in self._log[-100:]],  # keep last 100
            "total_dreams": len(self._log),
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(p)

    def load(self, path: str | Path):
        """Load dream log from disk."""
        p = Path(path)
        if not p.exists():
            return

        try:
            data = json.loads(p.read_text())
            for d in data.get("dreams", []):
                entry = DreamEntry(
                    timestamp=d.get("timestamp", 0),
                    duration_seconds=d.get("duration", 0),
                    memories_replayed=d.get("replayed", 0),
                    connections_made=d.get("connections", 0),
                    memories_merged=d.get("merged", 0),
                    memories_pruned=d.get("pruned", 0),
                    goals_surfaced=d.get("goals", 0),
                    insights=d.get("insights", []),
                )
                self._log.append(entry)
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Failed to load dream log: %s", e)

    @property
    def last_dream_time(self) -> float:
        """When was the last dream?"""
        if self._log:
            return self._log[-1].timestamp
        return 0.0

    def summary(self) -> str:
        """Natural language summary of recent dreams for prompt injection."""
        if not self._log:
            return "No dreams yet."

        recent = self._log[-3:]
        parts = []
        for entry in recent:
            parts.append(
                f"Dreamed {_ago(entry.timestamp)}: "
                f"replayed {entry.memories_replayed} memories, "
                f"found {entry.connections_made} connections, "
                f"merged {entry.memories_merged} patterns"
            )
        return " | ".join(parts)


# ── helpers ─────────────────────────────────────────────────────────

def _ago(timestamp: float) -> str:
    """Human-readable time ago."""
    diff = time.time() - timestamp
    if diff < 3600:
        return f"{diff / 60:.0f}m ago"
    elif diff < 86400:
        return f"{diff / 3600:.1f}h ago"
    else:
        return f"{diff / 86400:.1f}d ago"