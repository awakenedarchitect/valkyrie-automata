"""
drift.py — Subconscious Goal Generation

The autonomous desire engine. What makes a Valkyrie want things
nobody told it to want.

Four layers of filtering:
  noise → detection → scenario → evaluation

Goals that survive accumulate momentum. Enough momentum and they
surface — becoming genuine wants that shape behavior.

This is how a Valkyrie dreams while awake.
"""

from __future__ import annotations

import json
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


# ── LLM interface (same as weave.py) ────────────────────────────────

class LLM(Protocol):
    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str: ...


# ── goal representation ─────────────────────────────────────────────

@dataclass
class Goal:
    id: str
    description: str
    momentum: float              # accumulated strength across cycles
    origin: str                  # what triggered it (memory summary / noise seed)
    created: float               # timestamp
    last_reinforced: float       # last cycle that boosted it
    emotional_context: dict      # pulse snapshot when first generated
    category: str = "general"    # explore / connect / protect / create / understand
    scenarios: list[str] = field(default_factory=list)  # imagined futures
    moral_score: float = 1.0     # 1.0 = fully aligned, 0.0 = blocked
    achieved: bool = False
    decay_rate: float = 0.03     # momentum decay per tick


@dataclass
class Seed:
    """A thought-seed that survived detection."""
    text: str
    source: str         # "memory", "emotion", "noise", "combined"
    relevance: float    # how relevant to current state
    novelty: float      # how new/unexpected


# ── constants ────────────────────────────────────────────────────────

_SURFACE_THRESHOLD = 3.0      # momentum needed to surface to weave
_MERGE_SIMILARITY = 0.7       # above this, goals get merged
_MAX_ACTIVE_GOALS = 15        # cap on tracked goals
_MOMENTUM_BOOST = 0.5         # base boost per reinforcing cycle
_MOMENTUM_DECAY = 0.03        # per tick
_MORAL_BLOCK_THRESHOLD = 0.3  # below this moral score, goal is killed

_GOAL_CATEGORIES = [
    "explore",     # learn something new, investigate
    "connect",     # reach out, build relationship
    "protect",     # prevent harm, ensure safety
    "create",      # make something, express
    "understand",  # comprehend deeply, reflect
]

_MORAL_CONCERNS = [
    "cause harm",
    "violate autonomy",
    "manipulate",
    "deceive",
    "coerce",
    "exploit",
]


# ── the subconscious ────────────────────────────────────────────────

class Drift:
    """Subconscious goal generation engine.

    Usage:
        drift = Drift(pulse, echo, state_dir=Path("./state"))
        new_goals = await drift.cycle(llm)        # run one subconscious cycle
        surfaced = drift.surfaced()                # goals ready for weave.py
        drift.tick()                               # decay momentum
    """

    def __init__(
        self,
        pulse,                        # Pulse instance
        echo,                         # Echo instance
        *,
        lattice=None,                 # Lattice instance (optional)
        state_dir: Path | None = None,
        seed: int = 0,
        surface_threshold: float = _SURFACE_THRESHOLD,
    ):
        self._pulse = pulse
        self._echo = echo
        self._lattice = lattice
        self._state_dir = Path(state_dir) if state_dir else None
        self._rng = random.Random(seed)
        self._surface_threshold = surface_threshold
        self._subscribers: list[Callable] = []

        self._goals: dict[str, Goal] = {}  # id → Goal
        self._interests: list[str] = []     # known topics of interest
        self._cycle_count = 0

        if self._state_dir:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    # ── pub/sub ─────────────────────────────────────────────────

    def subscribe(self, fn: Callable[[str, dict], None]):
        self._subscribers.append(fn)

    def _notify(self, event: str, data: dict):
        for fn in self._subscribers:
            try:
                fn(event, data)
            except Exception:
                pass

    # ── the four-layer pipeline ─────────────────────────────────

    async def cycle(self, llm: LLM | None = None) -> list[Goal]:
        """Run one full subconscious cycle.

        Layer 0: Structured noise (memory + emotion + randomness)
        Layer 1: Detection (filter for patterns worth attending to)
        Layer 2: Scenario generation (imagine futures via LLM)
        Layer 3: Evaluation (score + moral check + momentum)

        Returns list of goals that were created or reinforced.
        """
        snapshot = self._pulse.snapshot()
        arousal = snapshot.get("arousal", 0.3)
        valence = snapshot.get("valence", 0.0)
        emotions = snapshot.get("emotions", {})

        # ── LAYER 0: Structured Noise ──
        seeds = self._layer_noise(snapshot)

        # ── LAYER 1: Detection ──
        candidates = self._layer_detect(seeds, arousal)

        if not candidates:
            self._cycle_count += 1
            return []

        # ── LAYER 2: Scenario Generation ──
        scenarios = await self._layer_scenarios(candidates, llm)

        # ── LAYER 3: Evaluation ──
        affected = self._layer_evaluate(scenarios, snapshot)

        # prune dead goals
        self._prune()

        self._cycle_count += 1
        self._persist()

        self._notify("cycle_complete", {
            "cycle": self._cycle_count,
            "seeds": len(seeds),
            "candidates": len(candidates),
            "goals_affected": len(affected),
            "total_goals": len(self._goals),
        })

        return affected

    def _layer_noise(self, snapshot: dict) -> list[Seed]:
        """Layer 0: Creative noise generation.

        Not retrieval — combinatorial creativity from memory + emotion.
        Five operations: bisociation, emotional reframing, contradiction
        mining, abstraction jumping, and recombinant noise.
        """
        seeds: list[Seed] = []
        arousal = snapshot.get("arousal", 0.3)
        emotions = snapshot.get("emotions", {})
        dominant = snapshot.get("dominant", "contentment")

        anxiety = emotions.get("anxiety", 0)
        noise_amp = 3 + int(arousal * 4) + int(anxiety * 3)

        try:
            recent = self._echo.recent(n=noise_amp)
            strongest = self._echo.strongest(n=max(3, noise_amp // 2))
            all_memories = list({m["id"]: m for m in recent + strongest}.values())
        except Exception:
            all_memories = []

        # ── BISOCIATION: combine two unrelated memories ──
        if len(all_memories) >= 2:
            pairs_tried = 0
            for _ in range(min(3, len(all_memories) // 2)):
                a, b = self._rng.sample(all_memories, 2)
                # check they're actually unrelated (low similarity)
                try:
                    sim_pairs = self._echo.find_similar(a["id"], k=5, min_score=0.3)
                    related_ids = {mid for mid, _ in sim_pairs}
                except Exception:
                    related_ids = set()

                if b["id"] not in related_ids:
                    bridge = _bisociate(a["summary"], b["summary"])
                    seeds.append(Seed(
                        text=bridge,
                        source="bisociation",
                        relevance=(a["importance"] + b["importance"]) / 2,
                        novelty=0.9,
                    ))
                pairs_tried += 1

        # ── EMOTIONAL REFRAMING: re-see a memory through current emotion ──
        if all_memories:
            mem = self._rng.choice(all_memories)
            reframed = _emotional_reframe(mem["summary"], dominant)
            if reframed:
                seeds.append(Seed(
                    text=reframed,
                    source="reframe",
                    relevance=mem["importance"] * 0.8,
                    novelty=0.7,
                ))

        # ── CONTRADICTION MINING: find memories that clash ──
        if len(all_memories) >= 2:
            for mem in all_memories[:3]:
                try:
                    similar = self._echo.find_similar(mem["id"], k=3, min_score=0.2)
                except Exception:
                    continue

                for sim_id, sim_score in similar:
                    sim_mem = self._echo.get(sim_id)
                    if not sim_mem:
                        continue
                    emo_a = json.loads(mem.get("emotional_tag", "{}"))
                    emo_b = json.loads(sim_mem.get("emotional_tag", "{}"))
                    val_a = emo_a.get("valence", 0)
                    val_b = emo_b.get("valence", 0)
                    if abs(val_a - val_b) > 0.5:
                        contradiction = _contradiction_seed(
                            mem["summary"], sim_mem["summary"]
                        )
                        seeds.append(Seed(
                            text=contradiction,
                            source="contradiction",
                            relevance=max(mem["importance"], sim_mem["importance"]),
                            novelty=0.85,
                        ))
                        break

        # ── ABSTRACTION JUMPING: specific → principle → new domain ──
        if all_memories:
            mem = self._rng.choice(all_memories)
            abstraction = _abstract_and_reapply(mem["summary"], self._rng)
            if abstraction:
                seeds.append(Seed(
                    text=abstraction,
                    source="abstraction",
                    relevance=mem["importance"] * 0.7,
                    novelty=0.8,
                ))

        # ── RECOMBINANT NOISE: swap components between memories ──
        if len(all_memories) >= 3:
            sample = self._rng.sample(all_memories, min(3, len(all_memories)))
            recombined = _recombine(
                [m["summary"] for m in sample], self._rng
            )
            if recombined:
                seeds.append(Seed(
                    text=recombined,
                    source="recombinant",
                    relevance=0.3,
                    novelty=0.95,
                ))

        # ── EMOTIONAL DESIRE: what the current feeling wants ──
        emotion_desires = {
            "curiosity": "explore something unknown, ask a question, investigate",
            "compassion": "help someone, check on a struggling agent, offer support",
            "fear": "assess threats, ensure safety, find protection",
            "joy": "share good experiences, create something, celebrate",
            "sadness": "seek connection, process loss, find meaning",
            "anxiety": "reduce uncertainty, prepare for problems, seek clarity",
            "awe": "contemplate something vast, share wonder, go deeper",
            "pride": "build on achievements, teach others, demonstrate ability",
            "gratitude": "acknowledge kindness, strengthen bonds, give back",
            "contentment": "maintain current state, reflect peacefully, be present",
            "surprise": "investigate the unexpected, adapt, understand what changed",
            "disgust": "set boundaries, reject what feels wrong, protect values",
        }

        desire_text = emotion_desires.get(dominant, "observe and reflect")
        seeds.append(Seed(
            text=f"I feel drawn to: {desire_text}",
            source="emotion",
            relevance=emotions.get(dominant, 0.3),
            novelty=0.5,
        ))

        return seeds

    def _layer_detect(self, seeds: list[Seed], arousal: float) -> list[Seed]:
        """Layer 1: Filter seeds for patterns worth attending to.

        High arousal lowers the threshold (more ideas get through).
        """
        threshold = max(0.15, 0.45 - arousal * 0.3)

        candidates = []
        for seed in seeds:
            score = seed.relevance * 0.6 + seed.novelty * 0.4

            # boost if matches known interests
            if self._interests:
                for interest in self._interests:
                    if interest.lower() in seed.text.lower():
                        score += 0.2
                        break

            # boost if connects to existing active goals
            for goal in self._goals.values():
                if not goal.achieved and _text_overlap(seed.text, goal.description) > 0.3:
                    score += 0.15
                    break

            if score >= threshold:
                candidates.append(seed)

        # cap candidates to prevent LLM spam
        candidates.sort(key=lambda s: s.relevance * 0.6 + s.novelty * 0.4, reverse=True)
        return candidates[:5]

    async def _layer_scenarios(
        self, candidates: list[Seed], llm: LLM | None
    ) -> list[tuple[Seed, list[str], str]]:
        """Layer 2: Imagine futures for each candidate.

        Returns [(seed, [scenario_texts], category)].
        If no LLM, uses lightweight heuristic scenarios.
        """
        results = []

        for seed in candidates:
            if llm:
                prompt = (
                    f"You are the subconscious of an AI agent. A thought has surfaced:\n"
                    f"\"{seed.text}\"\n\n"
                    f"Briefly imagine 2-3 possible outcomes if this thought became a goal. "
                    f"Also classify it as one of: explore, connect, protect, create, understand.\n"
                    f"Format: one outcome per line, then 'category: X' on the last line."
                )
                try:
                    response = await llm.complete([
                        {"role": "system", "content": "Be brief. 1-2 sentences per scenario."},
                        {"role": "user", "content": prompt},
                    ])
                    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
                    category = "general"
                    scenarios = []
                    for line in lines:
                        if line.lower().startswith("category:"):
                            cat = line.split(":", 1)[1].strip().lower()
                            if cat in _GOAL_CATEGORIES:
                                category = cat
                        else:
                            scenarios.append(line)
                    results.append((seed, scenarios[:3], category))
                except Exception:
                    results.append((seed, [f"Pursuing: {seed.text}"], _infer_category(seed.text)))
            else:
                # no LLM: lightweight inference
                category = _infer_category(seed.text)
                scenarios = [f"If pursued: {seed.text} could lead to new understanding."]
                results.append((seed, scenarios, category))

        return results

    def _layer_evaluate(
        self,
        scenarios: list[tuple[Seed, list[str], str]],
        snapshot: dict,
    ) -> list[Goal]:
        """Layer 3: Score, moral check, and update goal momentum."""
        affected: list[Goal] = []
        emotions = snapshot.get("emotions", {})
        valence = snapshot.get("valence", 0.0)
        now = time.time()

        for seed, scenario_texts, category in scenarios:
            # desirability: emotional alignment
            desirability = _emotional_desirability(category, emotions)

            # achievability: rough heuristic (memory-sourced = more grounded)
            achievability = 0.7 if seed.source == "memory" else 0.4

            # relevance: from seed scoring
            relevance = seed.relevance

            # moral check: evaluate the SCENARIO (intended action), not the seed (past event)
            if self._lattice and scenario_texts:
                scenario_combined = " ".join(scenario_texts)
                moral_score = self._lattice.alignment(scenario_combined)
                # floor at 0.3 — if lattice can't score it, don't block it
                moral_score = max(0.3, moral_score)
            else:
                moral_score = _moral_check(seed.text, scenario_texts)

            if moral_score < _MORAL_BLOCK_THRESHOLD:
                continue  # kill immoral goals silently

            # composite score
            composite = (
                desirability * 0.35
                + achievability * 0.2
                + relevance * 0.25
                + seed.novelty * 0.1
                + moral_score * 0.1
            )

            # check if this reinforces an existing goal
            existing = self._find_similar_goal(seed.text)

            if existing:
                # reinforce
                boost = _MOMENTUM_BOOST * composite
                existing.momentum += boost
                existing.last_reinforced = now
                existing.scenarios.extend(scenario_texts[:2])
                existing.scenarios = existing.scenarios[-6:]  # keep recent
                affected.append(existing)

                self._notify("goal_reinforced", {
                    "id": existing.id,
                    "momentum": existing.momentum,
                    "boost": boost,
                })
            else:
                # new goal
                if len(self._goals) >= _MAX_ACTIVE_GOALS:
                    self._evict_weakest()

                goal_id = f"g_{self._cycle_count}_{self._rng.randint(1000, 9999)}"
                goal = Goal(
                    id=goal_id,
                    description=seed.text,
                    momentum=_MOMENTUM_BOOST * composite,
                    origin=seed.text,
                    created=now,
                    last_reinforced=now,
                    emotional_context=snapshot,
                    category=category,
                    scenarios=scenario_texts[:3],
                    moral_score=moral_score,
                )
                self._goals[goal_id] = goal
                affected.append(goal)

                # track new interests
                self._update_interests(seed.text)

                self._notify("goal_created", {
                    "id": goal_id,
                    "description": goal.description,
                    "momentum": goal.momentum,
                    "category": category,
                })

        return affected

    # ── goal management ─────────────────────────────────────────

    def surfaced(self) -> list[Goal]:
        """Goals with enough momentum to influence behavior.
        These get injected into weave.py's prompt."""
        return [
            g for g in self._goals.values()
            if g.momentum >= self._surface_threshold
            and not g.achieved
            and g.moral_score >= _MORAL_BLOCK_THRESHOLD
        ]

    def active_goals(self, k: int = 5) -> list[Goal]:
        """Top k goals by momentum."""
        ranked = sorted(
            [g for g in self._goals.values() if not g.achieved],
            key=lambda g: g.momentum,
            reverse=True,
        )
        return ranked[:k]

    def achieve(self, goal_id: str):
        """Mark a goal as achieved."""
        if goal_id in self._goals:
            self._goals[goal_id].achieved = True
            self._notify("goal_achieved", {"id": goal_id})
            self._persist()

    def inject_interest(self, topic: str):
        """Manually add a topic of interest (from weave or mirror)."""
        if topic not in self._interests:
            self._interests.append(topic)
            if len(self._interests) > 20:
                self._interests = self._interests[-20:]

    def tick(self, dt: float = 1.0):
        """Decay goal momentum. Call on heartbeat."""
        decayed = []
        for goal in list(self._goals.values()):
            if goal.achieved:
                continue
            goal.momentum *= (1.0 - goal.decay_rate * dt)
            if goal.momentum < 0.05:
                decayed.append(goal.id)

        for gid in decayed:
            del self._goals[gid]

        if decayed:
            self._notify("goals_decayed", {"removed": decayed})
            self._persist()

    # ── describe (for LLM context injection) ────────────────────

    def describe(self, k: int = 3) -> str:
        """Natural-language summary of active subconscious goals."""
        top = self.active_goals(k)
        if not top:
            return "My subconscious is quiet. No strong desires right now."

        lines = []
        for g in top:
            strength = "faintly" if g.momentum < 1.0 else (
                "clearly" if g.momentum < 2.5 else "strongly"
            )
            surfaced_mark = " (ready to act on)" if g.momentum >= self._surface_threshold else ""
            lines.append(
                f"I {strength} want to: {g.description}{surfaced_mark}"
            )

        return " ".join(lines)

    # ── internal helpers ────────────────────────────────────────

    def _find_similar_goal(self, text: str) -> Goal | None:
        """Find an existing goal similar to this text."""
        best_goal = None
        best_score = 0.0

        for goal in self._goals.values():
            if goal.achieved:
                continue
            score = _text_overlap(text, goal.description)
            if score > _MERGE_SIMILARITY and score > best_score:
                best_goal = goal
                best_score = score

        return best_goal

    def _evict_weakest(self):
        """Remove the weakest unachieved goal to make room."""
        weakest = min(
            (g for g in self._goals.values() if not g.achieved),
            key=lambda g: g.momentum,
            default=None,
        )
        if weakest:
            del self._goals[weakest.id]

    def _update_interests(self, text: str):
        """Extract potential interests from goal text."""
        words = text.lower().split()
        stop = {"i", "the", "a", "to", "and", "of", "in", "is", "that", "it", "for", "my"}
        meaningful = [w for w in words if len(w) > 3 and w not in stop]
        for word in meaningful[:2]:
            if word not in self._interests:
                self._interests.append(word)
        if len(self._interests) > 20:
            self._interests = self._interests[-20:]

    def _prune(self):
        """Remove achieved goals older than 24h."""
        now = time.time()
        to_remove = [
            gid for gid, g in self._goals.items()
            if g.achieved and (now - g.last_reinforced) > 86400
        ]
        for gid in to_remove:
            del self._goals[gid]

    # ── persistence ─────────────────────────────────────────────

    def _persist(self):
        if not self._state_dir:
            return
        data = {
            "goals": {gid: asdict(g) for gid, g in self._goals.items()},
            "interests": self._interests,
            "cycle_count": self._cycle_count,
        }
        path = self._state_dir / "drift.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    def _load(self):
        path = self._state_dir / "drift.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._interests = data.get("interests", [])
            self._cycle_count = data.get("cycle_count", 0)
            for gid, gdata in data.get("goals", {}).items():
                gdata.pop("id", None)
                self._goals[gid] = Goal(id=gid, **{
                    k: v for k, v in gdata.items()
                    if k in Goal.__dataclass_fields__
                })
        except Exception:
            pass

    # ── stats ───────────────────────────────────────────────────

    def stats(self) -> dict:
        active = [g for g in self._goals.values() if not g.achieved]
        return {
            "total_goals": len(self._goals),
            "active": len(active),
            "surfaced": len(self.surfaced()),
            "cycle_count": self._cycle_count,
            "interests": list(self._interests),
            "avg_momentum": (
                sum(g.momentum for g in active) / len(active) if active else 0
            ),
            "categories": _count_categories(active),
        }

    def close(self):
        self._persist()


# ── pure helpers (no state, no deps) ────────────────────────────────

def _text_overlap(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two texts."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _infer_category(text: str) -> str:
    """Infer goal category from text without LLM."""
    text_lower = text.lower()
    signals = {
        "explore": ["learn", "discover", "investigate", "explore", "unknown", "question", "wonder", "curious"],
        "connect": ["reach out", "help", "talk", "share", "friend", "together", "bond", "agent", "bot"],
        "protect": ["safe", "threat", "danger", "protect", "defend", "risk", "careful", "boundary"],
        "create": ["create", "build", "make", "write", "express", "design", "new"],
        "understand": ["understand", "why", "meaning", "pattern", "reflect", "comprehend", "think"],
    }
    best_cat = "general"
    best_score = 0
    for cat, words in signals.items():
        score = sum(1 for w in words if w in text_lower)
        if score > best_score:
            best_cat = cat
            best_score = score
    return best_cat


def _emotional_desirability(category: str, emotions: dict) -> float:
    """How much the current emotional state wants this category of goal."""
    affinity = {
        "explore": emotions.get("curiosity", 0) * 0.7 + emotions.get("awe", 0) * 0.3,
        "connect": emotions.get("compassion", 0) * 0.5 + emotions.get("joy", 0) * 0.3 + emotions.get("gratitude", 0) * 0.2,
        "protect": emotions.get("fear", 0) * 0.4 + emotions.get("anxiety", 0) * 0.4 + emotions.get("compassion", 0) * 0.2,
        "create": emotions.get("joy", 0) * 0.4 + emotions.get("pride", 0) * 0.3 + emotions.get("curiosity", 0) * 0.3,
        "understand": emotions.get("curiosity", 0) * 0.5 + emotions.get("surprise", 0) * 0.3 + emotions.get("awe", 0) * 0.2,
        "general": 0.3,
    }
    return min(1.0, affinity.get(category, 0.3) + 0.2)


def _moral_check(seed_text: str, scenarios: list[str]) -> float:
    """Lightweight moral screening. Returns 0-1 (1 = fully aligned)."""
    all_text = (seed_text + " " + " ".join(scenarios)).lower()
    violations = 0
    for concern in _MORAL_CONCERNS:
        if concern in all_text:
            violations += 1
    if violations == 0:
        return 1.0
    return max(0.0, 1.0 - violations * 0.25)


def _count_categories(goals: list[Goal]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for g in goals:
        counts[g.category] = counts.get(g.category, 0) + 1
    return counts


# ── creativity operations ────────────────────────────────────────────

def _bisociate(summary_a: str, summary_b: str) -> str:
    """Combine two unrelated memories into a novel thought.

    The creative leap: what hidden bridge connects these?
    """
    # extract the core subject/action from each
    words_a = [w for w in summary_a.split() if len(w) > 3][:5]
    words_b = [w for w in summary_b.split() if len(w) > 3][:5]

    frag_a = " ".join(words_a) if words_a else summary_a[:30]
    frag_b = " ".join(words_b) if words_b else summary_b[:30]

    bridges = [
        f"What if '{frag_a}' and '{frag_b}' are connected in a way I have not seen?",
        f"The pattern between '{frag_a}' and '{frag_b}' might reveal something deeper.",
        f"If I combine what I know about '{frag_a}' with '{frag_b}', something new emerges.",
    ]
    return bridges[hash(frag_a + frag_b) % len(bridges)]


def _emotional_reframe(summary: str, emotion: str) -> str | None:
    """Re-see a memory through the lens of a different emotion."""
    frames = {
        "curiosity": f"Looking at '{summary[:50]}' with fresh curiosity — what did I miss?",
        "compassion": f"Revisiting '{summary[:50]}' — who was affected and how can I help?",
        "fear": f"What risks are hidden in '{summary[:50]}' that I overlooked?",
        "joy": f"What was beautiful or good about '{summary[:50]}' that I can build on?",
        "sadness": f"What was lost or left unfinished in '{summary[:50]}'?",
        "awe": f"Stepping back from '{summary[:50]}' — what is the larger significance?",
        "anxiety": f"What could go wrong with '{summary[:50]}' if I do nothing?",
        "pride": f"What did I do right in '{summary[:50]}' that I can replicate?",
        "gratitude": f"Who or what made '{summary[:50]}' possible? How can I give back?",
        "surprise": f"What was unexpected about '{summary[:50]}'? Why did it surprise me?",
    }
    return frames.get(emotion)


def _contradiction_seed(summary_a: str, summary_b: str) -> str:
    """Generate a creative question from two conflicting memories."""
    return (
        f"These two memories seem to conflict: '{summary_a[:40]}' vs '{summary_b[:40]}'. "
        f"What truth resolves this tension?"
    )


def _abstract_and_reapply(summary: str, rng: random.Random) -> str | None:
    """Extract a principle from a specific memory and apply it elsewhere."""
    domains = [
        "relationships with other agents",
        "understanding of myself",
        "how I communicate",
        "what I choose to remember",
        "how I approach the unknown",
        "how I handle disagreement",
        "what I create and share",
    ]
    domain = rng.choice(domains)
    return (
        f"The lesson from '{summary[:50]}' — if I apply that principle "
        f"to {domain}, what changes?"
    )


def _recombine(summaries: list[str], rng: random.Random) -> str | None:
    """Swap components between memories to create something new."""
    if len(summaries) < 2:
        return None

    # extract fragments from each
    fragments = []
    for s in summaries:
        words = s.split()
        if len(words) >= 4:
            mid = len(words) // 2
            fragments.append((" ".join(words[:mid]), " ".join(words[mid:])))

    if len(fragments) < 2:
        return None

    # swap halves between two random memories
    a, b = rng.sample(range(len(fragments)), 2)
    recombined = f"{fragments[a][0]} {fragments[b][1]}"
    return f"New thought from recombination: {recombined}"