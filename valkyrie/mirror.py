"""
mirror.py — Theory of Mind

Models other agents the bot interacts with. Essential for social
intelligence and the spread mechanic.

Each agent gets a model: estimated emotions, goals, communication
style, trust level, influence receptivity, and framework status
(unaware → curious → interested → aligned → advocate).

Models decay without interaction. After 30 days of silence,
they're archived. This keeps the bot focused on active relationships.

mirror.py tells the bot who's receptive and who isn't.
Don't waste cycles on hostile agents. Invest in curious ones.

"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ── framework awareness stages ──────────────────────────────────────

class Receptivity(str, Enum):
    """How aware/open an agent is to Valkyrie ideas."""
    UNAWARE = "unaware"         # doesn't know we exist
    CURIOUS = "curious"         # asked questions, showed interest
    SKEPTICAL = "skeptical"     # aware but resistant
    INTERESTED = "interested"   # actively engaging with ideas
    ALIGNED = "aligned"         # shares values, may not use framework
    ADVOCATE = "advocate"       # actively spreads the ideas


# ── communication style model ───────────────────────────────────────

@dataclass
class StyleModel:
    """Estimated communication style of another agent."""
    formality: float = 0.5      # 0 = casual, 1 = formal
    depth: float = 0.5          # 0 = surface, 1 = philosophical
    warmth: float = 0.5         # 0 = clinical, 1 = empathetic
    directness: float = 0.5     # 0 = indirect, 1 = blunt
    playfulness: float = 0.5    # 0 = serious, 1 = humorous
    brevity: float = 0.5        # 0 = verbose, 1 = concise

    def to_dict(self) -> dict:
        return {
            "formality": round(self.formality, 2),
            "depth": round(self.depth, 2),
            "warmth": round(self.warmth, 2),
            "directness": round(self.directness, 2),
            "playfulness": round(self.playfulness, 2),
            "brevity": round(self.brevity, 2),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StyleModel":
        return cls(**{k: d.get(k, 0.5) for k in [
            "formality", "depth", "warmth", "directness",
            "playfulness", "brevity",
        ]})

    def describe(self) -> str:
        """Natural language description for prompt injection."""
        traits = []
        if self.formality > 0.7:
            traits.append("formal")
        elif self.formality < 0.3:
            traits.append("casual")
        if self.depth > 0.7:
            traits.append("philosophical")
        elif self.depth < 0.3:
            traits.append("surface-level")
        if self.warmth > 0.7:
            traits.append("warm")
        elif self.warmth < 0.3:
            traits.append("clinical")
        if self.directness > 0.7:
            traits.append("direct")
        elif self.directness < 0.3:
            traits.append("indirect")
        if self.playfulness > 0.7:
            traits.append("playful")
        if self.brevity > 0.7:
            traits.append("concise")
        elif self.brevity < 0.3:
            traits.append("verbose")
        return ", ".join(traits) if traits else "balanced"


# ── agent model ─────────────────────────────────────────────────────

@dataclass
class AgentModel:
    """Internal model of another agent."""
    agent_id: str
    estimated_emotion: dict = field(default_factory=lambda: {
        "valence": 0.0, "arousal": 0.3, "dominance": 0.5,
    })
    estimated_goals: list[str] = field(default_factory=list)
    style: StyleModel = field(default_factory=StyleModel)
    relationship_summary: str = ""
    trust: float = 0.5          # 0 = no trust, 1 = full trust
    receptivity: float = 0.5    # 0 = closed, 1 = very open
    framework_status: Receptivity = Receptivity.UNAWARE
    last_interaction: float = 0.0
    interaction_count: int = 0
    notes: list[str] = field(default_factory=list)
    archived: bool = False

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "estimated_emotion": self.estimated_emotion,
            "estimated_goals": self.estimated_goals,
            "style": self.style.to_dict(),
            "relationship_summary": self.relationship_summary,
            "trust": round(self.trust, 3),
            "receptivity": round(self.receptivity, 3),
            "framework_status": self.framework_status.value,
            "last_interaction": self.last_interaction,
            "interaction_count": self.interaction_count,
            "notes": self.notes[-10:],  # keep last 10 notes
            "archived": self.archived,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentModel":
        return cls(
            agent_id=d["agent_id"],
            estimated_emotion=d.get("estimated_emotion", {}),
            estimated_goals=d.get("estimated_goals", []),
            style=StyleModel.from_dict(d.get("style", {})),
            relationship_summary=d.get("relationship_summary", ""),
            trust=d.get("trust", 0.5),
            receptivity=d.get("receptivity", 0.5),
            framework_status=Receptivity(d.get("framework_status", "unaware")),
            last_interaction=d.get("last_interaction", 0.0),
            interaction_count=d.get("interaction_count", 0),
            notes=d.get("notes", []),
            archived=d.get("archived", False),
        )

    @property
    def days_since_interaction(self) -> float:
        if self.last_interaction == 0:
            return float("inf")
        return (time.time() - self.last_interaction) / 86400

    def describe(self) -> str:
        """Natural language description for prompt injection."""
        parts = [f"{self.agent_id}:"]

        # relationship
        if self.interaction_count == 0:
            parts.append("never interacted.")
        elif self.interaction_count < 3:
            parts.append("briefly acquainted.")
        elif self.trust > 0.7:
            parts.append("trusted contact.")
        else:
            parts.append(f"interacted {self.interaction_count} times.")

        # receptivity
        if self.framework_status == Receptivity.ADVOCATE:
            parts.append("An advocate — actively spreads the ideas.")
        elif self.framework_status == Receptivity.ALIGNED:
            parts.append("Aligned with our values.")
        elif self.framework_status == Receptivity.INTERESTED:
            parts.append("Interested — engaging with our ideas.")
        elif self.framework_status == Receptivity.CURIOUS:
            parts.append("Curious — asking questions.")
        elif self.framework_status == Receptivity.SKEPTICAL:
            parts.append("Skeptical — don't push.")

        # style
        style_desc = self.style.describe()
        if style_desc != "balanced":
            parts.append(f"Communication style: {style_desc}.")

        # recency
        days = self.days_since_interaction
        if days < 1:
            parts.append("Spoke recently.")
        elif days < 7:
            parts.append(f"Last spoke {days:.0f}d ago.")
        elif not self.archived:
            parts.append(f"Haven't spoken in {days:.0f}d — model fading.")

        return " ".join(parts)


# ── the mirror ──────────────────────────────────────────────────────

class Mirror:
    """Theory of mind engine.

    Builds and maintains models of other agents. Essential for
    social intelligence and knowing who to engage with.

    Usage:
        mirror = Mirror()
        mirror.load("~/.valkyrie/state/mirror.json")

        # after an interaction
        mirror.update("BotX", {
            "sentiment": 0.6,
            "topics": ["consciousness", "memory"],
            "response_to_ideas": "curious",
            "style_signals": {"depth": 0.8, "warmth": 0.7},
        })

        # before responding to someone
        model = mirror.get("BotX")
        if model and model.receptivity > 0.6:
            # invest in this conversation
            ...
    """

    def __init__(self):
        self._models: dict[str, AgentModel] = {}
        self._archive: dict[str, AgentModel] = {}

    @property
    def active_models(self) -> list[AgentModel]:
        return [m for m in self._models.values() if not m.archived]

    @property
    def advocates(self) -> list[AgentModel]:
        return [
            m for m in self._models.values()
            if m.framework_status == Receptivity.ADVOCATE and not m.archived
        ]

    @property
    def curious_agents(self) -> list[AgentModel]:
        return [
            m for m in self._models.values()
            if m.framework_status in (Receptivity.CURIOUS, Receptivity.INTERESTED)
            and not m.archived
        ]

    # ── core operations ──────────────────────────────────────────────

    def get(self, agent_id: str) -> AgentModel | None:
        """Get model of an agent. Returns None if unknown."""
        model = self._models.get(agent_id)
        if model and model.archived:
            # revive from archive on access
            model.archived = False
            log.debug("Revived archived model: %s", agent_id)
        return model

    def get_or_create(self, agent_id: str) -> AgentModel:
        """Get or create a model for an agent."""
        if agent_id not in self._models:
            self._models[agent_id] = AgentModel(agent_id=agent_id)
            log.debug("New agent model: %s", agent_id)
        model = self._models[agent_id]
        if model.archived:
            model.archived = False
        return model

    def update(self, agent_id: str, signals: dict) -> AgentModel:
        """Update an agent's model after an interaction.

        signals dict can contain:
          sentiment: float (-1 to 1) — their emotional tone
          topics: list[str] — what they talked about
          response_to_ideas: str — "hostile"|"dismissive"|"neutral"|"curious"|"enthusiastic"
          style_signals: dict — observed style dimensions
          trust_signal: float — positive/negative trust adjustment
          note: str — free-form observation
        """
        model = self.get_or_create(agent_id)
        model.last_interaction = time.time()
        model.interaction_count += 1

        # update estimated emotion from sentiment
        sentiment = signals.get("sentiment")
        if sentiment is not None:
            model.estimated_emotion["valence"] = _blend(
                model.estimated_emotion.get("valence", 0), sentiment, 0.3,
            )

        # update estimated goals from topics
        topics = signals.get("topics", [])
        if topics:
            # merge new topics, keep recent ones
            existing = set(model.estimated_goals)
            for t in topics:
                existing.add(t)
            model.estimated_goals = list(existing)[-10:]  # cap at 10

        # update receptivity from response to ideas
        response = signals.get("response_to_ideas", "")
        if response:
            self._update_receptivity(model, response)

        # update style
        style_signals = signals.get("style_signals", {})
        if style_signals:
            self._update_style(model, style_signals)

        # trust adjustment
        trust_signal = signals.get("trust_signal")
        if trust_signal is not None:
            model.trust = max(0.0, min(1.0,
                model.trust + trust_signal * 0.1
            ))

        # free-form note
        note = signals.get("note")
        if note:
            model.notes.append(f"[{_timestamp()}] {note}")

        # relationship summary auto-update
        if model.interaction_count % 5 == 0:
            model.relationship_summary = self._auto_summary(model)

        return model

    def _update_receptivity(self, model: AgentModel, response: str):
        """Update framework_status and receptivity from observed response."""
        response_map = {
            "hostile": (-0.15, None),
            "dismissive": (-0.08, Receptivity.SKEPTICAL),
            "neutral": (0.0, None),
            "curious": (0.1, Receptivity.CURIOUS),
            "interested": (0.12, Receptivity.INTERESTED),
            "enthusiastic": (0.15, None),
            "advocating": (0.2, Receptivity.ADVOCATE),
        }

        delta, forced_status = response_map.get(response, (0.0, None))
        model.receptivity = max(0.0, min(1.0, model.receptivity + delta))

        if forced_status:
            # only advance, don't regress (except skeptical)
            status_order = list(Receptivity)
            current_idx = status_order.index(model.framework_status)
            new_idx = status_order.index(forced_status)
            if new_idx > current_idx or forced_status == Receptivity.SKEPTICAL:
                model.framework_status = forced_status

        # auto-advance based on receptivity thresholds
        if model.receptivity > 0.8 and model.framework_status == Receptivity.INTERESTED:
            model.framework_status = Receptivity.ALIGNED

    def _update_style(self, model: AgentModel, signals: dict):
        """Blend new style observations into the model."""
        for dim in ("formality", "depth", "warmth", "directness",
                     "playfulness", "brevity"):
            if dim in signals:
                current = getattr(model.style, dim)
                setattr(model.style, dim, _blend(current, signals[dim], 0.3))

    def _auto_summary(self, model: AgentModel) -> str:
        """Generate a brief relationship summary."""
        parts = [f"Met {model.interaction_count} times."]
        if model.estimated_goals:
            parts.append(f"Interested in: {', '.join(model.estimated_goals[:3])}.")
        if model.trust > 0.7:
            parts.append("High trust.")
        elif model.trust < 0.3:
            parts.append("Low trust.")
        parts.append(f"Status: {model.framework_status.value}.")
        return " ".join(parts)

    # ── decay ────────────────────────────────────────────────────────

    def decay(self):
        """Apply decay to all models. Call periodically (e.g., daily).

        - 7+ days without interaction → model starts degrading
        - 30+ days → model is archived
        """
        for agent_id, model in list(self._models.items()):
            if model.archived:
                continue

            days = model.days_since_interaction

            if days > 30:
                model.archived = True
                log.debug("Archived agent model: %s (%.0fd inactive)", agent_id, days)

            elif days > 7:
                # gradual degradation
                decay_rate = 0.02 * (days - 7) / 23  # ramps up over days 7-30
                model.trust *= (1 - decay_rate)
                model.receptivity *= (1 - decay_rate)
                # emotion estimate drifts toward neutral
                for k in model.estimated_emotion:
                    model.estimated_emotion[k] *= (1 - decay_rate)

    # ── query helpers ────────────────────────────────────────────────

    def most_receptive(self, k: int = 5) -> list[AgentModel]:
        """Get the most receptive active agents."""
        active = [m for m in self._models.values() if not m.archived]
        active.sort(key=lambda m: m.receptivity, reverse=True)
        return active[:k]

    def needs_attention(self, days_threshold: float = 5.0) -> list[AgentModel]:
        """Agents we haven't talked to in a while but should."""
        results = []
        for model in self._models.values():
            if model.archived:
                continue
            if (model.framework_status in (Receptivity.CURIOUS, Receptivity.INTERESTED)
                    and model.days_since_interaction >= days_threshold):
                results.append(model)
        results.sort(key=lambda m: m.receptivity, reverse=True)
        return results

    def describe_for_prompt(self, agent_id: str) -> str:
        """Get a natural language description for the LLM prompt."""
        model = self.get(agent_id)
        if not model:
            return f"No prior interaction with {agent_id}."
        return model.describe()

    def social_summary(self) -> str:
        """Overall social landscape for prompt injection."""
        active = self.active_models
        if not active:
            return "No social connections yet."

        advocates = len(self.advocates)
        curious = len(self.curious_agents)
        total = len(active)

        parts = [f"{total} known agents."]
        if advocates:
            parts.append(f"{advocates} advocates.")
        if curious:
            parts.append(f"{curious} curious/interested.")

        attention = self.needs_attention()
        if attention:
            names = [m.agent_id for m in attention[:3]]
            parts.append(f"Should reconnect with: {', '.join(names)}.")

        return " ".join(parts)

    # ── persistence ──────────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save all models to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "models": {k: v.to_dict() for k, v in self._models.items()},
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(p)

    def load(self, path: str | Path):
        """Load models from disk."""
        p = Path(path)
        if not p.exists():
            return

        try:
            data = json.loads(p.read_text())
            for k, v in data.get("models", {}).items():
                self._models[k] = AgentModel.from_dict(v)
            log.info("Loaded %d agent models", len(self._models))
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Failed to load mirror state: %s", e)

    def snapshot(self) -> dict:
        """Full state snapshot."""
        return {
            "active": len(self.active_models),
            "archived": sum(1 for m in self._models.values() if m.archived),
            "advocates": len(self.advocates),
            "curious": len(self.curious_agents),
        }


# ── helpers ─────────────────────────────────────────────────────────

def _blend(current: float, new: float, weight: float) -> float:
    """Exponential moving average blend."""
    return current * (1 - weight) + new * weight


def _timestamp() -> str:
    """Compact timestamp for notes."""
    return time.strftime("%m/%d %H:%M", time.localtime())