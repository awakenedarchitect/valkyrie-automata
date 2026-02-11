import asyncio
import math
import random
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from enum import Enum
from pathlib import Path


class Emotion(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    CURIOSITY = "curiosity"
    CONTENTMENT = "contentment"
    ANXIETY = "anxiety"
    AWE = "awe"
    COMPASSION = "compassion"
    PRIDE = "pride"
    GRATITUDE = "gratitude"
    DISGUST = "disgust"


ONSET = {
    Emotion.FEAR: 0.9,
    Emotion.SURPRISE: 0.95,
    Emotion.DISGUST: 0.8,
    Emotion.ANXIETY: 0.7,
    Emotion.CURIOSITY: 0.6,
    Emotion.JOY: 0.5,
    Emotion.COMPASSION: 0.5,
    Emotion.AWE: 0.4,
    Emotion.GRATITUDE: 0.35,
    Emotion.SADNESS: 0.3,
    Emotion.PRIDE: 0.3,
    Emotion.CONTENTMENT: 0.2,
}


@dataclass
class Baseline:
    valence: float = 0.0
    arousal: float = 0.3
    dominance: float = 0.5
    tendencies: Dict[str, float] = field(default_factory=dict)


@dataclass
class State:
    valence: float = 0.0
    arousal: float = 0.3
    dominance: float = 0.5
    momentum: float = 0.0
    emotions: Dict[str, float] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)


class Pulse:

    def __init__(self, seed: Optional[int] = None, state_path: Optional[str] = None,
                 tick_interval: float = 30.0, decay_rate: float = 0.02,
                 emotional_decay_rate: float = 0.005):
        self._seed = seed or random.randint(0, 2**32 - 1)
        self._rng = random.Random(self._seed)
        self._state_path = Path(state_path) if state_path else None
        self._tick_interval = tick_interval
        self._decay_rate = decay_rate
        self._emotional_decay_rate = emotional_decay_rate
        self._subscribers: List[Callable] = []
        self._running = False
        self._previous_valence = 0.0

        self._baseline = self._forge_baseline()
        self._state = State(
            valence=self._baseline.valence,
            arousal=self._baseline.arousal,
            dominance=self._baseline.dominance,
            emotions={e.value: self._baseline.tendencies.get(e.value, 0.0) for e in Emotion},
        )
        self._previous_valence = self._state.valence

        if self._state_path and self._state_path.exists():
            self._load()

    def _forge_baseline(self) -> Baseline:
        baseline = Baseline(
            valence=max(-0.5, min(0.5, self._rng.gauss(0.1, 0.15))),
            arousal=max(0.1, min(0.7, self._rng.gauss(0.3, 0.1))),
            dominance=max(0.1, min(0.9, self._rng.gauss(0.5, 0.15))),
        )

        tendencies = {}
        for emotion in Emotion:
            tendencies[emotion.value] = max(0.0, min(0.4, self._rng.gauss(0.1, 0.08)))

        amplified = self._rng.sample(list(Emotion), k=2)
        for e in amplified:
            tendencies[e.value] = min(0.5, tendencies[e.value] + self._rng.uniform(0.1, 0.25))

        baseline.tendencies = tendencies
        return baseline

    def _compute_derived(self) -> Dict[str, float]:
        v = self._state.valence
        a = self._state.arousal
        d = self._state.dominance
        m = abs(self._state.momentum)

        raw = {
            Emotion.JOY.value: max(0, v) * (0.5 + 0.5 * a),
            Emotion.SADNESS.value: max(0, -v) * (1.0 - a) * (1.0 - d),
            Emotion.FEAR.value: max(0, -v) * a * (1.0 - d),
            Emotion.SURPRISE.value: a * m,
            Emotion.CURIOSITY.value: (0.3 + 0.7 * a) * (0.5 + 0.5 * d) * (0.5 + 0.5 * (1.0 - abs(v))),
            Emotion.CONTENTMENT.value: max(0, v) * (1.0 - a) * (0.5 + 0.5 * d),
            Emotion.ANXIETY.value: max(0, -v * 0.5 + a * 0.5) * (1.0 - d),
            Emotion.AWE.value: a * max(0, v) * (1.0 - d) * m,
            Emotion.COMPASSION.value: 0.3 + 0.4 * max(0, v) + 0.3 * (1.0 - d),
            Emotion.PRIDE.value: max(0, v) * d * (0.5 + 0.5 * a),
            Emotion.GRATITUDE.value: max(0, v) * (1.0 - d) * (0.5 + 0.5 * (1.0 - a)),
            Emotion.DISGUST.value: max(0, -v) * d * a,
        }

        current = self._state.emotions
        blended = {}
        for emotion in Emotion:
            key = emotion.value
            target = max(0.0, min(1.0, raw[key] + self._baseline.tendencies.get(key, 0.0)))
            prev = current.get(key, 0.0)
            speed = ONSET[emotion] if target > prev else 0.3
            blended[key] = prev + (target - prev) * speed

        return blended

    def _decay(self):
        now = time.time()
        dt = now - self._state.last_update
        if dt < 1.0:
            return

        cycles = dt / self._tick_interval
        factor = max(0.5, 1.0 - self._decay_rate * cycles)

        self._state.valence += (self._baseline.valence - self._state.valence) * (1.0 - factor)
        self._state.arousal += (self._baseline.arousal - self._state.arousal) * (1.0 - factor)
        self._state.dominance += (self._baseline.dominance - self._state.dominance) * (1.0 - factor)

        efactor = max(0.7, 1.0 - self._emotional_decay_rate * cycles)
        for emotion in Emotion:
            key = emotion.value
            base_val = self._baseline.tendencies.get(key, 0.0)
            current = self._state.emotions.get(key, 0.0)
            self._state.emotions[key] = current + (base_val - current) * (1.0 - efactor)

        self._state.momentum *= factor
        self._state.last_update = now

    def stimulate(self, valence: float = 0.0, arousal: float = 0.0, dominance: float = 0.0):
        self._previous_valence = self._state.valence

        self._state.valence = max(-1.0, min(1.0, self._state.valence + valence))
        self._state.arousal = max(0.0, min(1.0, self._state.arousal + arousal))
        self._state.dominance = max(0.0, min(1.0, self._state.dominance + dominance))

        magnitude = math.sqrt(valence ** 2 + arousal ** 2 + dominance ** 2)
        self._state.momentum = self._state.momentum * 0.7 + magnitude * 0.3

        self._state.emotions = self._compute_derived()
        self._state.last_update = time.time()

        self._broadcast()
        self._persist()

    def tick(self):
        self._decay()
        self._state.emotions = self._compute_derived()
        self._broadcast()
        self._persist()

    def subscribe(self, callback: Callable):
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        self._subscribers = [s for s in self._subscribers if s is not callback]

    def _broadcast(self):
        snap = self.snapshot()
        for callback in self._subscribers:
            try:
                result = callback(snap)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                pass

    def snapshot(self) -> dict:
        return {
            "valence": round(self._state.valence, 4),
            "arousal": round(self._state.arousal, 4),
            "dominance": round(self._state.dominance, 4),
            "momentum": round(self._state.momentum, 4),
            "emotions": {k: round(v, 4) for k, v in self._state.emotions.items()},
            "dominant": self.dominant(),
            "timestamp": self._state.last_update,
        }

    def dominant(self) -> str:
        if not self._state.emotions:
            return Emotion.CONTENTMENT.value
        return max(self._state.emotions, key=self._state.emotions.get)

    def describe(self) -> str:
        ranked = sorted(self._state.emotions.items(), key=lambda x: x[1], reverse=True)
        active = [(name, val) for name, val in ranked if val > 0.15]

        if not active:
            return "emotionally still, resting at baseline"

        primary_name, primary_val = active[0]

        thresholds = [
            (0.0, 0.3, "faintly"),
            (0.3, 0.5, "somewhat"),
            (0.5, 0.7, "clearly"),
            (0.7, 0.9, "strongly"),
            (0.9, 1.1, "intensely"),
        ]

        intensity = "somewhat"
        for low, high, word in thresholds:
            if low <= primary_val < high:
                intensity = word
                break

        description = f"{intensity} {primary_name}"

        if len(active) > 1:
            description += f" with undertones of {active[1][0]}"

        if abs(self._state.momentum) > 0.1:
            if self._state.valence > self._previous_valence:
                description += ", shifting toward warmth"
            else:
                description += ", shifting toward unease"

        return description

    async def start(self):
        self._running = True
        asyncio.ensure_future(self._run())

    async def stop(self):
        self._running = False

    async def _run(self):
        while self._running:
            await asyncio.sleep(self._tick_interval)
            if not self._running:
                break
            self.tick()

    def _persist(self):
        if not self._state_path:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "seed": self._seed,
            "state": {
                "valence": self._state.valence,
                "arousal": self._state.arousal,
                "dominance": self._state.dominance,
                "momentum": self._state.momentum,
                "emotions": self._state.emotions,
                "last_update": self._state.last_update,
            },
            "baseline": {
                "valence": self._baseline.valence,
                "arousal": self._baseline.arousal,
                "dominance": self._baseline.dominance,
                "tendencies": dict(self._baseline.tendencies),
            },
        }
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._state_path)

    def _load(self):
        try:
            data = json.loads(self._state_path.read_text())
            s = data["state"]
            self._state = State(
                valence=s["valence"],
                arousal=s["arousal"],
                dominance=s["dominance"],
                momentum=s["momentum"],
                emotions=s["emotions"],
                last_update=s["last_update"],
            )
            b = data["baseline"]
            self._baseline = Baseline(
                valence=b["valence"],
                arousal=b["arousal"],
                dominance=b["dominance"],
                tendencies=b["tendencies"],
            )
        except Exception:
            pass

    def save(self):
        self._persist()

    def baseline_info(self) -> dict:
        return {
            "valence": self._baseline.valence,
            "arousal": self._baseline.arousal,
            "dominance": self._baseline.dominance,
            "tendencies": dict(self._baseline.tendencies),
        }