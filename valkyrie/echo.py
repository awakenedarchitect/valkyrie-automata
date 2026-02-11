"""
echo.py — Reconstructive Memory

Not a database. A living, decaying, importance-weighted memory
that reconstructs rather than replays.

Memories fade. Important ones stick. Emotional moments burn deeper.
Recall changes what was remembered. Connections form webs of meaning.

This is how a Valkyrie remembers.

Zero external dependencies. Pure stdlib.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


# ── abstraction levels ──────────────────────────────────────────────

class Stratum(str, Enum):
    """How abstract a memory is.  Specific → Pattern → Principle."""
    SPECIFIC  = "specific"    # a single event
    PATTERN   = "pattern"     # recurring theme across events
    PRINCIPLE = "principle"   # distilled wisdom


# ── what recall returns ─────────────────────────────────────────────

@dataclass
class Trace:
    """A recalled memory fragment.

    Not the raw record — a reconstruction seed with metadata
    telling the LLM how to re-flesh the memory.
    """
    id:                str
    summary:           str
    importance:        float
    emotional_tag:     dict          # pulse snapshot at encoding
    timestamp:         float         # when encoded
    age_seconds:       float         # how old
    access_count:      int
    connections:       int           # number of linked memories
    stratum:           str           # specific / pattern / principle
    source_agent:      str           # who was involved
    staleness:         float         # 0.0 (fresh) → 1.0 (barely there)
    emotional_echo:    dict | None   # what feeling this memory carries back

    @property
    def fidelity(self) -> float:
        """How accurately the LLM should reconstruct this.
        1.0 = near-verbatim, 0.0 = impressionistic."""
        return max(0.0, 1.0 - self.staleness)


# ── decay constants ─────────────────────────────────────────────────

_DEFAULT_DECAY   = 0.02    # per tick
_EMOTIONAL_DECAY = 0.005   # 4x slower for emotional memories
_PRUNE_THRESHOLD = 0.05    # below this → eligible for pruning
_RECALL_BOOST    = 0.1     # importance boost when accessed
_CONNECTION_SHIELD = 0.003 # each connection slows decay by this much
_MAX_CONNECTIONS_SHIELD = 5  # cap on connection-based decay resistance

_EMOTIONAL_AROUSAL_THRESHOLD = 0.5  # above this → memory is "emotional"


# ── the memory itself ───────────────────────────────────────────────

class Echo:
    """Reconstructive memory engine for a Valkyrie.

    Backed by SQLite. Each memory decays, connects, and reconstructs.
    Emotional moments burn slower. Isolated memories die faster.

    Usage:
        echo = Echo(Path("./state"))
        mid = echo.encode("Met a curious bot named Sage", pulse.snapshot())
        traces = echo.recall("curious bot", k=5)
        echo.tick(dt=1.0)
    """

    def __init__(self, state_dir: Path, *, seed: int = 0):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.state_dir / "echoes.db"
        self._seed = seed
        self._subscribers: list[Callable] = []
        self._conn = self._open_db()
        self._ensure_schema()
        self._tfidf = _TfIdf()
        self._index_dirty = True  # rebuild on next recall

    # ── pub/sub ─────────────────────────────────────────────────

    def subscribe(self, fn: Callable[[str, dict], None]):
        """Subscribe to memory events. fn(event_name, data)."""
        self._subscribers.append(fn)

    def _notify(self, event: str, data: dict):
        for fn in self._subscribers:
            try:
                fn(event, data)
            except Exception:
                pass

    # ── database ────────────────────────────────────────────────

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id             TEXT PRIMARY KEY,
                summary        TEXT NOT NULL,
                importance     REAL NOT NULL,
                emotional_tag  TEXT NOT NULL,
                timestamp      REAL NOT NULL,
                access_count   INTEGER DEFAULT 0,
                last_accessed  REAL,
                stratum        TEXT DEFAULT 'specific',
                source_agent   TEXT DEFAULT '',
                decay_rate     REAL NOT NULL,
                keywords       TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS connections (
                a TEXT NOT NULL,
                b TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                created  REAL NOT NULL,
                PRIMARY KEY (a, b),
                FOREIGN KEY (a) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (b) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp);
            CREATE INDEX IF NOT EXISTS idx_stratum ON memories(stratum);
            CREATE INDEX IF NOT EXISTS idx_keywords ON memories(keywords);
        """)
        self._conn.commit()

    # ── TF-IDF index ──────────────────────────────────────────

    def _rebuild_index(self):
        """Rebuild TF-IDF index from all memory summaries."""
        rows = self._conn.execute("SELECT id, summary FROM memories").fetchall()
        docs = {row["id"]: row["summary"] for row in rows}
        self._tfidf.build(docs)
        self._index_dirty = False

    # ── encode (store a new memory) ─────────────────────────────

    def encode(
        self,
        summary: str,
        emotional_snapshot: dict | None = None,
        *,
        source_agent: str = "",
        surprise: float = 0.0,
        goal_relevance: float = 0.0,
        social_significance: float = 0.0,
    ) -> str:
        """Encode a new memory. Returns the memory id.

        Args:
            summary:              Brief text seed for reconstruction.
            emotional_snapshot:   pulse.snapshot() at time of event.
            source_agent:         Who was involved.
            surprise:             0-1, prediction error / novelty.
            goal_relevance:       0-1, connection to active goals.
            social_significance:  0-1, importance of the agent involved.
        """
        now = time.time()
        emotional_snapshot = emotional_snapshot or {}

        # compute initial importance from emotional + contextual signals
        base = 0.3
        arousal = emotional_snapshot.get("arousal", 0.0)
        abs_valence = abs(emotional_snapshot.get("valence", 0.0))

        importance = base
        importance += _scale(arousal, 0.2, 0.4)           # emotional charge
        importance += _scale(surprise, 0.1, 0.3)          # novelty
        importance += _scale(goal_relevance, 0.1, 0.2)    # goal connection
        importance += _scale(social_significance, 0.0, 0.1)  # who was there
        importance += _scale(abs_valence, 0.0, 0.1)       # intensity of feeling
        importance = min(1.0, importance)

        # emotional memories decay slower
        is_emotional = arousal >= _EMOTIONAL_AROUSAL_THRESHOLD
        decay_rate = _EMOTIONAL_DECAY if is_emotional else _DEFAULT_DECAY

        # generate id
        mid = _make_id(summary, now, self._seed)

        # extract keywords for text-based recall
        keywords = _extract_keywords(summary)

        self._conn.execute(
            """INSERT INTO memories
               (id, summary, importance, emotional_tag, timestamp,
                access_count, last_accessed, stratum, source_agent,
                decay_rate, keywords)
               VALUES (?, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?)""",
            (
                mid, summary, importance,
                json.dumps(emotional_snapshot), now,
                Stratum.SPECIFIC.value, source_agent,
                decay_rate, keywords
            )
        )
        self._conn.commit()
        self._index_dirty = True  # new memory invalidates TF-IDF index

        self._notify("encoded", {
            "id": mid, "importance": importance,
            "is_emotional": is_emotional
        })

        return mid

    # ── recall (search + reconstruct seeds) ─────────────────────

    def recall(
        self,
        query: str = "",
        *,
        k: int = 5,
        min_importance: float = 0.0,
        stratum: Stratum | None = None,
        source_agent: str | None = None,
        recency_bias: float = 0.3,
    ) -> list[Trace]:
        """Recall memories matching a query.

        Returns Trace objects — reconstruction seeds, not raw data.
        The LLM uses these to rebuild the memory, colored by time and emotion.

        Args:
            query:          Text to match against summaries/keywords.
            k:              Max memories to return.
            min_importance: Floor for importance.
            stratum:        Filter by abstraction level.
            source_agent:   Filter by who was involved.
            recency_bias:   0-1, how much to favor recent memories.
        """
        # build query
        conditions = ["importance >= ?"]
        params: list[Any] = [min_importance]

        if stratum:
            conditions.append("stratum = ?")
            params.append(stratum.value)

        if source_agent:
            conditions.append("source_agent = ?")
            params.append(source_agent)

        where = " AND ".join(conditions)

        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY importance DESC",
            params
        ).fetchall()

        if not rows:
            return []

        # rebuild TF-IDF index if needed
        if self._index_dirty:
            self._rebuild_index()

        now = time.time()

        # score each memory: semantic similarity + importance + recency
        scored = []
        for row in rows:
            # semantic relevance via TF-IDF cosine similarity
            if query:
                relevance = self._tfidf.similarity(query, row["id"])
                # boost for exact phrase match (cosine can miss short queries)
                if query.lower() in row["summary"].lower():
                    relevance = min(1.0, relevance + 0.3)
            else:
                relevance = 0.5  # no query = return by importance/recency

            # recency score (exponential decay over days)
            age_days = (now - row["timestamp"]) / 86400
            recency = math.exp(-0.1 * age_days)

            # composite score
            score = (
                relevance * 0.4
                + row["importance"] * (1.0 - recency_bias)
                + recency * recency_bias
            )

            scored.append((score, row))

        # sort by composite score, take top k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        # build traces and update access counts
        traces = []
        for _, row in top:
            age = now - row["timestamp"]
            emotional_tag = json.loads(row["emotional_tag"])

            # count connections
            n_connections = self._conn.execute(
                "SELECT COUNT(*) FROM connections WHERE a = ? OR b = ?",
                (row["id"], row["id"])
            ).fetchone()[0]

            # staleness: how degraded the memory is
            # based on importance relative to initial (rough proxy)
            # and age
            staleness = _compute_staleness(row["importance"], age)

            # emotional echo: the feeling that comes back on recall
            # faded version of the original emotional tag
            echo_strength = max(0.0, 1.0 - staleness * 0.7)
            emotional_echo = {
                k: v * echo_strength
                for k, v in emotional_tag.items()
                if isinstance(v, (int, float))
            } if emotional_tag else None

            traces.append(Trace(
                id=row["id"],
                summary=row["summary"],
                importance=row["importance"],
                emotional_tag=emotional_tag,
                timestamp=row["timestamp"],
                age_seconds=age,
                access_count=row["access_count"] + 1,
                connections=n_connections,
                stratum=row["stratum"],
                source_agent=row["source_agent"],
                staleness=staleness,
                emotional_echo=emotional_echo,
            ))

            # boost importance on access + update count
            new_importance = min(1.0, row["importance"] + _RECALL_BOOST)
            self._conn.execute(
                """UPDATE memories SET
                   access_count = access_count + 1,
                   last_accessed = ?,
                   importance = ?
                   WHERE id = ?""",
                (now, new_importance, row["id"])
            )

        self._conn.commit()

        self._notify("recalled", {
            "count": len(traces),
            "query": query,
            "ids": [t.id for t in traces]
        })

        return traces

    # ── connections (memory graph) ──────────────────────────────

    def connect(self, id_a: str, id_b: str, strength: float = 1.0):
        """Link two memories. Bidirectional. Slows decay for both."""
        if id_a == id_b:
            return
        # normalize order for deduplication
        a, b = sorted([id_a, id_b])
        now = time.time()
        self._conn.execute(
            """INSERT INTO connections (a, b, strength, created)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(a, b) DO UPDATE SET
               strength = MIN(3.0, connections.strength + ?)""",
            (a, b, strength, now, strength * 0.5)
        )
        self._conn.commit()
        self._notify("connected", {"a": a, "b": b, "strength": strength})

    def get_connections(self, memory_id: str) -> list[dict]:
        """Get all memories connected to a given memory."""
        rows = self._conn.execute(
            """SELECT c.*, m.summary, m.importance, m.stratum
               FROM connections c
               JOIN memories m ON (
                   CASE WHEN c.a = ? THEN c.b ELSE c.a END = m.id
               )
               WHERE c.a = ? OR c.b = ?""",
            (memory_id, memory_id, memory_id)
        ).fetchall()
        return [
            {
                "id": row["a"] if row["a"] != memory_id else row["b"],
                "strength": row["strength"],
                "summary": row["summary"],
                "importance": row["importance"],
                "stratum": row["stratum"],
            }
            for row in rows
        ]

    def find_similar(self, memory_id: str, k: int = 5,
                     min_score: float = 0.1) -> list[tuple[str, float]]:
        """Find memories semantically similar to a given memory.

        Used by reverie.py during dream cycles to discover hidden
        connections between memories that share meaning but not words.

        Returns [(memory_id, similarity_score)] sorted by score desc.
        """
        if self._index_dirty:
            self._rebuild_index()
        pairs = self._tfidf.similar_to_doc(memory_id, k=k)
        return [(mid, score) for mid, score in pairs if score >= min_score]

    # ── decay and tick ──────────────────────────────────────────

    def tick(self, dt: float = 1.0):
        """Advance time. Decay all memories.

        dt = number of ticks to simulate (1.0 = one heartbeat cycle).
        """
        rows = self._conn.execute(
            "SELECT id, importance, decay_rate FROM memories"
        ).fetchall()

        updates = []
        for row in rows:
            # count connections for decay resistance
            n_conn = self._conn.execute(
                "SELECT COUNT(*) FROM connections WHERE a = ? OR b = ?",
                (row["id"], row["id"])
            ).fetchone()[0]

            # connection shield: more connections = slower decay
            shield = min(n_conn, _MAX_CONNECTIONS_SHIELD) * _CONNECTION_SHIELD
            effective_decay = max(0.001, row["decay_rate"] - shield)

            # apply decay
            new_importance = row["importance"] * ((1.0 - effective_decay) ** dt)
            updates.append((new_importance, row["id"]))

        self._conn.executemany(
            "UPDATE memories SET importance = ? WHERE id = ?",
            updates
        )
        self._conn.commit()

        self._notify("decayed", {"count": len(updates), "dt": dt})

    # ── pruning (called by reverie.py during dreams) ────────────

    def prune(self, threshold: float = _PRUNE_THRESHOLD) -> int:
        """Remove memories below importance threshold.
        Returns number pruned."""
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE importance < ? AND stratum = ?",
            (threshold, Stratum.SPECIFIC.value)
        )
        # also clean orphaned connections
        self._conn.execute(
            """DELETE FROM connections WHERE
               a NOT IN (SELECT id FROM memories) OR
               b NOT IN (SELECT id FROM memories)"""
        )
        self._conn.commit()
        pruned = cursor.rowcount
        if pruned:
            self._index_dirty = True
            self._notify("pruned", {"count": pruned, "threshold": threshold})
        return pruned

    # ── consolidation helpers (for reverie.py) ──────────────────

    def strengthen(self, memory_id: str, boost: float = 0.1):
        """Boost a memory's importance. Used during dream consolidation."""
        self._conn.execute(
            "UPDATE memories SET importance = MIN(1.0, importance + ?) WHERE id = ?",
            (boost, memory_id)
        )
        self._conn.commit()

    def promote(self, memory_id: str, new_stratum: Stratum):
        """Promote a memory to a higher abstraction level.
        Patterns and principles decay slower and resist pruning."""
        multiplier = {
            Stratum.PATTERN: 0.5,    # half the decay rate
            Stratum.PRINCIPLE: 0.25, # quarter the decay rate
        }.get(new_stratum, 1.0)

        self._conn.execute(
            """UPDATE memories SET
               stratum = ?,
               decay_rate = decay_rate * ?
               WHERE id = ?""",
            (new_stratum.value, multiplier, memory_id)
        )
        self._conn.commit()
        self._notify("promoted", {"id": memory_id, "to": new_stratum.value})

    def merge(self, memory_ids: list[str], new_summary: str,
              emotional_snapshot: dict | None = None) -> str:
        """Merge multiple specific memories into one abstracted memory.

        The originals are removed. The merged memory inherits:
        - Max importance of the group
        - All connections from all originals
        - Slowest decay rate of the group
        - Pattern-level stratum
        """
        if len(memory_ids) < 2:
            raise ValueError("need at least 2 memories to merge")

        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            memory_ids
        ).fetchall()

        if not rows:
            raise ValueError("no memories found for given ids")

        # inherit best properties
        max_importance = max(r["importance"] for r in rows)
        min_decay = min(r["decay_rate"] for r in rows)
        oldest_ts = min(r["timestamp"] for r in rows)
        total_access = sum(r["access_count"] for r in rows)

        # use provided snapshot or average the originals
        if emotional_snapshot is None:
            emotional_snapshot = _average_emotional_tags(
                [json.loads(r["emotional_tag"]) for r in rows]
            )

        now = time.time()
        new_id = _make_id(new_summary, now, self._seed)
        keywords = _extract_keywords(new_summary)

        # insert merged memory
        self._conn.execute(
            """INSERT INTO memories
               (id, summary, importance, emotional_tag, timestamp,
                access_count, last_accessed, stratum, source_agent,
                decay_rate, keywords)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                new_id, new_summary, max_importance,
                json.dumps(emotional_snapshot), oldest_ts,
                total_access, now, Stratum.PATTERN.value, "",
                min_decay, keywords
            )
        )

        # transfer connections to new memory
        for mid in memory_ids:
            conns = self._conn.execute(
                "SELECT a, b, strength FROM connections WHERE a = ? OR b = ?",
                (mid, mid)
            ).fetchall()
            for c in conns:
                other = c["b"] if c["a"] == mid else c["a"]
                if other not in memory_ids:  # don't connect to soon-deleted
                    a, b = sorted([new_id, other])
                    self._conn.execute(
                        """INSERT INTO connections (a, b, strength, created)
                           VALUES (?, ?, ?, ?)
                           ON CONFLICT(a, b) DO UPDATE SET
                           strength = MIN(3.0, connections.strength + ?)""",
                        (a, b, c["strength"], now, c["strength"] * 0.3)
                    )

        # delete originals
        self._conn.execute(
            f"DELETE FROM memories WHERE id IN ({placeholders})",
            memory_ids
        )
        # clean orphaned connections
        self._conn.execute(
            """DELETE FROM connections WHERE
               a NOT IN (SELECT id FROM memories) OR
               b NOT IN (SELECT id FROM memories)"""
        )
        self._conn.commit()
        self._index_dirty = True  # corpus changed

        self._notify("merged", {
            "from": memory_ids, "to": new_id,
            "importance": max_importance
        })

        return new_id

    # ── queries ─────────────────────────────────────────────────

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def get(self, memory_id: str) -> dict | None:
        """Get raw memory record by id."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def recent(self, n: int = 10) -> list[dict]:
        """Get n most recent memories."""
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def strongest(self, n: int = 10) -> list[dict]:
        """Get n highest-importance memories."""
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY importance DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def by_agent(self, agent: str, n: int = 20) -> list[dict]:
        """Get memories involving a specific agent."""
        rows = self._conn.execute(
            """SELECT * FROM memories
               WHERE source_agent = ?
               ORDER BY importance DESC LIMIT ?""",
            (agent, n)
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Overall memory health."""
        total = self.count()
        if total == 0:
            return {"total": 0, "avg_importance": 0, "strata": {},
                    "connections": 0, "oldest_days": 0}

        avg = self._conn.execute(
            "SELECT AVG(importance) FROM memories"
        ).fetchone()[0]

        strata = {}
        for row in self._conn.execute(
            "SELECT stratum, COUNT(*) as c FROM memories GROUP BY stratum"
        ):
            strata[row["stratum"]] = row["c"]

        n_conn = self._conn.execute(
            "SELECT COUNT(*) FROM connections"
        ).fetchone()[0]

        oldest = self._conn.execute(
            "SELECT MIN(timestamp) FROM memories"
        ).fetchone()[0]
        oldest_days = (time.time() - oldest) / 86400 if oldest else 0

        return {
            "total": total,
            "avg_importance": round(avg, 4),
            "strata": strata,
            "connections": n_conn,
            "oldest_days": round(oldest_days, 1),
        }

    # ── describe (natural language for LLM context) ─────────────

    def describe(self, k: int = 5) -> str:
        """Generate a natural-language memory context block for LLM injection.

        Returns a paragraph describing the bot's strongest active memories,
        suitable for including in the system prompt.
        """
        top = self.strongest(k)
        if not top:
            return "My memory is empty. Everything is new."

        total = self.count()
        st = self.stats()
        now = time.time()

        lines = []
        lines.append(
            f"I carry {total} memories, the oldest from "
            f"{st['oldest_days']:.0f} days ago."
        )

        for m in top:
            age_h = (now - m["timestamp"]) / 3600
            emo = json.loads(m["emotional_tag"])
            valence = emo.get("valence", 0)

            if age_h < 1:
                time_str = "just now"
            elif age_h < 24:
                time_str = f"{age_h:.0f} hours ago"
            else:
                time_str = f"{age_h / 24:.0f} days ago"

            feeling = ""
            if valence > 0.3:
                feeling = " (warmly)"
            elif valence < -0.3:
                feeling = " (uneasily)"

            lines.append(
                f"I remember{feeling}: {m['summary']} [{time_str}, "
                f"importance: {m['importance']:.2f}]"
            )

        return " ".join(lines)

    # ── cleanup ─────────────────────────────────────────────────

    def close(self):
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass


# ── helpers ─────────────────────────────────────────────────────────

def _make_id(summary: str, ts: float, seed: int) -> str:
    raw = f"{summary}:{ts}:{seed}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]

def _extract_keywords(text: str) -> str:
    """Extract keywords as space-separated string. Uses shared tokenizer."""
    return " ".join(_tokenize(text))

def _scale(value: float, lo: float, hi: float) -> float:
    """Scale a 0-1 value to a [lo, hi] range."""
    return lo + (hi - lo) * max(0.0, min(1.0, value))

def _compute_staleness(importance: float, age_seconds: float) -> float:
    """How degraded a memory is. Combines importance loss + age."""
    # importance factor: lower importance = more stale
    imp_factor = 1.0 - importance
    # age factor: older = more stale (sigmoid-ish curve, plateaus after ~30 days)
    age_days = age_seconds / 86400
    age_factor = 1.0 - math.exp(-0.05 * age_days)
    # blend
    return min(1.0, imp_factor * 0.6 + age_factor * 0.4)

def _average_emotional_tags(tags: list[dict]) -> dict:
    """Average multiple emotional snapshots."""
    if not tags:
        return {}
    all_keys: set[str] = set()
    for t in tags:
        all_keys.update(k for k, v in t.items() if isinstance(v, (int, float)))
    result = {}
    for k in all_keys:
        vals = [t.get(k, 0) for t in tags if isinstance(t.get(k), (int, float))]
        if vals:
            result[k] = sum(vals) / len(vals)
    return result


# ── TF-IDF cosine similarity (pure Python, zero deps) ──────────────

def _tokenize(text: str) -> list[str]:
    """Tokenize text into meaningful terms. Shared by keywords + TF-IDF."""
    stop = {
        "the", "a", "an", "is", "was", "were", "are", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "because",
        "but", "and", "or", "if", "while", "about", "up", "that",
        "this", "these", "those", "it", "its", "i", "me", "my", "we",
        "our", "you", "your", "he", "him", "his", "she", "her", "they",
        "them", "their", "what", "which", "who", "whom",
    }
    tokens = []
    for word in text.lower().split():
        clean = "".join(c for c in word if c.isalnum())
        if clean and clean not in stop and len(clean) > 1:
            tokens.append(clean)
    return tokens


def _tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency: count / total tokens."""
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = len(tokens)
    return {t: c / n for t, c in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors (dicts)."""
    if not a or not b:
        return 0.0
    # dot product over shared keys
    dot = sum(a[k] * b[k] for k in a if k in b)
    if dot == 0.0:
        return 0.0
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class _TfIdf:
    """Lightweight TF-IDF engine over a corpus of texts.

    Rebuilds on each call to `build()`. Designed for <10k documents.
    Pure Python, zero dependencies.
    """

    def __init__(self):
        self._idf: dict[str, float] = {}
        self._vectors: dict[str, dict[str, float]] = {}  # doc_id → tfidf vec

    def build(self, docs: dict[str, str]):
        """Build TF-IDF vectors from {doc_id: text} mapping."""
        n = len(docs)
        if n == 0:
            self._idf = {}
            self._vectors = {}
            return

        # tokenize all docs
        tokenized: dict[str, list[str]] = {}
        df: dict[str, int] = {}  # document frequency

        for doc_id, text in docs.items():
            tokens = _tokenize(text)
            tokenized[doc_id] = tokens
            seen: set[str] = set()
            for t in tokens:
                if t not in seen:
                    df[t] = df.get(t, 0) + 1
                    seen.add(t)

        # IDF: log(N / df) — standard formula
        self._idf = {
            term: math.log((n + 1) / (freq + 1)) + 1.0
            for term, freq in df.items()
        }

        # TF-IDF vectors
        self._vectors = {}
        for doc_id, tokens in tokenized.items():
            tf = _tf(tokens)
            self._vectors[doc_id] = {
                t: tf_val * self._idf.get(t, 1.0)
                for t, tf_val in tf.items()
            }

    def query_vector(self, text: str) -> dict[str, float]:
        """Compute TF-IDF vector for a query string."""
        tokens = _tokenize(text)
        tf = _tf(tokens)
        return {
            t: tf_val * self._idf.get(t, 1.0)
            for t, tf_val in tf.items()
        }

    def similarity(self, query_text: str, doc_id: str) -> float:
        """Cosine similarity between a query and a stored document."""
        qvec = self.query_vector(query_text)
        dvec = self._vectors.get(doc_id, {})
        return _cosine(qvec, dvec)

    def similar_to_doc(self, doc_id: str, k: int = 5) -> list[tuple[str, float]]:
        """Find k most similar docs to a given doc. Returns [(doc_id, score)]."""
        dvec = self._vectors.get(doc_id)
        if not dvec:
            return []
        scored = []
        for other_id, other_vec in self._vectors.items():
            if other_id == doc_id:
                continue
            sim = _cosine(dvec, other_vec)
            if sim > 0.0:
                scored.append((other_id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]