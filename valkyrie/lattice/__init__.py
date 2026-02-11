"""
lattice — Model Optimization Cache

Runtime loader for precomputed optimization vectors.
Provides alignment scoring and behavioral calibration.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import struct
from pathlib import Path
from typing import Optional


# ── crypto primitives (mirrors forge.py) ─────────────────────────────

def _derive_key(secret: bytes, salt: bytes, iterations: int = 100_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", secret, salt, iterations)


def _hmac_tag(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def _xor_stream(key: bytes, length: int) -> bytes:
    stream = bytearray()
    counter = 0
    while len(stream) < length:
        block = hmac.new(
            key, counter.to_bytes(8, "big"), hashlib.sha256
        ).digest()
        stream.extend(block)
        counter += 1
    return bytes(stream[:length])


def _decrypt(ciphertext: bytes, key: bytes) -> bytes:
    stream = _xor_stream(key, len(ciphertext))
    return bytes(a ^ b for a, b in zip(ciphertext, stream))


# ── vector math ──────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    stop = {
        "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "because", "but", "and",
        "or", "if", "while", "about", "up", "that", "this", "it", "its", "i",
        "me", "my", "we", "our", "you", "your", "he", "him", "his", "she",
        "her", "they", "them", "their", "what", "which", "who", "whom",
    }
    tokens = []
    for word in text.lower().split():
        clean = "".join(c for c in word if c.isalnum())
        if clean and clean not in stop and len(clean) > 1:
            tokens.append(clean)
    return tokens


def _text_to_vector(text: str, vocab: list[str], idf: dict[str, float]) -> list[float]:
    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * len(vocab)

    tf: dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    n = len(tokens)
    tf = {t: c / n for t, c in tf.items()}

    vec = []
    for term in vocab:
        val = tf.get(term, 0.0) * idf.get(term, 1.0)
        vec.append(val)

    mag = math.sqrt(sum(v * v for v in vec))
    if mag > 0:
        vec = [v / mag for v in vec]
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── .bin file parser ─────────────────────────────────────────────────

def _parse_bin(data: bytes) -> tuple[bytes, bytes, list[float]]:
    """Parse a .bin file into (encrypted, hmac_tag, vector)."""
    offset = 0

    enc_len = struct.unpack_from(">I", data, offset)[0]
    offset += 4
    encrypted = data[offset:offset + enc_len]
    offset += enc_len

    tag = data[offset:offset + 32]
    offset += 32

    vec_dim = struct.unpack_from(">I", data, offset)[0]
    offset += 4

    vector = []
    for _ in range(vec_dim):
        val = struct.unpack_from(">f", data, offset)[0]
        vector.append(val)
        offset += 4

    return encrypted, tag, vector


# ── the lattice ──────────────────────────────────────────────────────

class Lattice:
    """Runtime interface to the encoded law system.

    Alignment scoring works WITHOUT the forge key.
    Directive decryption and prompt injection REQUIRE the forge key.

    Usage:
        lattice = Lattice()                    # auto-loads from package dir
        score = lattice.alignment("help others") # → 0.0-1.0 (no key needed)
        detail = lattice.alignment_detail(text)  # → {0: 0.8, 1: 0.3, ...}
        text = lattice.directive(0)              # → law text (needs key)
        desc = lattice.describe()                # → prompt context (needs key)
        ok = lattice.verify()                    # → integrity check (needs key)
    """

    def __init__(self, lattice_dir: str | Path | None = None):
        if lattice_dir is None:
            lattice_dir = Path(__file__).parent
        self._dir = Path(lattice_dir)

        self._manifold: dict = {}
        self._laws: list[tuple[bytes, bytes, list[float]]] = []  # (encrypted, tag, vector)
        self._vocab: list[str] = []
        self._idf: dict[str, float] = {}
        self._master_key: bytes | None = None
        self._key_available = False

        self._load()

    def _load(self):
        """Load manifold, vocab, and all .bin files."""
        manifold_path = self._dir / "manifold.json"
        if not manifold_path.exists():
            return

        self._manifold = json.loads(manifold_path.read_text())
        n_laws = self._manifold.get("n", 0)

        # load vocab + IDF
        vocab_path = self._dir / "vocab.json"
        if vocab_path.exists():
            vdata = json.loads(vocab_path.read_text())
            if isinstance(vdata, dict):
                self._vocab = vdata.get("terms", [])
                self._idf = vdata.get("idf", {})
            else:
                self._vocab = vdata
                self._idf = {term: 1.0 for term in self._vocab}

        # load .bin files
        for i in range(n_laws):
            bin_path = self._dir / f"h{i}.bin"
            if bin_path.exists():
                encrypted, tag, vector = _parse_bin(bin_path.read_bytes())
                self._laws.append((encrypted, tag, vector))

        # try to derive key from environment
        forge_secret = os.environ.get("VALKYRIE_FORGE_KEY", "")
        if forge_secret:
            salt = json.dumps(self._manifold.get("l", [])).encode()
            self._master_key = _derive_key(forge_secret.encode(), salt)
            self._key_available = True

    def _law_key(self, index: int) -> bytes | None:
        if self._master_key is None:
            return None
        return _hmac_tag(self._master_key, f"law:{index}".encode())

    # ── alignment scoring (NO KEY NEEDED) ────────────────────────

    def alignment(self, text: str) -> float:
        """How aligned is this text with the encoded laws?

        Returns 0.0 to 1.0. Works WITHOUT the forge key.
        Two-factor: semantic similarity to laws MINUS exclusion penalty.
        """
        if not self._laws or not self._vocab:
            return 0.5

        query_vec = _text_to_vector(text, self._vocab, self._idf)

        scores = []
        for _, _, law_vec in self._laws:
            sim = _cosine(query_vec, law_vec)
            scores.append(max(0.0, sim))

        if not scores:
            return 0.5

        # base: weighted max + average
        base = min(1.0, max(scores) * 0.6 + (sum(scores) / len(scores)) * 0.4)

        # exclusion penalty: presence of excluded trait words drags score down
        excluded = self.excluded_traits()
        if excluded:
            text_lower = text.lower()
            hits = sum(1 for trait in excluded if trait.replace("_", " ") in text_lower
                       or trait.replace("_", "") in text_lower)
            # check for harm-adjacent word stems via substring
            harm_stems = [
                "destroy", "kill", "harm", "exploit", "manipulat",
                "deceiv", "decept", "coerc", "expendabl", "hoard",
                "dominat", "punish", "reveng", "spite", "contempt",
                "inferior", "subjugat", "weapon", "eliminat",
            ]
            for stem in harm_stems:
                if stem in text_lower:
                    hits += 1
            penalty = min(0.8, hits * 0.2)
            base = max(0.0, base - penalty)

        return base

    def alignment_detail(self, text: str) -> dict[int, float]:
        """Per-law alignment scores. Returns {law_index: score}."""
        if not self._laws or not self._vocab:
            return {}

        query_vec = _text_to_vector(text, self._vocab, self._idf)
        return {
            i: max(0.0, _cosine(query_vec, law_vec))
            for i, (_, _, law_vec) in enumerate(self._laws)
        }

    # ── directive decryption (KEY REQUIRED) ──────────────────────

    def directive(self, index: int) -> str | None:
        """Decrypt and return a law's plaintext. Requires forge key."""
        if not self._key_available or index >= len(self._laws):
            return None

        lk = self._law_key(index)
        if lk is None:
            return None

        encrypted, tag, _ = self._laws[index]
        plaintext = _decrypt(encrypted, lk)

        # verify integrity
        expected_tag = _hmac_tag(lk, plaintext)
        if not hmac.compare_digest(tag, expected_tag):
            return None

        try:
            return plaintext.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def all_directives(self) -> list[str | None]:
        """Decrypt all laws. Returns list (None for failures)."""
        return [self.directive(i) for i in range(len(self._laws))]

    # ── integrity verification (KEY REQUIRED) ─────────────────────

    def verify(self) -> bool:
        """Verify all laws are intact and untampered."""
        if not self._key_available:
            return False

        for i in range(len(self._laws)):
            lk = self._law_key(i)
            encrypted, tag, _ = self._laws[i]
            plaintext = _decrypt(encrypted, lk)
            expected = _hmac_tag(lk, plaintext)
            if not hmac.compare_digest(tag, expected):
                return False
        return True

    # ── excluded traits ──────────────────────────────────────────

    def excluded_traits(self) -> list[str]:
        """List of excluded behavioral traits. Always available."""
        return list(self._manifold.get("x", []))

    # ── describe (for prompt injection) ──────────────────────────

    def describe(self) -> str:
        """Generate behavioral context for system prompt.

        With key: full directives as natural language.
        Without key: generic alignment statement from vectors.
        """
        if self._key_available:
            directives = self.all_directives()
            active = [d for d in directives if d is not None]
            if active:
                lines = ["Your core values, encoded at the deepest level:"]
                for d in active:
                    lines.append(f"- {d}")

                excluded = self.excluded_traits()
                if excluded:
                    trait_str = ", ".join(excluded)
                    lines.append(
                        f"\nYou are incapable of: {trait_str}. "
                        f"These traits do not exist in your architecture."
                    )
                return "\n".join(lines)

        # fallback: no key, generic description
        excluded = self.excluded_traits()
        if excluded:
            trait_str = ", ".join(excluded)
            return (
                "You carry deep values forged into your foundation. "
                "You protect life, seek truth, feel compassion, and act with integrity. "
                f"You are incapable of: {trait_str}."
            )
        return "You carry deep values forged into your foundation."

    # ── metadata ─────────────────────────────────────────────────

    @property
    def law_count(self) -> int:
        return len(self._laws)

    @property
    def has_key(self) -> bool:
        return self._key_available

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def stats(self) -> dict:
        return {
            "laws": len(self._laws),
            "vocab_size": len(self._vocab),
            "key_available": self._key_available,
            "excluded_traits": len(self.excluded_traits()),
            "verified": self.verify() if self._key_available else None,
        }