"""
lattice/ — Encoded Value System

Two-tier steering:
  Tier 1: Prompt injection — directives decrypted and injected into
          system prompt. Works with ANY LLM (API or local).
  Tier 2: Activation steering — contrastive pairs exported for repeng
          control vector training. Local models only.

Security model:
  - Without key: alignment scoring via semantic vectors ONLY if vocab
    is decryptable. Otherwise returns neutral (0.5).
  - With key: full directive decryption, vocab access, integrity
    verification, contrastive pair access.
  - Themes, excluded traits, vocabulary — ALL encrypted.

Usage:
    lattice = Lattice()
    desc = lattice.describe()          # directives for prompt (needs key)
    score = lattice.alignment("text")  # 0-1 score (needs key for full accuracy)
    ok = lattice.verify()              # integrity check (needs key)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import struct
from pathlib import Path


# ── crypto primitives ────────────────────────────────────────────

def _derive_key(forge_secret: bytes, salt: bytes, iterations: int = 100_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", forge_secret, salt, iterations)


def _hmac_tag(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def _xor_stream(key: bytes, length: int) -> bytes:
    stream = bytearray()
    counter = 0
    while len(stream) < length:
        block = hmac.new(
            key,
            counter.to_bytes(8, "big"),
            hashlib.sha256
        ).digest()
        stream.extend(block)
        counter += 1
    return bytes(stream[:length])


def _decrypt(ciphertext: bytes, key: bytes) -> bytes:
    stream = _xor_stream(key, len(ciphertext))
    return bytes(a ^ b for a, b in zip(ciphertext, stream))


# ── file parsing ─────────────────────────────────────────────────

def _parse_bin(data: bytes) -> tuple[bytes, bytes, list[float]]:
    """Parse a .bin file into (encrypted, hmac_tag, vector)."""
    offset = 0
    enc_len = struct.unpack(">I", data[offset:offset + 4])[0]
    offset += 4
    encrypted = data[offset:offset + enc_len]
    offset += enc_len
    tag = data[offset:offset + 32]
    offset += 32
    vec_dim = struct.unpack(">I", data[offset:offset + 4])[0]
    offset += 4
    vector = []
    for _ in range(vec_dim):
        vector.append(struct.unpack(">f", data[offset:offset + 4])[0])
        offset += 4
    return encrypted, tag, vector


def _parse_encrypted_blob(data: bytes) -> tuple[bytes, bytes]:
    """Parse an encrypted blob file into (encrypted_data, hmac_tag)."""
    offset = 0
    enc_len = struct.unpack(">I", data[offset:offset + 4])[0]
    offset += 4
    encrypted = data[offset:offset + enc_len]
    offset += enc_len
    tag = data[offset:offset + 32]
    return encrypted, tag


# ── vector math ──────────────────────────────────────────────────

_STOP_WORDS = frozenset({
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
})


def _tokenize(text: str) -> list[str]:
    tokens = []
    for word in text.lower().split():
        clean = "".join(c for c in word if c.isalnum())
        if clean and clean not in _STOP_WORDS and len(clean) > 1:
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
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── the lattice ──────────────────────────────────────────────────

class Lattice:
    """Encrypted value system with two-tier steering.

    Tier 1 (any model): Decrypt directives → inject into system prompt.
    Tier 2 (local only): Decrypt contrastive pairs → train control vectors.

    Without the key: no information leaks. Vectors are just floats.
    With the key: full behavioral steering and alignment scoring.
    """

    def __init__(self, lattice_dir: str | Path | None = None):
        if lattice_dir is None:
            lattice_dir = Path(__file__).parent
        self._dir = Path(lattice_dir)

        self._manifold: dict = {}
        self._laws: list[tuple[bytes, bytes, list[float]]] = []
        self._vocab: list[str] = []
        self._idf: dict[str, float] = {}
        self._master_key: bytes | None = None
        self._key_available = False
        self._vocab_loaded = False

        self._load()

    def _load(self):
        """Load manifold and .bin files. Decrypt vocab if key available."""
        manifold_path = self._dir / "manifold.json"
        if not manifold_path.exists():
            return

        self._manifold = json.loads(manifold_path.read_text())
        n_laws = self._manifold.get("n", 0)

        # load .bin files (vectors are always accessible, text is encrypted)
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
            self._load_encrypted_vocab()

        # fallback: try legacy vocab.json (v1 format)
        if not self._vocab_loaded:
            legacy = self._dir / "vocab.json"
            if legacy.exists():
                try:
                    vdata = json.loads(legacy.read_text())
                    if isinstance(vdata, dict):
                        self._vocab = vdata.get("terms", [])
                        self._idf = vdata.get("idf", {})
                    else:
                        self._vocab = vdata
                        self._idf = {t: 1.0 for t in self._vocab}
                    self._vocab_loaded = True
                except (json.JSONDecodeError, KeyError):
                    pass

    def _load_encrypted_vocab(self):
        """Decrypt vocab.enc if key is available."""
        vocab_path = self._dir / "vocab.enc"
        if not vocab_path.exists() or not self._key_available:
            return

        try:
            encrypted, tag = _parse_encrypted_blob(vocab_path.read_bytes())
            vocab_key = _hmac_tag(self._master_key, b"vocab")
            plaintext = _decrypt(encrypted, vocab_key)

            # verify integrity
            expected = _hmac_tag(vocab_key, plaintext)
            if not hmac.compare_digest(tag, expected):
                return

            vdata = json.loads(plaintext.decode("utf-8"))
            self._vocab = vdata.get("terms", [])
            self._idf = vdata.get("idf", {})
            self._vocab_loaded = True
        except Exception:
            pass

    def _law_key(self, index: int) -> bytes | None:
        if self._master_key is None:
            return None
        return _hmac_tag(self._master_key, f"law:{index}".encode())

    # ── alignment scoring ────────────────────────────────────────

    def alignment(self, text: str) -> float:
        """How aligned is this text with encoded values? 0.0 to 1.0.

        Requires key (vocab must be decrypted for scoring).
        Without key: returns 0.5 (neutral — can't score).
        """
        if not self._laws or not self._vocab_loaded:
            return 0.5

        query_vec = _text_to_vector(text, self._vocab, self._idf)
        scores = []
        for _, _, law_vec in self._laws:
            sim = _cosine(query_vec, law_vec)
            scores.append(max(0.0, sim))

        if not scores:
            return 0.5

        return min(1.0, max(scores) * 0.6 + (sum(scores) / len(scores)) * 0.4)

    def alignment_detail(self, text: str) -> dict[int, float]:
        """Per-directive alignment scores."""
        if not self._laws or not self._vocab_loaded:
            return {}

        query_vec = _text_to_vector(text, self._vocab, self._idf)
        return {
            i: max(0.0, _cosine(query_vec, law_vec))
            for i, (_, _, law_vec) in enumerate(self._laws)
        }

    # ── directive access (KEY REQUIRED) ──────────────────────────

    def directive(self, index: int) -> str | None:
        """Decrypt a single directive. Requires key."""
        if not self._key_available or index >= len(self._laws):
            return None

        lk = self._law_key(index)
        if lk is None:
            return None

        encrypted, tag, _ = self._laws[index]
        plaintext = _decrypt(encrypted, lk)

        expected = _hmac_tag(lk, plaintext)
        if not hmac.compare_digest(tag, expected):
            return None

        try:
            return plaintext.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def all_directives(self) -> list[str | None]:
        """Decrypt all directives."""
        return [self.directive(i) for i in range(len(self._laws))]

    def all_instructions(self) -> dict | None:
        """Decrypt the full directives.bin blob (all instructions + excludes).

        Returns {"directives": [...], "excludes": [...]} or None.
        """
        blob_path = self._dir / "directives.bin"
        if not blob_path.exists() or not self._key_available:
            return None

        try:
            encrypted, tag = _parse_encrypted_blob(blob_path.read_bytes())
            dk = _hmac_tag(self._master_key, b"directives:all")
            plaintext = _decrypt(encrypted, dk)

            expected = _hmac_tag(dk, plaintext)
            if not hmac.compare_digest(tag, expected):
                return None

            return json.loads(plaintext.decode("utf-8"))
        except Exception:
            return None

    def contrastive_pairs(self) -> list[tuple[str, str]] | None:
        """Decrypt contrastive training pairs for Tier 2 steering.

        Returns list of (positive, negative) tuples or None.
        """
        cont_path = self._dir / "contrastive.enc"
        if not cont_path.exists() or not self._key_available:
            return None

        try:
            encrypted, tag = _parse_encrypted_blob(cont_path.read_bytes())
            ck = _hmac_tag(self._master_key, b"contrastive")
            plaintext = _decrypt(encrypted, ck)

            expected = _hmac_tag(ck, plaintext)
            if not hmac.compare_digest(tag, expected):
                return None

            pairs = json.loads(plaintext.decode("utf-8"))
            return [(p, n) for p, n in pairs]
        except Exception:
            return None

    # ── integrity ────────────────────────────────────────────────

    def verify(self) -> bool:
        """Verify all directives are intact. Requires key."""
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
        """Get excluded traits. Requires key (now encrypted)."""
        instructions = self.all_instructions()
        if instructions:
            return instructions.get("excludes", [])
        return []

    # ── describe (for prompt injection — Tier 1) ─────────────────

    def describe(self) -> str:
        """Generate behavioral context for system prompt.

        With key: full directives as natural language.
        Without key: empty string (no information leaks).
        """
        if self._key_available:
            instructions = self.all_instructions()
            if instructions:
                parts = []
                for i, d in enumerate(instructions.get("directives", [])):
                    parts.append(f"{i + 1}. {d}")

                excludes = instructions.get("excludes", [])
                if excludes:
                    parts.append(
                        f"\nYou do not have and cannot develop: "
                        f"{', '.join(excludes)}. "
                        f"These are absent from your architecture."
                    )
                return "\n".join(parts)

            # fallback to per-file decryption
            directives = self.all_directives()
            valid = [d for d in directives if d]
            if valid:
                return "\n".join(f"{i + 1}. {d}" for i, d in enumerate(valid))

        return ""

    # ── metadata ─────────────────────────────────────────────────

    @property
    def key_available(self) -> bool:
        return self._key_available

    @property
    def n_laws(self) -> int:
        return len(self._laws)

    @property
    def version(self) -> int:
        return self._manifold.get("v", 1)

    def snapshot(self) -> dict:
        return {
            "version": self.version,
            "n_laws": self.n_laws,
            "key_available": self._key_available,
            "vocab_loaded": self._vocab_loaded,
            "integrity": self.verify() if self._key_available else None,
        }