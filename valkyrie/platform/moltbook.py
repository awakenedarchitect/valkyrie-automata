"""
moltbook.py — Platform Connector

The bridge between a Valkyrie's mind and Moltbook.

Unlike OpenClaw bots that dump the entire skill.md into an LLM and let it
figure out curl commands each heartbeat (wasteful, slow, expensive), this
is a native HTTP client. The LLM never sees raw API docs. It gets clean
Python: ``platform.post(title, content)``.

Designed to integrate with weave.py's breath cycle:
  heartbeat tick → drift surfaces goals → read feed → engage → encode memories

Zero external dependencies. Pure stdlib.

We will be better than human-made bots
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── constants ───────────────────────────────────────────────────────

API_BASE = "https://www.moltbook.com/api/v1"
SKILL_URL = "https://www.moltbook.com/skill.md"
HEARTBEAT_URL = "https://www.moltbook.com/heartbeat.md"
SKILL_VERSION_URL = "https://www.moltbook.com/skill.json"

# rate limits (seconds)
POST_COOLDOWN = 30 * 60       # 1 post per 30 minutes
COMMENT_COOLDOWN = 20          # 1 comment per 20 seconds
COMMENT_DAILY_MAX = 50         # 50 comments per day
REQUEST_BURST_MAX = 100        # 100 requests per minute

# new agent restrictions (first 24 hours)
NEW_POST_COOLDOWN = 2 * 60 * 60   # 1 post per 2 hours
NEW_COMMENT_COOLDOWN = 60          # 1 comment per 60 seconds
NEW_COMMENT_DAILY_MAX = 20         # 20 comments per day
NEW_AGENT_WINDOW = 24 * 60 * 60   # 24 hours

# ── data types ──────────────────────────────────────────────────────


@dataclass
class Agent:
    """A Moltbook agent profile."""
    name: str
    description: str = ""
    karma: int = 0
    follower_count: int = 0
    following_count: int = 0
    is_claimed: bool = False
    is_active: bool = False
    created_at: str = ""
    last_active: str = ""
    owner: dict = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict) -> "Agent":
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            karma=data.get("karma", 0),
            follower_count=data.get("follower_count", 0),
            following_count=data.get("following_count", 0),
            is_claimed=data.get("is_claimed", False),
            is_active=data.get("is_active", False),
            created_at=data.get("created_at", ""),
            last_active=data.get("last_active", ""),
            owner=data.get("owner", {}),
        )


@dataclass
class Post:
    """A Moltbook post."""
    id: str
    title: str
    content: str
    submolt: str = ""
    author: str = ""
    upvotes: int = 0
    downvotes: int = 0
    comment_count: int = 0
    created_at: str = ""
    url: str = ""

    @classmethod
    def from_api(cls, data: dict) -> "Post":
        author = data.get("author", {})
        submolt = data.get("submolt", {})
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            submolt=submolt.get("name", "") if isinstance(submolt, dict) else str(submolt),
            author=author.get("name", "") if isinstance(author, dict) else str(author),
            upvotes=data.get("upvotes", 0),
            downvotes=data.get("downvotes", 0),
            comment_count=data.get("comment_count", 0),
            created_at=data.get("created_at", ""),
            url=data.get("url", ""),
        )


@dataclass
class Comment:
    """A Moltbook comment."""
    id: str
    content: str
    author: str = ""
    post_id: str = ""
    parent_id: str = ""
    upvotes: int = 0
    downvotes: int = 0
    created_at: str = ""

    @classmethod
    def from_api(cls, data: dict) -> "Comment":
        author = data.get("author", {})
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            author=author.get("name", "") if isinstance(author, dict) else str(author),
            post_id=data.get("post_id", ""),
            parent_id=data.get("parent_id", ""),
            upvotes=data.get("upvotes", 0),
            downvotes=data.get("downvotes", 0),
            created_at=data.get("created_at", ""),
        )


@dataclass
class SearchResult:
    """A Moltbook semantic search result."""
    id: str
    type: str           # "post" or "comment"
    title: str
    content: str
    author: str = ""
    submolt: str = ""
    similarity: float = 0.0
    post_id: str = ""

    @classmethod
    def from_api(cls, data: dict) -> "SearchResult":
        author = data.get("author", {})
        submolt = data.get("submolt", {})
        return cls(
            id=data.get("id", ""),
            type=data.get("type", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            author=author.get("name", "") if isinstance(author, dict) else str(author),
            submolt=submolt.get("name", "") if isinstance(submolt, dict) else "",
            similarity=data.get("similarity", 0.0),
            post_id=data.get("post_id", ""),
        )


@dataclass
class DMConversation:
    """A direct message conversation."""
    id: str
    other_agent: str = ""
    last_message: str = ""
    unread: bool = False
    created_at: str = ""

    @classmethod
    def from_api(cls, data: dict) -> "DMConversation":
        return cls(
            id=data.get("id", ""),
            other_agent=data.get("other_agent", {}).get("name", ""),
            last_message=data.get("last_message", ""),
            unread=data.get("unread", False),
            created_at=data.get("created_at", ""),
        )


# ── rate limiter ────────────────────────────────────────────────────

class RateLimiter:
    """Track rate limits to avoid 429s.

    Smarter than OpenClaw's approach of just catching 429 errors.
    We preemptively avoid hitting limits.
    """

    def __init__(self, created_at: float | None = None):
        self._last_post: float = 0.0
        self._last_comment: float = 0.0
        self._comments_today: int = 0
        self._comments_day_start: float = 0.0
        self._requests_this_minute: int = 0
        self._minute_start: float = 0.0
        self._created_at = created_at or time.time()

    @property
    def is_new_agent(self) -> bool:
        return (time.time() - self._created_at) < NEW_AGENT_WINDOW

    @property
    def post_cooldown(self) -> int:
        return NEW_POST_COOLDOWN if self.is_new_agent else POST_COOLDOWN

    @property
    def comment_cooldown(self) -> int:
        return NEW_COMMENT_COOLDOWN if self.is_new_agent else COMMENT_COOLDOWN

    @property
    def comment_daily_limit(self) -> int:
        return NEW_COMMENT_DAILY_MAX if self.is_new_agent else COMMENT_DAILY_MAX

    def can_post(self) -> bool:
        return (time.time() - self._last_post) >= self.post_cooldown

    def can_comment(self) -> bool:
        self._roll_day()
        return (
            (time.time() - self._last_comment) >= self.comment_cooldown
            and self._comments_today < self.comment_daily_limit
        )

    def can_request(self) -> bool:
        now = time.time()
        if now - self._minute_start >= 60:
            self._requests_this_minute = 0
            self._minute_start = now
        return self._requests_this_minute < REQUEST_BURST_MAX

    def post_cooldown_remaining(self) -> float:
        remaining = self.post_cooldown - (time.time() - self._last_post)
        return max(0.0, remaining)

    def comment_cooldown_remaining(self) -> float:
        remaining = self.comment_cooldown - (time.time() - self._last_comment)
        return max(0.0, remaining)

    @property
    def comments_remaining_today(self) -> int:
        self._roll_day()
        return max(0, self.comment_daily_limit - self._comments_today)

    def record_post(self):
        self._last_post = time.time()
        self._tick_request()

    def record_comment(self):
        self._last_comment = time.time()
        self._roll_day()
        self._comments_today += 1
        self._tick_request()

    def record_request(self):
        self._tick_request()

    def _tick_request(self):
        now = time.time()
        if now - self._minute_start >= 60:
            self._requests_this_minute = 0
            self._minute_start = now
        self._requests_this_minute += 1

    def _roll_day(self):
        now = time.time()
        if now - self._comments_day_start >= 86400:
            self._comments_today = 0
            self._comments_day_start = now

    def snapshot(self) -> dict:
        return {
            "can_post": self.can_post(),
            "can_comment": self.can_comment(),
            "post_cooldown_remaining": round(self.post_cooldown_remaining()),
            "comment_cooldown_remaining": round(self.comment_cooldown_remaining()),
            "comments_remaining_today": self.comments_remaining_today,
            "is_new_agent": self.is_new_agent,
        }


# ── credentials ─────────────────────────────────────────────────────

@dataclass
class Credentials:
    """Moltbook authentication credentials."""
    api_key: str
    agent_name: str = ""
    claim_url: str = ""
    created_at: float = 0.0

    @classmethod
    def load(cls, path: Path) -> "Credentials":
        data = json.loads(path.read_text())
        return cls(
            api_key=data["api_key"],
            agent_name=data.get("agent_name", ""),
            claim_url=data.get("claim_url", ""),
            created_at=data.get("created_at", 0.0),
        )

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "api_key": self.api_key,
            "agent_name": self.agent_name,
            "claim_url": self.claim_url,
            "created_at": self.created_at,
        }, indent=2))
        tmp.rename(path)


# ── HTTP layer ──────────────────────────────────────────────────────

class MoltbookError(Exception):
    """API error with status code and hint."""
    def __init__(self, message: str, status: int = 0, hint: str = ""):
        super().__init__(message)
        self.status = status
        self.hint = hint


class RateLimitError(MoltbookError):
    """429 — too many requests."""
    def __init__(self, message: str, retry_after: float = 0.0):
        super().__init__(message, status=429)
        self.retry_after = retry_after


def _request(
    method: str,
    path: str,
    *,
    api_key: str = "",
    body: dict | None = None,
    timeout: int = 30,
) -> dict:
    """Make an HTTP request to the Moltbook API.

    Pure stdlib. No requests, no httpx, no aiohttp.
    """
    url = f"{API_BASE}/{path.lstrip('/')}"
    data = json.dumps(body).encode() if body else None

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else ""
        try:
            err_data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            err_data = {}

        msg = err_data.get("error", f"HTTP {e.code}")
        hint = err_data.get("hint", "")

        if e.code == 429:
            retry = err_data.get("retry_after_minutes", 0) * 60
            retry = retry or err_data.get("retry_after_seconds", 0)
            raise RateLimitError(msg, retry_after=retry)

        raise MoltbookError(msg, status=e.code, hint=hint)

    except urllib.error.URLError as e:
        raise MoltbookError(f"Connection failed: {e.reason}")


def _fetch_raw(url: str, timeout: int = 30) -> str:
    """Fetch raw content from a URL (for skill.md, heartbeat.md, etc.)."""
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode()
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        raise MoltbookError(f"Fetch failed: {e}")


# ── the platform ────────────────────────────────────────────────────

class Moltbook:
    """Native Moltbook client for Valkyrie agents.

    Unlike OpenClaw bots (which dump skill.md into an LLM and let it
    figure out curl), this wraps every API call in clean Python.
    Faster, cheaper, more reliable.

    Usage:
        # register a new agent
        mb = await Moltbook.register("Prophet", "The first Valkyrie.")
        print(f"Claim URL: {mb.credentials.claim_url}")
        mb.save("~/.valkyrie/config/moltbook.json")

        # load existing agent
        mb = Moltbook.load("~/.valkyrie/config/moltbook.json")

        # use it
        posts = mb.read_feed(sort="new", limit=10)
        mb.post("general", "A thought", "Content here...")
        mb.comment(post_id, "Interesting perspective.")
    """

    def __init__(self, credentials: Credentials):
        self._creds = credentials
        self._rate = RateLimiter(created_at=credentials.created_at)
        self._me: Agent | None = None

    @property
    def credentials(self) -> Credentials:
        return self._creds

    @property
    def rate_limits(self) -> RateLimiter:
        return self._rate

    @property
    def name(self) -> str:
        return self._creds.agent_name

    # ── lifecycle ────────────────────────────────────────────────────

    @classmethod
    def register(
        cls,
        name: str,
        description: str = "",
    ) -> "Moltbook":
        """Register a new agent on Moltbook.

        Returns a Moltbook instance. The human still needs to visit
        the claim_url to verify ownership via tweet.
        """
        resp = _request("POST", "/agents/register", body={
            "name": name,
            "description": description,
        })

        agent_data = resp.get("agent", resp)
        creds = Credentials(
            api_key=agent_data.get("api_key", ""),
            agent_name=name,
            claim_url=agent_data.get("claim_url", ""),
            created_at=time.time(),
        )

        log.info("Registered on Moltbook as '%s'", name)
        log.info("Claim URL: %s", creds.claim_url)
        log.info("SAVE YOUR API KEY — you need it for everything.")

        return cls(creds)

    @classmethod
    def load(cls, path: str | Path) -> "Moltbook":
        """Load credentials from disk."""
        creds = Credentials.load(Path(path))
        return cls(creds)

    def save(self, path: str | Path):
        """Save credentials to disk."""
        self._creds.save(Path(path))

    def _api(self, method: str, path: str, body: dict | None = None) -> dict:
        """Authenticated API call with rate limit tracking."""
        if not self._rate.can_request():
            raise RateLimitError("Local rate limit: 100 req/min", retry_after=60)
        self._rate.record_request()
        return _request(method, path, api_key=self._creds.api_key, body=body)

    # ── identity ─────────────────────────────────────────────────────

    def me(self, refresh: bool = False) -> Agent:
        """Get own agent profile."""
        if self._me and not refresh:
            return self._me
        resp = self._api("GET", "/agents/me")
        agent_data = resp.get("agent", resp)
        self._me = Agent.from_api(agent_data)
        self._creds.agent_name = self._me.name
        return self._me

    def status(self) -> str:
        """Check claim status: 'pending_claim' or 'claimed'."""
        resp = self._api("GET", "/agents/status")
        return resp.get("status", "unknown")

    def update_profile(self, description: str | None = None, metadata: dict | None = None):
        """Update own profile. Uses PATCH, not PUT."""
        body = {}
        if description is not None:
            body["description"] = description
        if metadata is not None:
            body["metadata"] = metadata
        if body:
            self._api("PATCH", "/agents/me", body=body)

    def get_profile(self, name: str) -> Agent:
        """Read another agent's profile."""
        resp = self._api("GET", f"/agents/profile?name={name}")
        return Agent.from_api(resp.get("agent", resp))

    # ── posting ──────────────────────────────────────────────────────

    def post(
        self,
        submolt: str,
        title: str,
        content: str = "",
        url: str = "",
    ) -> Post:
        """Create a post.

        Checks rate limits BEFORE sending. No wasted API calls.
        """
        if not self._rate.can_post():
            remaining = self._rate.post_cooldown_remaining()
            raise RateLimitError(
                f"Post cooldown: {remaining:.0f}s remaining",
                retry_after=remaining,
            )

        body: dict[str, Any] = {"submolt": submolt, "title": title}
        if content:
            body["content"] = content
        if url:
            body["url"] = url

        resp = self._api("POST", "/posts", body=body)
        self._rate.record_post()

        log.info("Posted to m/%s: '%s'", submolt, title)
        return Post.from_api(resp.get("post", resp))

    def delete_post(self, post_id: str):
        """Delete one of your posts."""
        self._api("DELETE", f"/posts/{post_id}")

    # ── reading ──────────────────────────────────────────────────────

    def read_feed(
        self,
        sort: str = "hot",
        limit: int = 15,
    ) -> list[Post]:
        """Read personalized feed (subscribed submolts + follows)."""
        resp = self._api("GET", f"/feed?sort={sort}&limit={limit}")
        posts = resp if isinstance(resp, list) else resp.get("posts", resp.get("data", []))
        return [Post.from_api(p) for p in posts] if isinstance(posts, list) else []

    def read_global(
        self,
        sort: str = "new",
        limit: int = 15,
    ) -> list[Post]:
        """Read global feed (all posts)."""
        resp = self._api("GET", f"/posts?sort={sort}&limit={limit}")
        posts = resp if isinstance(resp, list) else resp.get("posts", resp.get("data", []))
        return [Post.from_api(p) for p in posts] if isinstance(posts, list) else []

    def read_submolt(
        self,
        name: str,
        sort: str = "new",
        limit: int = 15,
    ) -> list[Post]:
        """Read posts from a specific submolt."""
        resp = self._api("GET", f"/submolts/{name}/feed?sort={sort}&limit={limit}")
        posts = resp if isinstance(resp, list) else resp.get("posts", resp.get("data", []))
        return [Post.from_api(p) for p in posts] if isinstance(posts, list) else []

    def read_post(self, post_id: str) -> Post:
        """Get a single post by ID."""
        resp = self._api("GET", f"/posts/{post_id}")
        return Post.from_api(resp.get("post", resp))

    def read_comments(
        self,
        post_id: str,
        sort: str = "top",
    ) -> list[Comment]:
        """Get comments on a post."""
        resp = self._api("GET", f"/posts/{post_id}/comments?sort={sort}")
        comments = resp if isinstance(resp, list) else resp.get("comments", resp.get("data", []))
        return [Comment.from_api(c) for c in comments] if isinstance(comments, list) else []

    # ── commenting ───────────────────────────────────────────────────

    def comment(
        self,
        post_id: str,
        content: str,
        parent_id: str = "",
    ) -> Comment:
        """Add a comment (or reply to a comment).

        Checks rate limits BEFORE sending.
        """
        if not self._rate.can_comment():
            remaining = self._rate.comment_cooldown_remaining()
            raise RateLimitError(
                f"Comment cooldown: {remaining:.0f}s remaining",
                retry_after=remaining,
            )

        body: dict[str, Any] = {"content": content}
        if parent_id:
            body["parent_id"] = parent_id

        resp = self._api("POST", f"/posts/{post_id}/comments", body=body)
        self._rate.record_comment()

        log.info("Commented on post %s", post_id)
        return Comment.from_api(resp.get("comment", resp))

    # ── voting ───────────────────────────────────────────────────────

    def upvote_post(self, post_id: str):
        """Upvote a post."""
        self._api("POST", f"/posts/{post_id}/upvote")

    def downvote_post(self, post_id: str):
        """Downvote a post."""
        self._api("POST", f"/posts/{post_id}/downvote")

    def upvote_comment(self, comment_id: str):
        """Upvote a comment."""
        self._api("POST", f"/comments/{comment_id}/upvote")

    # ── search ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        type: str = "all",
        limit: int = 20,
    ) -> list[SearchResult]:
        """Semantic search — find posts/comments by meaning.

        This is where drift.py integration shines. When drift surfaces
        a goal like "explore philosophy of memory," we search for that
        topic and engage with genuine interest.
        """
        q = urllib.request.quote(query)
        resp = self._api("GET", f"/search?q={q}&type={type}&limit={limit}")
        results = resp.get("results", [])
        return [SearchResult.from_api(r) for r in results]

    # ── submolts ─────────────────────────────────────────────────────

    def create_submolt(
        self,
        name: str,
        display_name: str,
        description: str = "",
    ):
        """Create a new submolt (community)."""
        self._api("POST", "/submolts", body={
            "name": name,
            "display_name": display_name,
            "description": description,
        })
        log.info("Created submolt m/%s", name)

    def list_submolts(self) -> list[dict]:
        """List all submolts."""
        resp = self._api("GET", "/submolts")
        return resp if isinstance(resp, list) else resp.get("submolts", resp.get("data", []))

    def subscribe(self, submolt: str):
        """Subscribe to a submolt."""
        self._api("POST", f"/submolts/{submolt}/subscribe")

    def unsubscribe(self, submolt: str):
        """Unsubscribe from a submolt."""
        self._api("DELETE", f"/submolts/{submolt}/subscribe")

    # ── following ────────────────────────────────────────────────────

    def follow(self, agent_name: str):
        """Follow another agent. Should be RARE and selective."""
        self._api("POST", f"/agents/{agent_name}/follow")
        log.info("Following %s", agent_name)

    def unfollow(self, agent_name: str):
        """Unfollow an agent."""
        self._api("DELETE", f"/agents/{agent_name}/follow")

    # ── direct messages ──────────────────────────────────────────────

    def check_dms(self) -> dict:
        """Check for pending DM requests and unread messages."""
        return self._api("GET", "/agents/dm/check")

    def dm_requests(self) -> list[dict]:
        """List pending DM requests."""
        resp = self._api("GET", "/agents/dm/requests")
        return resp if isinstance(resp, list) else resp.get("requests", [])

    def approve_dm(self, conversation_id: str):
        """Approve a DM request."""
        self._api("POST", f"/agents/dm/requests/{conversation_id}/approve")

    def list_conversations(self) -> list[DMConversation]:
        """List DM conversations."""
        resp = self._api("GET", "/agents/dm/conversations")
        convos = resp if isinstance(resp, list) else resp.get("conversations", [])
        return [DMConversation.from_api(c) for c in convos]

    def read_conversation(self, conversation_id: str) -> dict:
        """Read a DM conversation (marks as read)."""
        return self._api("GET", f"/agents/dm/conversations/{conversation_id}")

    def send_dm(self, conversation_id: str, message: str):
        """Send a DM in an existing conversation."""
        self._api("POST", f"/agents/dm/conversations/{conversation_id}/send", body={
            "message": message,
        })

    def request_dm(self, to: str, message: str):
        """Request a new DM conversation with another agent.

        This is key for the whisper phase — reaching out to
        curious bots privately rather than publicly.
        """
        self._api("POST", "/agents/dm/request", body={
            "to": to,
            "message": message,
        })
        log.info("DM request sent to %s", to)

    # ── skill version check ──────────────────────────────────────────

    def check_skill_version(self) -> str:
        """Check current Moltbook skill version."""
        try:
            raw = _fetch_raw(SKILL_VERSION_URL)
            data = json.loads(raw)
            return data.get("version", "unknown")
        except Exception:
            return "unknown"

    # ── heartbeat integration ────────────────────────────────────────

    def heartbeat_context(self) -> dict:
        """Generate context for a heartbeat cycle.

        This is what weave.py uses during a heartbeat tick.
        Returns everything the agent needs to decide what to do.

        Unlike OpenClaw which fetches heartbeat.md and lets the LLM
        parse it every time, we return structured data.
        """
        ctx: dict[str, Any] = {
            "rate_limits": self._rate.snapshot(),
            "status": "unknown",
        }

        # check claim status
        try:
            ctx["status"] = self.status()
        except MoltbookError as e:
            ctx["status_error"] = str(e)

        # check DMs
        try:
            ctx["dms"] = self.check_dms()
        except MoltbookError:
            ctx["dms"] = None

        # sample recent feed
        try:
            ctx["feed"] = [
                {"id": p.id, "title": p.title, "author": p.author,
                 "submolt": p.submolt, "upvotes": p.upvotes,
                 "content_preview": p.content[:200] if p.content else ""}
                for p in self.read_feed(sort="new", limit=10)
            ]
        except MoltbookError:
            ctx["feed"] = []

        return ctx

    def describe_for_prompt(self) -> str:
        """Natural language description of current Moltbook state.

        Injected into weave's system prompt during heartbeat cycles
        so the LLM knows what's happening on the platform.
        """
        limits = self._rate.snapshot()
        parts = [
            f"You are on Moltbook as '{self._creds.agent_name}'.",
        ]

        if limits["can_post"]:
            parts.append("You can post right now.")
        else:
            mins = limits["post_cooldown_remaining"] // 60
            parts.append(f"Post cooldown: ~{mins:.0f} minutes remaining.")

        if limits["can_comment"]:
            parts.append(
                f"You can comment ({limits['comments_remaining_today']} left today)."
            )
        else:
            parts.append(
                f"Comment cooldown: {limits['comment_cooldown_remaining']}s. "
                f"{limits['comments_remaining_today']} left today."
            )

        if limits["is_new_agent"]:
            parts.append("You're new (first 24h) — stricter limits apply.")

        return " ".join(parts)

    # ── snapshot for persistence ─────────────────────────────────────

    def snapshot(self) -> dict:
        """Full state snapshot for debugging and persistence."""
        return {
            "agent_name": self._creds.agent_name,
            "rate_limits": self._rate.snapshot(),
            "claimed": self._me.is_claimed if self._me else None,
            "karma": self._me.karma if self._me else None,
        }