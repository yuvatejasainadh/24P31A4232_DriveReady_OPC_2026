"""Task 5: Controlled Conversational Memory Manager for Day 2.

Manages short-term, session-isolated conversation history with a configurable
sliding window (`window_size` messages). Supports stateless mode (`window_size=0`),
session clearing, and debug inspection.

NOTE: This stores volatile in-memory session context only—distinct from persistent
database storage such as SQLite, PostgreSQL, or Firestore.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from business_config import ConversationTurn, ValidationError, validate_non_empty_text


SENSITIVE_KEYS = {"password", "secret", "token", "api_key", "credit_card", "cvv"}


@dataclass(frozen=True)
class MemoryEntry:
    """Single message stored in a session's bounded memory window."""

    role: str  # "user" or "assistant"
    text: str
    intent: str | None = None
    entities: dict[str, Any] | None = None
    timestamp_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sanitize_entities(entities: dict[str, Any] | None) -> dict[str, Any] | None:
    if not entities:
        return None
    return {
        k: v
        for k, v in entities.items()
        if str(k).lower() not in SENSITIVE_KEYS and v is not None
    }


class ConversationMemoryManager:
    """Session-isolated, bounded sliding-window conversation memory manager."""

    storage_type = "volatile_in_memory_session_store"

    def __init__(self, window_size: int = 6) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 0:
            raise ValidationError("'window_size' must be a non-negative integer.")
        self.window_size = window_size
        self._sessions: dict[str, list[MemoryEntry]] = {}

    @property
    def is_stateless(self) -> bool:
        return self.window_size == 0

    def add_message(
        self,
        session_id: str,
        role: str,
        text: str,
        intent: str | None = None,
        entities: dict[str, Any] | None = None,
    ) -> None:
        """Record a message inside `session_id` while enforcing `window_size`."""
        sid = validate_non_empty_text(session_id, "session_id")
        clean_text = validate_non_empty_text(text, "text")
        if role not in {"user", "assistant"}:
            raise ValidationError("'role' must be 'user' or 'assistant'.")

        if self.window_size == 0:
            return

        history = self._sessions.setdefault(sid, [])
        history.append(
            MemoryEntry(
                role=role,
                text=clean_text,
                intent=intent,
                entities=_sanitize_entities(entities),
                timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        )
        if len(history) > self.window_size:
            self._sessions[sid] = history[-self.window_size :]

    def get_history(self, session_id: str) -> list[MemoryEntry]:
        """Return a shallow copy of recent messages for `session_id` only."""
        sid = validate_non_empty_text(session_id, "session_id")
        if self.window_size == 0:
            return []
        return list(self._sessions.get(sid, []))

    def get_recent_texts(self, session_id: str) -> list[str]:
        """Return formatted `role: text` lines for prompt or extraction context."""
        return [f"{entry.role}: {entry.text}" for entry in self.get_history(session_id)]

    def get_day1_turns(self, session_id: str) -> list[ConversationTurn]:
        """Convert stored session history into Day 1 `ConversationTurn` objects."""
        return [
            ConversationTurn(
                role="model" if entry.role == "assistant" else "user",
                text=entry.text,
            )
            for entry in self.get_history(session_id)
        ]

    def clear_session(self, session_id: str) -> bool:
        """Erase all stored messages for `session_id`."""
        sid = validate_non_empty_text(session_id, "session_id")
        return self._sessions.pop(sid, None) is not None

    def list_sessions(self) -> list[str]:
        """Return active session IDs currently holding memory entries."""
        return sorted(self._sessions.keys())

    def inspect_session(self, session_id: str) -> dict[str, Any]:
        """Return a safe debug snapshot of a single session's memory state."""
        sid = validate_non_empty_text(session_id, "session_id")
        entries = self.get_history(sid)
        return {
            "session_id": sid,
            "storage_type": self.storage_type,
            "mode": "stateless" if self.is_stateless else "session_window",
            "window_size": self.window_size,
            "message_count": len(entries),
            "messages": [entry.to_dict() for entry in entries],
        }
