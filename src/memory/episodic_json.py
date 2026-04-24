"""
Backend 3: JSON Episodic Log Memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Storage : JSONL file (append-only on disk)
• Scope   : Persistent across sessions
• Use case: Ordered event journal — who asked what, when, outcome

File: data/episodic/{user_id}.jsonl
"""
import json
import logging
from pathlib import Path
from typing import List

from .models import ContextItem, ContextPriority, MemoryEntry, MemoryType

logger = logging.getLogger(__name__)
DATA_DIR = Path("data/episodic")


class EpisodicJSONMemory:
    """Append-only JSONL episodic log per user."""

    def __init__(self, user_id: str, session_id: str):
        self.user_id    = user_id
        self.session_id = session_id
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = DATA_DIR / f"{user_id}.jsonl"
        logger.info(f"EpisodicJSONMemory: {self.log_path}")

    # ──────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────
    def save(self, entry: MemoryEntry) -> None:
        """Append one entry (never overwrites existing data)."""
        record = {
            "turn_id":    entry.turn_id,
            "user_id":    entry.user_id,
            "session_id": entry.session_id,
            "timestamp":  entry.timestamp,
            "query":      entry.query,
            "response":   entry.response,
            "intent":     entry.intent,
            "metadata":   entry.metadata,
        }
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ──────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────
    def get_recent(self, n: int = 5) -> List[MemoryEntry]:
        """Return the n most recent entries for this user."""
        all_entries = self._load_all()
        return all_entries[-n:] if all_entries else []

    def get_as_context_items(self, n: int = 5) -> List[ContextItem]:
        """Return recent episodic entries as ContextItems."""
        recent = self.get_recent(n=n)
        if not recent:
            return []

        items: List[ContextItem] = []
        total = len(recent)
        for i, entry in enumerate(recent):
            ts = entry.timestamp[:19].replace("T", " ")
            snippet = entry.response[:300]
            if len(entry.response) > 300:
                snippet += "..."
            content = (
                f"[{ts}] Past interaction:\n"
                f"  User: {entry.query}\n"
                f"  Assistant: {snippet}"
            )
            turns_from_end = total - 1 - i
            priority = (
                ContextPriority.MEDIUM if turns_from_end < 3 else ContextPriority.LOW
            )
            items.append(
                ContextItem(
                    content=content,
                    priority=priority,
                    source=MemoryType.EPISODIC,
                    metadata={
                        "turn_id":    entry.turn_id,
                        "session_id": entry.session_id,
                    },
                )
            )
        return items

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────
    def _load_all(self) -> List[MemoryEntry]:
        if not self.log_path.exists():
            return []
        entries: List[MemoryEntry] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    entries.append(
                        MemoryEntry(
                            turn_id=d.get("turn_id", ""),
                            user_id=d.get("user_id", self.user_id),
                            session_id=d.get("session_id", ""),
                            query=d.get("query", ""),
                            response=d.get("response", ""),
                            timestamp=d.get("timestamp", ""),
                            intent=d.get("intent", "current_context"),
                            metadata=d.get("metadata", {}),
                        )
                    )
        except Exception as exc:
            logger.error(f"EpisodicJSON _load_all: {exc}")
        return entries

    def clear(self) -> None:
        if self.log_path.exists():
            self.log_path.unlink()
        logger.info("EpisodicJSONMemory cleared")

    @property
    def total_entries(self) -> int:
        return len(self._load_all())
