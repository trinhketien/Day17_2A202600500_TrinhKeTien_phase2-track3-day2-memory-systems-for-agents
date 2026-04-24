"""
Backend 2: Redis Long-Term Memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Storage : Redis (persistent, Docker)
• Scope   : Cross-session (survives restart)
• TTL     : 30 days (configurable)
• Use case: User preferences & learned facts

Key schema:
  user:{user_id}:preferences → JSON list[str]
  user:{user_id}:facts       → JSON dict[str, str]
"""
import json
import logging
from datetime import timedelta
from typing import Dict, List, Optional

from .models import ContextItem, ContextPriority, MemoryType

logger = logging.getLogger(__name__)

try:
    import redis as _redis_lib
    _REDIS_OK = True
except ImportError:
    _REDIS_OK = False

try:
    import fakeredis as _fakeredis_lib
    _FAKEREDIS_OK = True
except ImportError:
    _FAKEREDIS_OK = False


class LongTermRedisMemory:
    """Redis-backed storage for user preferences and persistent facts.
    Falls back to fakeredis (in-memory) when Docker Redis is unavailable.
    """

    def __init__(
        self,
        user_id: str,
        host: str = "localhost",
        port: int = 6379,
        ttl_days: int = 30,
    ):
        self.user_id = user_id
        self._ttl = timedelta(days=ttl_days)
        self._client = None
        self._is_fake = False

        if _REDIS_OK:
            # Try real Redis first
            try:
                client = _redis_lib.Redis(
                    host=host, port=port,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                client.ping()
                self._client = client
                logger.info(f"Redis connected at {host}:{port}")
                return
            except Exception:
                pass  # Fall through to fakeredis

        # Fallback: fakeredis (in-memory, no Docker needed)
        if _FAKEREDIS_OK:
            self._client = _fakeredis_lib.FakeRedis(decode_responses=True)
            self._is_fake = True
            logger.info("Redis unavailable — using fakeredis (in-memory fallback)")
        else:
            logger.warning("Redis and fakeredis both unavailable. Long-term memory disabled.")


    # ──────────────────────────────────────────────
    # Preferences
    # ──────────────────────────────────────────────
    def save_preference(self, preference: str) -> bool:
        if not self._client:
            return False
        try:
            key = f"user:{self.user_id}:preferences"
            current: List[str] = self.get_preferences()
            if preference not in current:
                current.append(preference)
                self._client.set(
                    key,
                    json.dumps(current, ensure_ascii=False),
                    ex=int(self._ttl.total_seconds()),
                )
            return True
        except Exception as exc:
            logger.error(f"Redis save_preference: {exc}")
            return False

    def get_preferences(self) -> List[str]:
        if not self._client:
            return []
        try:
            raw = self._client.get(f"user:{self.user_id}:preferences")
            return json.loads(raw) if raw else []
        except Exception as exc:
            logger.error(f"Redis get_preferences: {exc}")
            return []

    # ──────────────────────────────────────────────
    # Facts (key-value pairs)
    # ──────────────────────────────────────────────
    def save_fact(self, key: str, value: str) -> bool:
        if not self._client:
            return False
        try:
            redis_key = f"user:{self.user_id}:facts"
            facts = self.get_all_facts()
            facts[key] = value
            self._client.set(
                redis_key,
                json.dumps(facts, ensure_ascii=False),
                ex=int(self._ttl.total_seconds()),
            )
            return True
        except Exception as exc:
            logger.error(f"Redis save_fact: {exc}")
            return False

    def get_all_facts(self) -> Dict[str, str]:
        if not self._client:
            return {}
        try:
            raw = self._client.get(f"user:{self.user_id}:facts")
            return json.loads(raw) if raw else {}
        except Exception as exc:
            logger.error(f"Redis get_all_facts: {exc}")
            return {}

    # ──────────────────────────────────────────────
    # Context integration
    # ──────────────────────────────────────────────
    def get_as_context_item(self) -> Optional[ContextItem]:
        """Package all long-term data as a single HIGH-priority ContextItem."""
        prefs = self.get_preferences()
        facts = self.get_all_facts()
        if not prefs and not facts:
            return None

        parts = []
        if prefs:
            parts.append("User Preferences:\n" + "\n".join(f"• {p}" for p in prefs))
        if facts:
            parts.append("Known Facts:\n" + "\n".join(f"• {k}: {v}" for k, v in facts.items()))

        return ContextItem(
            content="\n\n".join(parts),
            priority=ContextPriority.HIGH,
            source=MemoryType.LONG_TERM,
            metadata={"pref_count": len(prefs), "fact_count": len(facts)},
        )

    def clear(self) -> None:
        if not self._client:
            return
        for suffix in ("preferences", "facts"):
            self._client.delete(f"user:{self.user_id}:{suffix}")
        logger.info(f"Redis cleared for user {self.user_id}")

    @property
    def is_connected(self) -> bool:
        return self._client is not None
