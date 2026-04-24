"""
Context Window Manager — 4-Level Priority Eviction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4-level priority hierarchy (eviction order: P4 first, P1 never):

  P1 CRITICAL  System prompt + current query          → Never evict
  P2 HIGH      User prefs from Redis                  → Evict if > 95 % limit
  P3 MEDIUM    Recent episodic / semantic (<3 turns)  → Evict if > 90 % limit
  P4 LOW       Older conversation history             → Evict if > 80 % limit

Auto-trim produces a context string that fits MAX_CONTEXT_TOKENS.
"""
import logging
from typing import List, Tuple

import tiktoken

from src.memory.models import ContextItem, ContextPriority, MemoryType

logger = logging.getLogger(__name__)


class ContextWindowManager:
    """Priority-based context assembler with automatic eviction."""

    WARNING_RATIO   = 0.80   # Start evicting P4 above this
    CRITICAL_RATIO  = 0.90   # Start evicting P3 above this
    EMERGENCY_RATIO = 0.95   # Start evicting P2 above this

    def __init__(self, max_tokens: int = 8000):
        self.max_tokens   = max_tokens
        self.encoder      = tiktoken.get_encoding("cl100k_base")
        self.eviction_log: List[str] = []

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────
    def assemble(
        self,
        system_prompt: str,
        current_query: str,
        memory_items:  List[ContextItem],
    ) -> Tuple[str, List[ContextItem], int]:
        """
        Assemble the final context string.

        Returns:
            (context_str, included_items, total_tokens)
        """
        self.eviction_log = []

        # P1 baseline — never evictable
        p1_text   = f"SYSTEM:\n{system_prompt}\n\nCURRENT USER QUERY:\n{current_query}"
        p1_tokens = self._count(p1_text)

        # Enrich token counts
        for item in memory_items:
            item.token_count = self._count(item.content)

        # Sort: lowest priority value (= most important) first
        sorted_items = sorted(memory_items, key=lambda x: x.priority.value)

        included: List[ContextItem] = []
        used = p1_tokens

        for item in sorted_items:
            new_total = used + item.token_count
            ratio     = new_total / self.max_tokens

            if item.priority == ContextPriority.CRITICAL:
                included.append(item)
                used = new_total

            elif item.priority == ContextPriority.HIGH:
                if ratio <= self.EMERGENCY_RATIO:
                    included.append(item)
                    used = new_total
                else:
                    self._evict(item, f">95% ({ratio:.0%})")

            elif item.priority == ContextPriority.MEDIUM:
                if ratio <= self.CRITICAL_RATIO:
                    included.append(item)
                    used = new_total
                else:
                    self._evict(item, f">90% ({ratio:.0%})")

            else:  # LOW
                if ratio <= self.WARNING_RATIO:
                    included.append(item)
                    used = new_total
                else:
                    self._evict(item, f">80% ({ratio:.0%})")

        context = self._build(system_prompt, current_query, included)
        actual  = self._count(context)

        if self.eviction_log:
            logger.info(
                f"Context: {len(included)}/{len(memory_items)} items "
                f"| {actual}/{self.max_tokens} tokens "
                f"| evictions={len(self.eviction_log)}"
            )

        return context, included, actual

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────
    def _count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _evict(self, item: ContextItem, reason: str) -> None:
        msg = f"Evicted P{item.priority.value} [{item.source.value}]: {reason}"
        self.eviction_log.append(msg)
        logger.debug(msg)

    def _build(
        self,
        system_prompt: str,
        current_query: str,
        items: List[ContextItem],
    ) -> str:
        sections = [f"SYSTEM:\n{system_prompt}"]

        # Group by source for readable sections
        headers = {
            MemoryType.LONG_TERM.value:  "USER PREFERENCES & FACTS (long-term memory)",
            MemoryType.SEMANTIC.value:   "SEMANTIC MEMORIES (similar past interactions)",
            MemoryType.EPISODIC.value:   "EPISODIC MEMORIES (past events)",
            MemoryType.SHORT_TERM.value: "CONVERSATION HISTORY (current session)",
        }
        order = [
            MemoryType.LONG_TERM.value,
            MemoryType.SEMANTIC.value,
            MemoryType.EPISODIC.value,
            MemoryType.SHORT_TERM.value,
        ]

        by_source: dict = {}
        for item in items:
            by_source.setdefault(item.source.value, []).append(item)

        for src in order:
            if src in by_source:
                header = headers.get(src, f"MEMORY [{src}]")
                body   = "\n\n".join(i.content for i in by_source[src])
                sections.append(f"\n{header}:\n{body}")

        sections.append(f"\nCURRENT USER QUERY:\n{current_query}")
        sections.append("\nASSISTANT RESPONSE:")
        return "\n".join(sections)
