"""
Backend 1: ConversationBufferMemory (Short-Term)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Storage : RAM (in-process)
• Scope   : Session-scoped (lost on restart)
• Speed   : sub-millisecond
• Use case: Lịch sử hội thoại hiện tại
"""
import logging
from typing import List, Tuple

import tiktoken
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage

from .models import ContextItem, ContextPriority, MemoryType

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    Wraps LangChain ConversationBufferMemory.
    Auto-trims oldest turns when max_token_limit is exceeded.
    """

    def __init__(self, max_token_limit: int = 2000):
        self.max_token_limit = max_token_limit
        self.buffer = ConversationBufferMemory(
            return_messages=True,
            human_prefix="User",
            ai_prefix="Assistant",
            memory_key="chat_history",
        )
        self.encoder = tiktoken.get_encoding("cl100k_base")
        logger.info(f"ShortTermMemory init (max_tokens={max_token_limit})")

    # ──────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────
    def save_turn(self, query: str, response: str) -> None:
        """Save one (human, ai) pair and auto-trim if over limit."""
        self.buffer.chat_memory.add_user_message(query)
        self.buffer.chat_memory.add_ai_message(response)
        # Trim oldest pair(s) until under limit
        while self._total_tokens() > self.max_token_limit:
            self._trim_oldest()

    # ──────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────
    def get_history(self) -> List[Tuple[str, str]]:
        """Return history as [(human, ai)] pairs."""
        msgs: List[BaseMessage] = self.buffer.chat_memory.messages
        pairs = []
        for i in range(0, len(msgs) - 1, 2):
            if i + 1 < len(msgs):
                pairs.append((msgs[i].content, msgs[i + 1].content))
        return pairs

    def get_as_context_items(self, recent_n: int = 10) -> List[ContextItem]:
        """Return history as ContextItems for the window manager."""
        history = self.get_history()
        if not history:
            return []

        items: List[ContextItem] = []
        n = len(history)
        start = max(0, n - recent_n)

        for idx, (human, ai) in enumerate(history[start:], start=start):
            turns_ago = n - 1 - idx
            content = f"Turn {idx + 1}:\nUser: {human}\nAssistant: {ai}"
            token_count = len(self.encoder.encode(content))
            priority = (
                ContextPriority.MEDIUM if turns_ago < 3 else ContextPriority.LOW
            )
            items.append(
                ContextItem(
                    content=content,
                    priority=priority,
                    source=MemoryType.SHORT_TERM,
                    token_count=token_count,
                    metadata={"turn_index": idx, "turns_ago": turns_ago},
                )
            )
        return items

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────
    def _total_tokens(self) -> int:
        pairs = self.get_history()
        return sum(
            len(self.encoder.encode(h)) + len(self.encoder.encode(a))
            for h, a in pairs
        )

    def _trim_oldest(self) -> None:
        msgs = self.buffer.chat_memory.messages
        if len(msgs) >= 2:
            self.buffer.chat_memory.messages = msgs[2:]

    def clear(self) -> None:
        self.buffer.clear()
        logger.info("ShortTermMemory cleared")

    @property
    def turn_count(self) -> int:
        return len(self.buffer.chat_memory.messages) // 2
