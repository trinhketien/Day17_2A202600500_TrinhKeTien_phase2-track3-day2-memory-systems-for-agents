"""
Memory Router — Phân loại query intent để chọn backend phù hợp
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Intent            → Primary backend
─────────────────────────────────────────
USER_PREFERENCE   → Redis (long-term)         "I like/hate/prefer..."
EXPERIENCE_RECALL → Episodic JSON             "Last time / you said..."
FACTUAL_RECALL    → ChromaDB semantic         "What is X? How does Y...?"
CURRENT_CONTEXT   → ConversationBuffer only  Everything else

Two-level classification:
  Level 1: Fast regex (zero API cost)
  Level 2: Keyword scoring fallback
"""
import re
import logging
from typing import Dict, Any, List

from .models import QueryIntent

logger = logging.getLogger(__name__)


class MemoryRouter:
    """Rule-based query intent classifier."""

    # ── Preference signals ──────────────────────────────────────────
    _PREF_PATTERNS = [
        r"tôi thích", r"tôi yêu", r"tôi ghét", r"tôi không thích",
        r"sở thích (của tôi)?", r"ưa thích", r"tôi muốn", r"tôi cần",
        r"(tôi|mình) thấy .*(hay|thú vị|chán|tệ)",
        r"i (like|love|hate|prefer|enjoy|dislike|want|need)",
        r"my (favorite|favourite|preference|preferred)",
        r"i('m| am) (a fan of|into|not a fan)",
    ]

    # ── Experience / recall signals ─────────────────────────────────
    _EXP_PATTERNS = [
        r"(hôm qua|hôm trước|tuần trước|tháng trước)",
        r"(lần trước|lần trước đó|lần cuối)",
        r"(trước đây|trước đó|khi nãy|lúc nãy)",
        r"(bạn đã nói|bạn đã trả lời|tôi đã hỏi)",
        r"(bạn có nhớ|bạn nhớ|nhớ không|đã từng)",
        r"(lượt trước|turn trước)",
        r"(yesterday|last (time|week|month|session))",
        r"(you (said|mentioned|told me|replied))",
        r"(i (asked|said|mentioned|told you))",
        r"(remember when|do you remember|recall|earlier (you|i))",
        r"(previously|based on (what|our|the) (i said|conversation))",
        r"going back to (what|the)",
    ]

    # ── Factual / question signals ──────────────────────────────────
    _FACT_PATTERNS = [
        r"(là gì|gì là|gì vậy|nghĩa là)\??",
        r"(như thế nào|thế nào|ra sao)\??",
        r"(tại sao|vì sao|lý do gì)\??",
        r"(khi nào|bao giờ|lúc nào)\??",
        r"(ở đâu|chỗ nào|nơi nào)\??",
        r"(ai là|ai vậy|là ai)\??",
        r"(giải thích|cho biết|hãy nói|mô tả|định nghĩa)",
        r"(cách|phương pháp|bước|quy trình)",
        r"what (is|are|was|were|does|do)\b",
        r"how (do|does|to|can|should)\b",
        r"why (is|are|do|does|would)\b",
        r"when (is|are|was|did)\b",
        r"where (is|are|do)\b",
        r"who (is|are|was)\b",
        r"(explain|describe|define|tell me about|what('s| is) the)",
        r"(steps|process|method|way to|how to)",
    ]

    def classify(self, query: str) -> QueryIntent:
        """
        Classify into one of 4 intents.
        Priority order: EXPERIENCE > PREFERENCE > FACTUAL > CURRENT_CONTEXT
        """
        q = query.lower().strip()

        for pat in self._EXP_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                logger.debug(f"EXPERIENCE_RECALL  ← '{query[:60]}'")
                return QueryIntent.EXPERIENCE_RECALL

        for pat in self._PREF_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                logger.debug(f"USER_PREFERENCE    ← '{query[:60]}'")
                return QueryIntent.USER_PREFERENCE

        for pat in self._FACT_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                logger.debug(f"FACTUAL_RECALL     ← '{query[:60]}'")
                return QueryIntent.FACTUAL_RECALL

        logger.debug(f"CURRENT_CONTEXT   ← '{query[:60]}'")
        return QueryIntent.CURRENT_CONTEXT

    def classify_with_explanation(self, query: str) -> Dict[str, Any]:
        intent = self.classify(query)
        return {
            "query":    query,
            "intent":   intent.value,
            "backends": self._backends_for(intent),
        }

    # ── Internal ────────────────────────────────────────────────────
    @staticmethod
    def _backends_for(intent: QueryIntent) -> List[str]:
        base = ["short_term"]
        mapping = {
            QueryIntent.USER_PREFERENCE:   base + ["long_term_redis"],
            QueryIntent.EXPERIENCE_RECALL: base + ["episodic_json"],
            QueryIntent.FACTUAL_RECALL:    base + ["semantic_chroma"],
            QueryIntent.CURRENT_CONTEXT:   base,
        }
        return mapping.get(intent, base)
