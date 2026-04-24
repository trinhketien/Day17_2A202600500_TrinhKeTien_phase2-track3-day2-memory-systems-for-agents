"""Shared data models for the memory system."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import uuid


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"   # ConversationBufferMemory (RAM)
    LONG_TERM   = "long_term"   # Redis (persistent)
    EPISODIC    = "episodic"    # JSON append-only log
    SEMANTIC    = "semantic"    # ChromaDB (vector)


class QueryIntent(str, Enum):
    USER_PREFERENCE   = "user_preference"    # → Redis
    EXPERIENCE_RECALL = "experience_recall"  # → Episodic JSON
    FACTUAL_RECALL    = "factual_recall"     # → ChromaDB semantic
    CURRENT_CONTEXT   = "current_context"   # → Short-term buffer only


class ContextPriority(int, Enum):
    """4-level hierarchy; lower value = higher priority = evicted last."""
    CRITICAL = 1  # System prompt + current turn — NEVER evict
    HIGH     = 2  # User preferences (Redis) — evict last
    MEDIUM   = 3  # Recent episodic / semantic (<3 turns back)
    LOW      = 4  # Older conversation history — evict first


@dataclass
class MemoryEntry:
    query:      str
    response:   str
    user_id:    str
    session_id: str
    turn_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:  str = field(default_factory=lambda: datetime.utcnow().isoformat())
    memory_type: MemoryType = MemoryType.EPISODIC
    intent:     str = QueryIntent.CURRENT_CONTEXT.value
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextItem:
    content:         str
    priority:        ContextPriority
    source:          MemoryType
    token_count:     int   = 0
    relevance_score: float = 1.0
    metadata:        Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    response:           str
    memories_used:      List[ContextItem]
    total_input_tokens: int
    output_tokens:      int
    session_id:         str
    turn_id:            str
    intent:             QueryIntent
    memory_hit:         bool = False
