"""
Memory Agent — Full pipeline with 4 memory backends
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Flow per turn:
  1. Classify query intent (MemoryRouter)
  2. Pull memories from relevant backends
  3. Assemble context (ContextWindowManager, 4-level priority)
  4. Call gpt-4o-mini
  5. Save turn to short_term + episodic (always)
     + Redis / ChromaDB when relevant
"""
import logging
import uuid
from typing import Optional

import tiktoken
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.context.window_manager import ContextWindowManager
from src.memory.episodic_json import EpisodicJSONMemory
from src.memory.long_term_redis import LongTermRedisMemory
from src.memory.memory_router import MemoryRouter
from src.memory.models import (
    AgentResponse,
    ContextItem,
    ContextPriority,
    MemoryEntry,
    MemoryType,
    QueryIntent,
)
from src.memory.semantic_chroma import SemanticChromaMemory
from src.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant with excellent memory.
When you have relevant past information about the user, use it naturally in your responses:
- If you know the user's preferences, acknowledge and apply them.
- If you recall a past conversation, reference it naturally (e.g., "As we discussed earlier...").
- If answering a factual question, be accurate and concise.
- Be conversational and personalized based on what you know about the user.
Always respond in the same language the user uses."""


class MemoryAgent:
    """Conversational agent with 4-backend memory system."""

    def __init__(
        self,
        user_id:            str  = "user_001",
        session_id:         Optional[str] = None,
        max_context_tokens: int  = 8000,
        llm_model:          str  = "gpt-4o-mini",
        redis_host:         str  = "localhost",
        redis_port:         int  = 6379,
        chroma_host:        str  = "localhost",
        chroma_port:        int  = 8000,
    ):
        self.user_id    = user_id
        self.session_id = session_id or str(uuid.uuid4())

        # 4 backends
        self.short_term = ShortTermMemory(max_token_limit=2000)
        self.long_term  = LongTermRedisMemory(user_id=user_id, host=redis_host, port=redis_port)
        self.episodic   = EpisodicJSONMemory(user_id=user_id, session_id=self.session_id)
        self.semantic   = SemanticChromaMemory(user_id=user_id, chroma_host=chroma_host, chroma_port=chroma_port)

        # Router + window manager
        self.router         = MemoryRouter()
        self.window_manager = ContextWindowManager(max_tokens=max_context_tokens)

        # LLM
        self.llm     = ChatOpenAI(model=llm_model, temperature=0.7)
        self.encoder = tiktoken.get_encoding("cl100k_base")

        logger.info(
            f"MemoryAgent[{user_id}] ready | session={self.session_id[:8]}..."
        )

    # ──────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────
    def chat(self, query: str) -> AgentResponse:
        """Process one user query through the full memory pipeline."""
        turn_id = str(uuid.uuid4())

        # Step 1: classify intent
        intent = self.router.classify(query)
        logger.info(f"Intent={intent.value} | '{query[:60]}'")

        # Step 2: pull from backends
        memory_items = []

        # Short-term: always included
        memory_items.extend(self.short_term.get_as_context_items(recent_n=8))

        if intent == QueryIntent.USER_PREFERENCE:
            lt = self.long_term.get_as_context_item()
            if lt:
                memory_items.append(lt)

        elif intent == QueryIntent.EXPERIENCE_RECALL:
            memory_items.extend(self.episodic.get_as_context_items(n=5))

        elif intent == QueryIntent.FACTUAL_RECALL:
            memory_items.extend(self.semantic.search(query, k=3))

        # Step 3: assemble context
        context, included, total_tokens = self.window_manager.assemble(
            system_prompt=SYSTEM_PROMPT,
            current_query=query,
            memory_items=memory_items,
        )

        # Step 4: call LLM
        llm_resp    = self.llm.invoke([HumanMessage(content=context)])
        response_text = llm_resp.content
        out_tokens  = len(self.encoder.encode(response_text))

        # Step 5: persist
        entry = MemoryEntry(
            turn_id=turn_id,
            user_id=self.user_id,
            session_id=self.session_id,
            query=query,
            response=response_text,
            intent=intent.value,
            metadata={"input_tokens": total_tokens},
        )
        self.short_term.save_turn(query, response_text)
        self.episodic.save(entry)

        if intent in (QueryIntent.FACTUAL_RECALL, QueryIntent.EXPERIENCE_RECALL):
            self.semantic.save(entry)

        if intent == QueryIntent.USER_PREFERENCE:
            pref = f"[{query[:120]}]"
            self.long_term.save_preference(pref)

        return AgentResponse(
            response=response_text,
            memories_used=included,
            total_input_tokens=total_tokens,
            output_tokens=out_tokens,
            session_id=self.session_id,
            turn_id=turn_id,
            intent=intent,
        )

    # ──────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────
    def get_status(self) -> dict:
        return {
            "user_id":           self.user_id,
            "session_id":        self.session_id,
            "st_turns":          self.short_term.turn_count,
            "redis_connected":   self.long_term.is_connected,
            "redis_prefs":       len(self.long_term.get_preferences()),
            "episodic_entries":  self.episodic.total_entries,
            "chroma_connected":  self.semantic.is_connected,
            "chroma_vectors":    self.semantic.total_vectors,
        }

    def reset_session(self, clear_long_term: bool = False) -> None:
        """Start a new session (keep long-term data unless specified)."""
        self.short_term.clear()
        self.session_id          = str(uuid.uuid4())
        self.episodic.session_id = self.session_id
        if clear_long_term:
            self.long_term.clear()
            self.semantic.clear()
        logger.info(f"New session: {self.session_id[:8]}...")
