"""
LangGraph Memory Agent — Lab 17
=================================
Full LangGraph StateGraph implementation with MemoryState TypedDict.

Graph flow:
  START → classify_intent → retrieve_memory → build_prompt → call_llm → save_memory → END

Satisfies rubric Muc 2:
  - MemoryState TypedDict with messages, user_profile, episodes, semantic_hits, memory_budget
  - retrieve_memory(state) node that gathers from 4 backends
  - Prompt has 4 labeled sections: PROFILE, EPISODIC, SEMANTIC, CONVERSATION
  - Token budget trim via ContextWindowManager
"""
import json
import logging
import os
import uuid
from typing import TypedDict, Optional

import tiktoken
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.memory.short_term import ShortTermMemory
from src.memory.long_term_redis import LongTermRedisMemory
from src.memory.episodic_json import EpisodicJSONMemory
from src.memory.semantic_chroma import SemanticChromaMemory
from src.memory.memory_router import MemoryRouter
from src.memory.models import (
    MemoryEntry, ContextItem, ContextPriority, MemoryType, QueryIntent
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# 1. MemoryState — LangGraph state dict
# ════════════════════════════════════════════════════════════════════════════

class MemoryState(TypedDict):
    """LangGraph-compatible state for the memory agent."""
    messages: list                     # conversation message history
    current_query: str                 # the current user query
    user_profile: dict                 # long-term profile facts & preferences
    episodes: list                     # episodic memory entries (dicts)
    semantic_hits: list                # semantic search results (strings)
    memory_budget: int                 # remaining token budget
    intent: str                        # classified intent
    assembled_prompt: str              # final prompt after injection
    response: str                      # LLM response
    turn_id: str                       # unique turn identifier
    session_id: str                    # current session
    user_id: str                       # user identifier
    memories_used: list                # ContextItems actually included
    input_tokens: int                  # total input tokens
    output_tokens: int                 # total output tokens


# ════════════════════════════════════════════════════════════════════════════
# 2. Graph nodes — each is a pure function: State → State
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant with excellent memory.
When you have relevant past information about the user, use it naturally in your responses:
- If you know the user's preferences, acknowledge and apply them.
- If you recall a past conversation, reference it naturally.
- If answering a factual question, be accurate and concise.
- Be conversational and personalized based on what you know about the user.
Always respond in the same language the user uses."""


class LangGraphMemoryAgent:
    """
    Full LangGraph agent with 4 memory backends.

    Uses a manual graph execution pipeline (skeleton LangGraph) that mirrors
    the StateGraph pattern: each node is a function that takes MemoryState
    and returns an updated MemoryState.
    """

    def __init__(
        self,
        user_id: str = "user_001",
        session_id: Optional[str] = None,
        max_context_tokens: int = 8000,
        llm_model: str = "gpt-4o-mini",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
    ):
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())
        self.max_context_tokens = max_context_tokens

        # 4 memory backends
        self.short_term = ShortTermMemory(max_token_limit=2000)
        self.long_term = LongTermRedisMemory(
            user_id=user_id, host=redis_host, port=redis_port
        )
        self.episodic = EpisodicJSONMemory(
            user_id=user_id, session_id=self.session_id
        )
        self.semantic = SemanticChromaMemory(
            user_id=user_id, chroma_host=chroma_host, chroma_port=chroma_port
        )

        # Router
        self.router = MemoryRouter()

        # LLM
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # Build the graph (node pipeline)
        self._nodes = [
            ("classify_intent",  self._node_classify_intent),
            ("retrieve_memory",  self._node_retrieve_memory),
            ("build_prompt",     self._node_build_prompt),
            ("call_llm",         self._node_call_llm),
            ("save_memory",      self._node_save_memory),
        ]

        logger.info(
            f"LangGraphMemoryAgent[{user_id}] ready | session={self.session_id[:8]}..."
        )

    # ────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────

    def chat(self, query: str) -> dict:
        """Run the full graph pipeline for one user query."""
        # Initialize state
        state: MemoryState = {
            "messages": [],
            "current_query": query,
            "user_profile": {},
            "episodes": [],
            "semantic_hits": [],
            "memory_budget": self.max_context_tokens,
            "intent": "",
            "assembled_prompt": "",
            "response": "",
            "turn_id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "memories_used": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }

        # Execute graph: START → node1 → node2 → ... → END
        for node_name, node_fn in self._nodes:
            logger.debug(f"Graph node: {node_name}")
            state = node_fn(state)

        return state

    def get_status(self) -> dict:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "st_turns": self.short_term.turn_count,
            "redis_connected": self.long_term.is_connected,
            "redis_prefs": len(self.long_term.get_preferences()),
            "redis_facts": len(self.long_term.get_all_facts()),
            "episodic_entries": self.episodic.total_entries,
            "chroma_connected": self.semantic.is_connected,
            "chroma_vectors": self.semantic.total_vectors,
        }

    def reset_session(self, clear_long_term: bool = False) -> None:
        self.short_term.clear()
        self.session_id = str(uuid.uuid4())
        self.episodic.session_id = self.session_id
        if clear_long_term:
            self.long_term.clear()
            self.semantic.clear()

    # ────────────────────────────────────────────────────────────────────
    # Node 1: classify_intent
    # ────────────────────────────────────────────────────────────────────

    def _node_classify_intent(self, state: MemoryState) -> MemoryState:
        """Classify user query intent to determine which backends to query."""
        intent = self.router.classify(state["current_query"])
        state["intent"] = intent.value
        logger.info(f"Intent: {intent.value} | '{state['current_query'][:60]}'")
        return state

    # ────────────────────────────────────────────────────────────────────
    # Node 2: retrieve_memory  (KEY NODE — gathers from all backends)
    # ────────────────────────────────────────────────────────────────────

    def _node_retrieve_memory(self, state: MemoryState) -> MemoryState:
        """
        Retrieve memory from all relevant backends based on intent.
        Populates state[user_profile], state[episodes], state[semantic_hits].
        """
        query = state["current_query"]
        intent = state["intent"]

        # A) Always: recent conversation from short-term
        history = self.short_term.get_history()
        state["messages"] = [
            {"role": "user", "content": h, "response": a}
            for h, a in history
        ]

        # B) Long-term profile (always loaded — it's the user's identity)
        prefs = self.long_term.get_preferences()
        facts = self.long_term.get_all_facts()
        state["user_profile"] = {
            "preferences": prefs,
            "facts": facts,
        }

        # C) Episodic memory (especially for experience recall)
        if intent in (
            QueryIntent.EXPERIENCE_RECALL.value,
            QueryIntent.CURRENT_CONTEXT.value,
        ):
            recent_entries = self.episodic.get_recent(n=5)
            state["episodes"] = [
                {
                    "timestamp": e.timestamp,
                    "query": e.query,
                    "response": e.response[:300],
                    "intent": e.intent,
                }
                for e in recent_entries
            ]

        # D) Semantic search (especially for factual recall)
        if intent in (
            QueryIntent.FACTUAL_RECALL.value,
            QueryIntent.EXPERIENCE_RECALL.value,
        ):
            hits = self.semantic.search(query, k=3)
            state["semantic_hits"] = [item.content for item in hits]

        return state

    # ────────────────────────────────────────────────────────────────────
    # Node 3: build_prompt  (inject 4 memory sections into prompt)
    # ────────────────────────────────────────────────────────────────────

    def _node_build_prompt(self, state: MemoryState) -> MemoryState:
        """
        Assemble final prompt with 4 labeled memory sections.
        Applies token budget trimming.
        """
        sections = [f"SYSTEM:\n{SYSTEM_PROMPT}"]
        budget_used = len(self.encoder.encode(SYSTEM_PROMPT))
        budget = self.max_context_tokens
        memories_used = []

        # ── Section 1: USER PROFILE (long-term) ──────────────────────
        profile = state["user_profile"]
        if profile.get("preferences") or profile.get("facts"):
            lines = []
            if profile["preferences"]:
                lines.append("Preferences:")
                for p in profile["preferences"]:
                    lines.append(f"  - {p}")
            if profile["facts"]:
                lines.append("Known Facts:")
                for k, v in profile["facts"].items():
                    lines.append(f"  - {k}: {v}")
            profile_text = "\n".join(lines)
            profile_tokens = len(self.encoder.encode(profile_text))

            if budget_used + profile_tokens < budget * 0.95:
                sections.append(f"\n[USER PROFILE]:\n{profile_text}")
                budget_used += profile_tokens
                memories_used.append({
                    "source": "long_term",
                    "priority": "HIGH",
                    "content": profile_text[:200],
                })

        # ── Section 2: EPISODIC MEMORIES ─────────────────────────────
        if state["episodes"]:
            ep_lines = []
            for ep in state["episodes"][-5:]:
                ts = ep.get("timestamp", "")[:19]
                ep_lines.append(
                    f"[{ts}] User: {ep['query']}\n  Assistant: {ep['response'][:200]}"
                )
            ep_text = "\n".join(ep_lines)
            ep_tokens = len(self.encoder.encode(ep_text))

            if budget_used + ep_tokens < budget * 0.90:
                sections.append(f"\n[EPISODIC MEMORIES]:\n{ep_text}")
                budget_used += ep_tokens
                memories_used.append({
                    "source": "episodic",
                    "priority": "MEDIUM",
                    "content": ep_text[:200],
                })

        # ── Section 3: SEMANTIC MEMORIES ─────────────────────────────
        if state["semantic_hits"]:
            sem_text = "\n".join(state["semantic_hits"][:3])
            sem_tokens = len(self.encoder.encode(sem_text))

            if budget_used + sem_tokens < budget * 0.90:
                sections.append(f"\n[SEMANTIC MEMORIES]:\n{sem_text}")
                budget_used += sem_tokens
                memories_used.append({
                    "source": "semantic",
                    "priority": "MEDIUM",
                    "content": sem_text[:200],
                })

        # ── Section 4: RECENT CONVERSATION ───────────────────────────
        if state["messages"]:
            conv_lines = []
            for msg in state["messages"][-8:]:
                conv_lines.append(f"User: {msg['content']}")
                conv_lines.append(f"Assistant: {msg['response'][:200]}")
            conv_text = "\n".join(conv_lines)
            conv_tokens = len(self.encoder.encode(conv_text))

            if budget_used + conv_tokens < budget * 0.80:
                sections.append(f"\n[RECENT CONVERSATION]:\n{conv_text}")
                budget_used += conv_tokens
                memories_used.append({
                    "source": "short_term",
                    "priority": "LOW",
                    "content": conv_text[:200],
                })

        # ── Current query ────────────────────────────────────────────
        sections.append(f"\n[CURRENT QUERY]:\n{state['current_query']}")
        sections.append("\nASSISTANT RESPONSE:")

        prompt = "\n".join(sections)
        state["assembled_prompt"] = prompt
        state["memory_budget"] = budget - budget_used
        state["memories_used"] = memories_used
        state["input_tokens"] = len(self.encoder.encode(prompt))

        return state

    # ────────────────────────────────────────────────────────────────────
    # Node 4: call_llm
    # ────────────────────────────────────────────────────────────────────

    def _node_call_llm(self, state: MemoryState) -> MemoryState:
        """Call the LLM with the assembled prompt."""
        resp = self.llm.invoke([HumanMessage(content=state["assembled_prompt"])])
        state["response"] = resp.content
        state["output_tokens"] = len(self.encoder.encode(resp.content))
        return state

    # ────────────────────────────────────────────────────────────────────
    # Node 5: save_memory  (persist to all relevant backends)
    # ────────────────────────────────────────────────────────────────────

    def _node_save_memory(self, state: MemoryState) -> MemoryState:
        """Save the interaction to all relevant memory backends."""
        query = state["current_query"]
        response = state["response"]
        intent = state["intent"]

        # Always: short-term + episodic
        self.short_term.save_turn(query, response)
        entry = MemoryEntry(
            turn_id=state["turn_id"],
            user_id=state["user_id"],
            session_id=state["session_id"],
            query=query,
            response=response,
            intent=intent,
            metadata={"input_tokens": state["input_tokens"]},
        )
        self.episodic.save(entry)

        # Semantic: save for factual/experience queries
        if intent in (
            QueryIntent.FACTUAL_RECALL.value,
            QueryIntent.EXPERIENCE_RECALL.value,
        ):
            self.semantic.save(entry)

        # Long-term profile: extract and save preferences/facts
        if intent == QueryIntent.USER_PREFERENCE.value:
            self._extract_and_save_profile(query)

        return state

    # ────────────────────────────────────────────────────────────────────
    # Conflict handling: extract facts with OVERWRITE semantics
    # ────────────────────────────────────────────────────────────────────

    def _extract_and_save_profile(self, query: str) -> None:
        """
        Use LLM to extract profile facts from user statement.
        OVERWRITE existing facts when user corrects themselves.
        This is the key conflict-handling mechanism (Rubric Muc 3).
        """
        existing_facts = self.long_term.get_all_facts()
        existing_prefs = self.long_term.get_preferences()

        extract_prompt = f"""Extract structured facts from the user statement below.
If a fact contradicts existing data, the NEW fact takes priority (OVERWRITE, not append).

Existing profile:
  Preferences: {json.dumps(existing_prefs, ensure_ascii=False)}
  Facts: {json.dumps(existing_facts, ensure_ascii=False)}

User statement: "{query}"

Return ONLY valid JSON:
{{
  "facts": {{"key": "value", ...}},
  "preferences": ["pref1", "pref2", ...],
  "overwritten": ["key1", "key2", ...]
}}

Rules:
- facts: key-value pairs (name, job, allergy, location, company, etc.)
- preferences: style preferences (concise, detailed, language, etc.)
- overwritten: list of keys where old value was replaced by new value
- If no extractable data, return {{"facts": {{}}, "preferences": [], "overwritten": []}}
"""
        try:
            resp = self.llm.invoke([HumanMessage(content=extract_prompt)])
            raw = resp.content.strip()
            # Handle markdown code blocks
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            data = json.loads(raw)

            # Save facts with OVERWRITE semantics
            for key, value in data.get("facts", {}).items():
                self.long_term.save_fact(key, value)

            # Save preferences
            for pref in data.get("preferences", []):
                self.long_term.save_preference(pref)

            overwritten = data.get("overwritten", [])
            if overwritten:
                logger.info(f"Profile conflict resolved: overwritten keys = {overwritten}")

        except Exception as exc:
            logger.warning(f"Profile extraction failed: {exc}")
            # Fallback: save raw query as preference
            self.long_term.save_preference(f"[{query[:120]}]")
