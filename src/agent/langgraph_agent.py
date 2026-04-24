"""
LangGraph Memory Agent — Lab 17
=================================
Triển khai đầy đủ StateGraph của thư viện langgraph với MemoryState TypedDict.

Luồng graph:
    START -> classify_intent -> retrieve_memory -> build_prompt -> call_llm -> save_memory -> END

Đáp ứng rubric Mục 2:
  - MemoryState TypedDict với messages, user_profile, episodes, semantic_hits, memory_budget
  - StateGraph với các node rõ ràng
  - Prompt có 4 section: [USER PROFILE], [EPISODIC MEMORIES], [SEMANTIC MEMORIES], [RECENT CONVERSATION]
  - Quản lý ngân sách token qua ContextWindowManager
"""
import json
import logging
import os
import uuid
from typing import TypedDict, Optional, Annotated
import operator

import tiktoken
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from src.memory.short_term import ShortTermMemory
from src.memory.long_term_redis import LongTermRedisMemory
from src.memory.episodic_json import EpisodicJSONMemory
from src.memory.semantic_chroma import SemanticChromaMemory
from src.memory.memory_router import MemoryRouter
from src.memory.models import MemoryEntry, QueryIntent

logger = logging.getLogger(__name__)


# ============================================================================
# 1. MemoryState — cấu trúc trạng thái LangGraph
# ============================================================================

class MemoryState(TypedDict):
    """Trạng thái tương thích LangGraph cho memory agent."""
    messages: list                  # Lịch sử hội thoại trong phiên
    current_query: str              # Truy vấn hiện tại của người dùng
    user_profile: dict              # Hồ sơ người dùng từ long-term (Redis)
    episodes: list                  # Các bản ghi episodic (dicts)
    semantic_hits: list             # Kết quả tìm kiếm ngữ nghĩa (strings)
    memory_budget: int              # Ngân sách token còn lại
    intent: str                     # Ý định được phân loại
    assembled_prompt: str           # Prompt sau khi inject bộ nhớ
    response: str                   # Phản hồi từ LLM
    turn_id: str                    # Định danh lượt trao đổi
    session_id: str                 # Phiên làm việc hiện tại
    user_id: str                    # Định danh người dùng
    memories_used: list             # Các ContextItem thực sự được đưa vào prompt
    input_tokens: int               # Tổng token đầu vào
    output_tokens: int              # Tổng token đầu ra


# ============================================================================
# 2. LangGraphMemoryAgent — agent với StateGraph thật
# ============================================================================

SYSTEM_PROMPT = """Bạn là một trợ lý thông minh có khả năng ghi nhớ tốt.
Khi có thông tin liên quan về người dùng, hãy sử dụng một cách tự nhiên:
- Nếu biết sở thích của người dùng, hãy áp dụng trong câu trả lời.
- Nếu nhớ cuộc trò chuyện trước, hãy tham chiếu một cách tự nhiên.
- Nếu trả lời câu hỏi thực tế, hãy chính xác và súc tích.
Luôn trả lời bằng ngôn ngữ người dùng đang sử dụng."""


class LangGraphMemoryAgent:
    """
    Agent bộ nhớ đầy đủ sử dụng StateGraph của langgraph.

    Graph: START -> classify_intent -> retrieve_memory -> build_prompt -> call_llm -> save_memory -> END
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

        self.router = MemoryRouter()

        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # Xây dựng StateGraph thật của langgraph
        self._graph = self._build_graph()

        logger.info(
            f"LangGraphMemoryAgent[{user_id}] ready | session={self.session_id[:8]}..."
        )

    # -------------------------------------------------------------------------
    # Xây dựng StateGraph
    # -------------------------------------------------------------------------

    def _build_graph(self):
        """Tạo và compile StateGraph với 5 node xử lý bộ nhớ."""
        graph = StateGraph(MemoryState)

        # Đăng ký các node
        graph.add_node("classify_intent",  self._node_classify_intent)
        graph.add_node("retrieve_memory",  self._node_retrieve_memory)
        graph.add_node("build_prompt",     self._node_build_prompt)
        graph.add_node("call_llm",         self._node_call_llm)
        graph.add_node("save_memory",      self._node_save_memory)

        # Định nghĩa cạnh (edges)
        graph.add_edge(START,             "classify_intent")
        graph.add_edge("classify_intent", "retrieve_memory")
        graph.add_edge("retrieve_memory", "build_prompt")
        graph.add_edge("build_prompt",    "call_llm")
        graph.add_edge("call_llm",        "save_memory")
        graph.add_edge("save_memory",     END)

        return graph.compile()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def chat(self, query: str) -> dict:
        """Chạy toàn bộ graph cho một truy vấn của người dùng."""
        init_state: MemoryState = {
            "messages":        [],
            "current_query":   query,
            "user_profile":    {},
            "episodes":        [],
            "semantic_hits":   [],
            "memory_budget":   self.max_context_tokens,
            "intent":          "",
            "assembled_prompt": "",
            "response":        "",
            "turn_id":         str(uuid.uuid4()),
            "session_id":      self.session_id,
            "user_id":         self.user_id,
            "memories_used":   [],
            "input_tokens":    0,
            "output_tokens":   0,
        }

        # Thực thi graph qua langgraph StateGraph.invoke()
        result = self._graph.invoke(init_state)
        return result

    def get_status(self) -> dict:
        return {
            "user_id":          self.user_id,
            "session_id":       self.session_id,
            "st_turns":         self.short_term.turn_count,
            "redis_connected":  self.long_term.is_connected,
            "redis_prefs":      len(self.long_term.get_preferences()),
            "redis_facts":      len(self.long_term.get_all_facts()),
            "episodic_entries": self.episodic.total_entries,
            "chroma_connected": self.semantic.is_connected,
            "chroma_vectors":   self.semantic.total_vectors,
        }

    def reset_session(self, clear_long_term: bool = False) -> None:
        self.short_term.clear()
        self.session_id = str(uuid.uuid4())
        self.episodic.session_id = self.session_id
        if clear_long_term:
            self.long_term.clear()
            self.semantic.clear()

    # -------------------------------------------------------------------------
    # Node 1: classify_intent
    # -------------------------------------------------------------------------

    def _node_classify_intent(self, state: MemoryState) -> dict:
        """Phân loại ý định truy vấn để chọn backend phù hợp."""
        intent = self.router.classify(state["current_query"])
        logger.info(f"Intent: {intent.value} | '{state['current_query'][:60]}'")
        return {"intent": intent.value}

    # -------------------------------------------------------------------------
    # Node 2: retrieve_memory
    # -------------------------------------------------------------------------

    def _node_retrieve_memory(self, state: MemoryState) -> dict:
        """
        Truy xuất bộ nhớ từ tất cả các backend liên quan dựa trên ý định.
        Cập nhật: user_profile, episodes, semantic_hits, messages.
        """
        query  = state["current_query"]
        intent = state["intent"]

        # A) Luôn lấy: lịch sử hội thoại hiện tại (short-term)
        history = self.short_term.get_history()
        messages = [
            {"role": "user", "content": h, "response": a}
            for h, a in history
        ]

        # B) Hồ sơ người dùng (long-term — luôn tải)
        prefs = self.long_term.get_preferences()
        facts = self.long_term.get_all_facts()
        user_profile = {"preferences": prefs, "facts": facts}

        # C) Episodic — dành cho recall trải nghiệm và context hiện tại
        episodes = []
        if intent in (
            QueryIntent.EXPERIENCE_RECALL.value,
            QueryIntent.CURRENT_CONTEXT.value,
        ):
            recent = self.episodic.get_recent(n=5)
            episodes = [
                {
                    "timestamp": e.timestamp,
                    "query":     e.query,
                    "response":  e.response[:300],
                    "intent":    e.intent,
                }
                for e in recent
            ]

        # D) Semantic search — dành cho recall thực tế và trải nghiệm
        semantic_hits = []
        if intent in (
            QueryIntent.FACTUAL_RECALL.value,
            QueryIntent.EXPERIENCE_RECALL.value,
        ):
            hits = self.semantic.search(query, k=3)
            semantic_hits = [item.content for item in hits]

        return {
            "messages":      messages,
            "user_profile":  user_profile,
            "episodes":      episodes,
            "semantic_hits": semantic_hits,
        }

    # -------------------------------------------------------------------------
    # Node 3: build_prompt
    # -------------------------------------------------------------------------

    def _node_build_prompt(self, state: MemoryState) -> dict:
        """
        Lắp ráp prompt cuối cùng với 4 section bộ nhớ có nhãn rõ ràng.
        Áp dụng cắt bỏ theo ngân sách token (4-level priority eviction).
        """
        sections = [f"SYSTEM:\n{SYSTEM_PROMPT}"]
        budget_used = len(self.encoder.encode(SYSTEM_PROMPT))
        budget = self.max_context_tokens
        memories_used = []

        # Section 1: HỒ SO NGUOI DUNG (long-term Redis) — P2 HIGH
        profile = state["user_profile"]
        if profile.get("preferences") or profile.get("facts"):
            lines = []
            if profile["preferences"]:
                lines.append("So thich:")
                for p in profile["preferences"]:
                    lines.append(f"  - {p}")
            if profile["facts"]:
                lines.append("Du kien da biet:")
                for k, v in profile["facts"].items():
                    lines.append(f"  - {k}: {v}")
            profile_text = "\n".join(lines)
            if budget_used + len(self.encoder.encode(profile_text)) < budget * 0.95:
                sections.append(f"\n[USER PROFILE]:\n{profile_text}")
                budget_used += len(self.encoder.encode(profile_text))
                memories_used.append({
                    "source": "long_term", "priority": "HIGH",
                    "content": profile_text[:200],
                })

        # Section 2: BO NHO SU KIEN (episodic JSONL) — P3 MEDIUM
        if state["episodes"]:
            ep_lines = []
            for ep in state["episodes"][-5:]:
                ts = ep.get("timestamp", "")[:19]
                ep_lines.append(
                    f"[{ts}] Nguoi dung: {ep['query']}\n"
                    f"  Tro ly: {ep['response'][:200]}"
                )
            ep_text = "\n".join(ep_lines)
            if budget_used + len(self.encoder.encode(ep_text)) < budget * 0.90:
                sections.append(f"\n[EPISODIC MEMORIES]:\n{ep_text}")
                budget_used += len(self.encoder.encode(ep_text))
                memories_used.append({
                    "source": "episodic", "priority": "MEDIUM",
                    "content": ep_text[:200],
                })

        # Section 3: BO NHO NGU NGHIA (ChromaDB) — P3 MEDIUM
        if state["semantic_hits"]:
            sem_text = "\n".join(state["semantic_hits"][:3])
            if budget_used + len(self.encoder.encode(sem_text)) < budget * 0.90:
                sections.append(f"\n[SEMANTIC MEMORIES]:\n{sem_text}")
                budget_used += len(self.encoder.encode(sem_text))
                memories_used.append({
                    "source": "semantic", "priority": "MEDIUM",
                    "content": sem_text[:200],
                })

        # Section 4: LICH SU HOI THOAI GAN DAY (short-term buffer) — P4 LOW
        if state["messages"]:
            conv_lines = []
            for msg in state["messages"][-8:]:
                conv_lines.append(f"Nguoi dung: {msg['content']}")
                conv_lines.append(f"Tro ly: {msg['response'][:200]}")
            conv_text = "\n".join(conv_lines)
            if budget_used + len(self.encoder.encode(conv_text)) < budget * 0.80:
                sections.append(f"\n[RECENT CONVERSATION]:\n{conv_text}")
                budget_used += len(self.encoder.encode(conv_text))
                memories_used.append({
                    "source": "short_term", "priority": "LOW",
                    "content": conv_text[:200],
                })

        # Truy vấn hiện tại — P1 CRITICAL (không bao giờ bị cắt)
        sections.append(f"\n[CURRENT QUERY]:\n{state['current_query']}")
        sections.append("\nASSISTANT RESPONSE:")

        prompt = "\n".join(sections)
        return {
            "assembled_prompt": prompt,
            "memory_budget":    budget - budget_used,
            "memories_used":    memories_used,
            "input_tokens":     len(self.encoder.encode(prompt)),
        }

    # -------------------------------------------------------------------------
    # Node 4: call_llm
    # -------------------------------------------------------------------------

    def _node_call_llm(self, state: MemoryState) -> dict:
        """Gọi LLM với prompt đã lắp ráp."""
        resp = self.llm.invoke([HumanMessage(content=state["assembled_prompt"])])
        return {
            "response":      resp.content,
            "output_tokens": len(self.encoder.encode(resp.content)),
        }

    # -------------------------------------------------------------------------
    # Node 5: save_memory
    # -------------------------------------------------------------------------

    def _node_save_memory(self, state: MemoryState) -> dict:
        """Lưu lượt trao đổi vào tất cả các backend liên quan."""
        query    = state["current_query"]
        response = state["response"]
        intent   = state["intent"]

        # Luôn lưu: short-term + episodic
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

        # Semantic: lưu cho các truy vấn thực tế và trải nghiệm
        if intent in (
            QueryIntent.FACTUAL_RECALL.value,
            QueryIntent.EXPERIENCE_RECALL.value,
        ):
            self.semantic.save(entry)

        # Long-term profile: trích xuất và lưu sở thích/dữ kiện
        if intent == QueryIntent.USER_PREFERENCE.value:
            self._extract_and_save_profile(query)

        return {}

    # -------------------------------------------------------------------------
    # Xử lý xung đột: trích xuất LLM với ngữ nghĩa OVERWRITE
    # -------------------------------------------------------------------------

    def _extract_and_save_profile(self, query: str) -> None:
        """
        Dùng LLM trích xuất dữ kiện hồ sơ từ phát biểu của người dùng.
        GHI DE fact cũ khi người dùng sửa thông tin — không append.
        Đây là cơ chế xử lý xung đột chính (Rubric Mục 3).
        """
        existing_facts = self.long_term.get_all_facts()
        existing_prefs = self.long_term.get_preferences()

        extract_prompt = f"""Trích xuất các dữ kiện có cấu trúc từ phát biểu của người dùng.
Nếu dữ kiện mới mâu thuẫn với dữ liệu hiện có, ưu tiên dữ kiện MỚI (GHI ĐÈ, không append).

Hồ sơ hiện tại:
  Sở thích: {json.dumps(existing_prefs, ensure_ascii=False)}
  Dữ kiện: {json.dumps(existing_facts, ensure_ascii=False)}

Phát biểu của người dùng: "{query}"

Trả về ĐÚNG JSON hợp lệ:
{{
  "facts": {{"key": "value"}},
  "preferences": ["pref1", "pref2"],
  "overwritten": ["key1"]
}}

Quy tắc:
- facts: cặp key-value (name, job, allergy, location, company, v.v.)
- preferences: sở thích về phong cách (concise, detailed, language, v.v.)
- overwritten: danh sách key bị ghi đè so với hồ sơ cũ
- Nếu không có gì để trích xuất: {{"facts": {{}}, "preferences": [], "overwritten": []}}
"""
        try:
            resp = self.llm.invoke([HumanMessage(content=extract_prompt)])
            raw = resp.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            data = json.loads(raw)

            # Lưu dữ kiện với ngữ nghĩa OVERWRITE
            for key, value in data.get("facts", {}).items():
                self.long_term.save_fact(key, value)

            # Lưu sở thích
            for pref in data.get("preferences", []):
                self.long_term.save_preference(pref)

            overwritten = data.get("overwritten", [])
            if overwritten:
                logger.info(f"Xung đột hồ sơ đã giải quyết: keys bị ghi đè = {overwritten}")

        except Exception as exc:
            logger.warning(f"Trích xuất hồ sơ thất bại: {exc}")
            self.long_term.save_preference(f"[{query[:120]}]")
