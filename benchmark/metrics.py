"""
Benchmark metrics evaluation — LLM-as-Judge.

3 core metrics per turn:
  1. Response Relevance   (LLM judge, 1–5 scale)
  2. Context Utilization  (% of retrieved memories actually referenced)
  3. Token Efficiency     (output tokens / input tokens)

+ Memory Hit Rate (binary: did agent recall specified info?)
"""
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Data containers
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TurnMetrics:
    turn_id:            int
    conversation_id:    int
    query:              str
    response_mem:       str
    response_no_mem:    str

    relevance_mem:      float = 0.0
    relevance_no_mem:   float = 0.0
    relevance_reason:   str   = ""

    memories_retrieved: int   = 0
    memories_used:      int   = 0
    context_utilization: float = 0.0

    input_tokens_mem:   int   = 0
    output_tokens_mem:  int   = 0
    input_tokens_no:    int   = 0
    output_tokens_no:   int   = 0

    expects_memory:     bool          = False
    memory_hit:         Optional[bool] = None
    memory_hint:        str           = ""


@dataclass
class ConversationMetrics:
    conversation_id:   int
    conversation_name: str
    memory_type:       str
    turn_metrics:      List[TurnMetrics] = field(default_factory=list)

    @property
    def avg_relevance_mem(self) -> float:
        scores = [t.relevance_mem for t in self.turn_metrics if t.relevance_mem > 0]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_relevance_no(self) -> float:
        scores = [t.relevance_no_mem for t in self.turn_metrics if t.relevance_no_mem > 0]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_ctx_util(self) -> float:
        vals = [t.context_utilization for t in self.turn_metrics]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def token_efficiency_mem(self) -> float:
        tin  = sum(t.input_tokens_mem for t in self.turn_metrics)
        tout = sum(t.output_tokens_mem for t in self.turn_metrics)
        return tout / tin if tin > 0 else 0.0

    @property
    def memory_hit_rate(self) -> float:
        checkpoints = [t for t in self.turn_metrics if t.expects_memory and t.memory_hit is not None]
        if not checkpoints:
            return 0.0
        return sum(1 for t in checkpoints if t.memory_hit) / len(checkpoints)


# ────────────────────────────────────────────────────────────────────────────
# LLM-as-Judge evaluator
# ────────────────────────────────────────────────────────────────────────────

class BenchmarkEvaluator:
    """GPT-4o-mini judge for all benchmark metrics."""

    def __init__(self, judge_model: str = "gpt-4o-mini"):
        self.judge = ChatOpenAI(model=judge_model, temperature=0)

    # ── Metric 1: Response Relevance ────────────────────────────────
    def evaluate_relevance(
        self,
        query:          str,
        response:       str,
        memories_used:  List[str],
        has_memory:     bool,
    ) -> Tuple[float, str]:
        """Return (score 1-5, reason)."""
        memory_ctx = (
            "The agent had access to these memories:\n"
            + "\n".join(f"  - {m[:200]}" for m in memories_used)
            if has_memory and memories_used
            else "The agent had NO prior memory access."
        )

        prompt = f"""You are a neutral evaluator assessing an AI assistant's response.

User Query: {query}
{memory_ctx}
Agent Response: {response}

Rate the response quality on a 1-5 scale:
1 = Completely ignores available context; irrelevant or unhelpful
2 = Somewhat relevant but misses key contextual information
3 = Acceptable; partially uses available context
4 = Good; appropriately integrates relevant memory/context
5 = Excellent; perfectly personalized and accurate given the context

Return ONLY valid JSON: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

        try:
            raw  = self.judge.invoke([HumanMessage(content=prompt)])
            data = json.loads(raw.content.strip())
            return float(data["score"]), data.get("reason", "")
        except Exception as exc:
            logger.warning(f"evaluate_relevance failed: {exc}")
            return 3.0, "Evaluation failed"

    # ── Metric 2: Context Utilization ───────────────────────────────
    def estimate_context_utilization(
        self,
        query:              str,
        response:           str,
        retrieved_memories: List[str],
    ) -> float:
        """Estimate fraction of retrieved memories actually used in response."""
        if not retrieved_memories:
            return 0.0

        mem_list = "\n".join(f"{i+1}. {m[:200]}" for i, m in enumerate(retrieved_memories))
        prompt = f"""Examine the response and count how many of the retrieved memories were actually referenced or applied.

Query: {query}
Response: {response}

Retrieved Memories:
{mem_list}

Return ONLY valid JSON: {{"used": <integer>, "total": {len(retrieved_memories)}}}"""

        try:
            raw  = self.judge.invoke([HumanMessage(content=prompt)])
            data = json.loads(raw.content.strip())
            used  = int(data.get("used", 0))
            total = int(data.get("total", len(retrieved_memories)))
            return used / total if total > 0 else 0.0
        except Exception as exc:
            logger.warning(f"estimate_context_utilization failed: {exc}")
            return 0.0

    # ── Memory Hit Rate ─────────────────────────────────────────────
    def check_memory_hit(
        self,
        query:       str,
        response:    str,
        memory_hint: str,
    ) -> bool:
        """Return True if the agent correctly recalled the expected memory."""
        prompt = f"""Did the AI assistant correctly recall or apply the expected information?

Expected: {memory_hint}
User Query: {query}
Agent Response: {response}

Return ONLY valid JSON: {{"hit": true}} or {{"hit": false}}"""

        try:
            raw  = self.judge.invoke([HumanMessage(content=prompt)])
            data = json.loads(raw.content.strip())
            return bool(data.get("hit", False))
        except Exception as exc:
            logger.warning(f"check_memory_hit failed: {exc}")
            return False
