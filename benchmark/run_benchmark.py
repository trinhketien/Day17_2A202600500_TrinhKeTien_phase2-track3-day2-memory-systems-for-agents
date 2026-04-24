"""
Benchmark runner — Lab 17: Memory Systems for Agents
=====================================================
Runs all 10 conversations against both agents,
scores 3 metrics per turn, and generates:
  reports/benchmark_report_YYYYMMDD_HHMM.md
  reports/benchmark_raw.json
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

# Ensure project root is on sys.path when run from any directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.conversations import CONVERSATIONS, Conversation
from benchmark.metrics import BenchmarkEvaluator, ConversationMetrics, TurnMetrics
from src.agent.agent_no_memory import NoMemoryAgent
from src.agent.langgraph_agent import LangGraphMemoryAgent

logging.basicConfig(level=logging.WARNING)
console = Console()
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Run one conversation
# ────────────────────────────────────────────────────────────────────────────

def run_conversation(
    conv:           Conversation,
    mem_agent:      LangGraphMemoryAgent,
    no_mem_agent:   NoMemoryAgent,
    evaluator:      BenchmarkEvaluator,
) -> ConversationMetrics:

    result = ConversationMetrics(
        conversation_id=conv.id,
        conversation_name=conv.name,
        memory_type=conv.memory_type_tested,
    )

    for idx, turn in enumerate(conv.turns):
        # ── Session-restart marker ────────────────────────────────
        if "SESSION RESTART" in turn.user:
            mem_agent.reset_session(clear_long_term=False)
            console.print("    [yellow]⟳  Session restarted[/yellow]")
            continue

        console.print(f"    [dim]turn {idx+1}: {turn.user[:70]}[/dim]")

        # ── Run both agents ───────────────────────────────────────
        mem_state   = mem_agent.chat(turn.user)
        no_mem_resp = no_mem_agent.chat(turn.user)

        mem_response_text = mem_state["response"]
        mem_memories = mem_state.get("memories_used", [])
        memories_str = [m.get("content", "")[:200] for m in mem_memories]

        # ── Metric 1: Relevance ───────────────────────────────────
        rel_mem,   reason = evaluator.evaluate_relevance(
            query=turn.user, response=mem_response_text,
            memories_used=memories_str, has_memory=True,
        )
        rel_no, _ = evaluator.evaluate_relevance(
            query=turn.user, response=no_mem_resp["response"],
            memories_used=[], has_memory=False,
        )

        # ── Metric 2: Context utilization ────────────────────────
        ctx_util = (
            evaluator.estimate_context_utilization(
                query=turn.user,
                response=mem_response_text,
                retrieved_memories=memories_str,
            )
            if memories_str else 0.0
        )

        # ── Memory Hit Rate ───────────────────────────────────────
        hit = None
        if turn.expects_memory:
            hit = evaluator.check_memory_hit(
                query=turn.user,
                response=mem_response_text,
                memory_hint=turn.memory_hint,
            )
            icon = "PASS" if hit else "FAIL"
            console.print(f"       Memory recall: {icon}  ({turn.memory_hint[:60]}...)")

        result.turn_metrics.append(TurnMetrics(
            turn_id=idx,
            conversation_id=conv.id,
            query=turn.user,
            response_mem=mem_response_text,
            response_no_mem=no_mem_resp["response"],
            relevance_mem=rel_mem,
            relevance_no_mem=rel_no,
            relevance_reason=reason,
            memories_retrieved=len(mem_memories),
            memories_used=round(ctx_util * len(mem_memories)),
            context_utilization=ctx_util,
            input_tokens_mem=mem_state.get("input_tokens", 0),
            output_tokens_mem=mem_state.get("output_tokens", 0),
            input_tokens_no=no_mem_resp["input_tokens"],
            output_tokens_no=no_mem_resp["output_tokens"],
            expects_memory=turn.expects_memory,
            memory_hit=hit,
            memory_hint=turn.memory_hint,
        ))

    return result


# ────────────────────────────────────────────────────────────────────────────
# Report generator
# ────────────────────────────────────────────────────────────────────────────

def generate_report(all_metrics: List[ConversationMetrics]) -> str:
    ts         = datetime.now().strftime("%Y-%m-%d %H:%M")
    all_turns  = [t for cm in all_metrics for t in cm.turn_metrics]
    n          = len(all_turns)

    avg_rel_mem = sum(t.relevance_mem    for t in all_turns) / n if n else 0
    avg_rel_no  = sum(t.relevance_no_mem for t in all_turns) / n if n else 0

    ctx_vals    = [t.context_utilization for t in all_turns if t.memories_retrieved > 0]
    avg_ctx     = (sum(ctx_vals) / len(ctx_vals) * 100) if ctx_vals else 0

    tin_mem  = sum(t.input_tokens_mem  for t in all_turns)
    tout_mem = sum(t.output_tokens_mem for t in all_turns)
    tin_no   = sum(t.input_tokens_no   for t in all_turns)
    tout_no  = sum(t.output_tokens_no  for t in all_turns)
    eff_mem  = tout_mem / tin_mem if tin_mem else 0
    eff_no   = tout_no  / tin_no  if tin_no  else 0

    hit_turns = [t for t in all_turns if t.expects_memory and t.memory_hit is not None]
    hit_rate  = (sum(1 for t in hit_turns if t.memory_hit) / len(hit_turns) * 100) if hit_turns else 0

    rel_delta = avg_rel_mem - avg_rel_no
    rel_pct   = rel_delta / avg_rel_no * 100 if avg_rel_no else 0

    L = []  # lines

    def h(text):    L.append(text)
    def br():       L.append("")
    def rule():     L.append("---"); br()

    h(f"# Benchmark Report — Lab 17: Memory Systems for Agents")
    h(f"**Generated:** {ts}  |  **Conversations:** {len(all_metrics)}  |  "
      f"**Total Turns:** {n}  |  **Memory Checkpoints:** {len(hit_turns)}")
    br(); rule()

    h("## Summary: Agent With Memory vs. Without Memory")
    br()
    h("| Metric | No Memory (Baseline) | With Memory | Delta |")
    h("|---|---|---|---|")
    h(f"| Response Relevance (1–5) | {avg_rel_no:.2f} | {avg_rel_mem:.2f} | {rel_delta:+.2f} ({rel_pct:+.0f}%) |")
    h(f"| Context Utilization | N/A | {avg_ctx:.1f}% | — |")
    h(f"| Token Efficiency (out/in) | {eff_no:.3f} | {eff_mem:.3f} | {eff_mem-eff_no:+.3f} |")
    h(f"| Memory Hit Rate | N/A | {hit_rate:.1f}% | — |")
    h(f"| Total Input Tokens | {tin_no:,} | {tin_mem:,} | +{tin_mem-tin_no:,} |")
    h(f"| Total Output Tokens | {tout_no:,} | {tout_mem:,} | {tout_mem-tout_no:+,} |")
    br(); rule()

    h("## Per-Conversation Results")
    br()
    h("| # | Conversation | Type Tested | Rel (Mem) | Rel (No Mem) | Ctx Util | Hit Rate |")
    h("|---|---|---|---|---|---|---|")
    for cm in all_metrics:
        h(f"| {cm.conversation_id} | {cm.conversation_name} | {cm.memory_type} | "
          f"{cm.avg_relevance_mem:.2f} | {cm.avg_relevance_no:.2f} | "
          f"{cm.avg_ctx_util*100:.0f}% | {cm.memory_hit_rate*100:.0f}% |")
    br(); rule()

    h("## Memory Hit Rate Analysis")
    br()
    h("| Conv | Turn | Query (truncated) | Expected Memory | Result |")
    h("|---|---|---|---|---|")
    for cm in all_metrics:
        for t in cm.turn_metrics:
            if t.expects_memory:
                icon = "HIT" if t.memory_hit else "MISS"
                h(f"| {cm.conversation_id} | {t.turn_id+1} | {t.query[:45]}... | "
                  f"{t.memory_hint[:55]}... | {icon} |")
    br(); rule()

    h("## Token Budget Breakdown")
    br()
    h("| Conversation | In (Mem) | Out (Mem) | In (No) | Out (No) | Token Overhead |")
    h("|---|---|---|---|---|---|")
    for cm in all_metrics:
        in_m  = sum(t.input_tokens_mem  for t in cm.turn_metrics)
        out_m = sum(t.output_tokens_mem for t in cm.turn_metrics)
        in_n  = sum(t.input_tokens_no   for t in cm.turn_metrics)
        out_n = sum(t.output_tokens_no  for t in cm.turn_metrics)
        ovhd  = (in_m - in_n) / in_n * 100 if in_n else 0
        h(f"| {cm.conversation_name} | {in_m:,} | {out_m:,} | {in_n:,} | {out_n:,} | +{ovhd:.0f}% |")
    br(); rule()

    h("## Architecture Summary")
    br()
    h("### 4 Memory Backends")
    h("| Backend | Technology | Persistence | Use Case |")
    h("|---|---|---|---|")
    h("| Short-term | ConversationBufferMemory | Session (RAM) | Current conversation history |")
    h("| Long-term | Redis 7 (Docker) | 30 days | User preferences & learned facts |")
    h("| Episodic | JSON JSONL file | Permanent | Ordered event journal |")
    h("| Semantic | ChromaDB (Docker) | Permanent | Cosine-similarity memory search |")
    br()
    h("### Memory Router Logic")
    h("| Intent | Backend | Trigger Keywords |")
    h("|---|---|---|")
    h("| USER_PREFERENCE | Redis | I like / I prefer / I want / sở thích |")
    h("| EXPERIENCE_RECALL | Episodic JSON | last time / you said / hôm qua / lần trước |")
    h("| FACTUAL_RECALL | ChromaDB | what is / how / why / explain / giải thích |")
    h("| CURRENT_CONTEXT | Short-term only | (all other queries) |")
    br()
    h("### Context Window — 4-Level Priority Eviction")
    h("| Priority | Content Type | Eviction Trigger |")
    h("|---|---|---|")
    h("| P1 CRITICAL | System prompt + current query | Never evicted |")
    h("| P2 HIGH | Redis preferences & facts | > 95% token limit |")
    h("| P3 MEDIUM | Recent episodic/semantic (<3 turns) | > 90% token limit |")
    h("| P4 LOW | Older conversation history | > 80% token limit |")

    return "\n".join(L)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold blue]Lab 17 — Memory Systems Benchmark[/bold blue]")

    # ── Init agents ───────────────────────────────────────────────
    mem_agent = LangGraphMemoryAgent(
        user_id="bm_user",
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "8000")),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
    )
    no_mem_agent = NoMemoryAgent(llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    evaluator    = BenchmarkEvaluator(judge_model="gpt-4o-mini")

    status = mem_agent.get_status()
    console.print(f"Redis   : {'connected' if status['redis_connected']  else 'offline'}")
    console.print(f"ChromaDB: {'connected' if status['chroma_connected'] else 'offline'}")

    # ── Run all conversations ─────────────────────────────────────
    all_metrics: List[ConversationMetrics] = []

    for conv in CONVERSATIONS:
        console.print(f"\n[bold yellow]▶ [{conv.id}/10] {conv.name}[/bold yellow]")
        console.print(f"  [dim]{conv.description}[/dim]")
        mem_agent.reset_session()
        cm = run_conversation(conv, mem_agent, no_mem_agent, evaluator)
        all_metrics.append(cm)
        console.print(
            f"  [green]Relevance {cm.avg_relevance_mem:.1f}/5 (mem) "
            f"vs {cm.avg_relevance_no:.1f}/5 (no-mem) | "
            f"Hit rate {cm.memory_hit_rate*100:.0f}%[/green]"
        )

    # ── Generate reports ──────────────────────────────────────────
    report_md = generate_report(all_metrics)
    ts_str    = datetime.now().strftime("%Y%m%d_%H%M")
    md_path   = REPORTS_DIR / f"benchmark_report_{ts_str}.md"
    md_path.write_text(report_md, encoding="utf-8")

    raw_path = REPORTS_DIR / "benchmark_raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "conversations": [
                    {
                        "id":              cm.conversation_id,
                        "name":            cm.conversation_name,
                        "type":            cm.memory_type,
                        "relevance_mem":   cm.avg_relevance_mem,
                        "relevance_no":    cm.avg_relevance_no,
                        "ctx_util":        cm.avg_ctx_util,
                        "hit_rate":        cm.memory_hit_rate,
                        "token_eff_mem":   cm.token_efficiency_mem,
                    }
                    for cm in all_metrics
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ── Print final table ─────────────────────────────────────────
    all_turns  = [t for cm in all_metrics for t in cm.turn_metrics]
    n          = len(all_turns)
    avg_r_m    = sum(t.relevance_mem    for t in all_turns) / n if n else 0
    avg_r_n    = sum(t.relevance_no_mem for t in all_turns) / n if n else 0
    ctx_vals   = [t.context_utilization for t in all_turns if t.memories_retrieved > 0]
    avg_ctx    = (sum(ctx_vals) / len(ctx_vals) * 100) if ctx_vals else 0
    hits       = [t for t in all_turns if t.expects_memory and t.memory_hit is not None]
    hit_rate   = (sum(1 for t in hits if t.memory_hit) / len(hits) * 100) if hits else 0

    tbl = Table(title="Final Benchmark Results", show_header=True)
    tbl.add_column("Metric",            style="cyan")
    tbl.add_column("No Memory",         style="red")
    tbl.add_column("With Memory",       style="green")
    tbl.add_column("Delta",             style="yellow")
    tbl.add_row("Response Relevance (1-5)", f"{avg_r_n:.2f}", f"{avg_r_m:.2f}", f"{avg_r_m-avg_r_n:+.2f}")
    tbl.add_row("Context Utilization",     "N/A",             f"{avg_ctx:.1f}%", "—")
    tbl.add_row("Memory Hit Rate",         "N/A",             f"{hit_rate:.1f}%","—")
    console.print(tbl)
    console.print(f"\n[bold green]Report → {md_path}[/bold green]")
    console.print(f"[bold green]Raw data → {raw_path}[/bold green]")


if __name__ == "__main__":
    main()
