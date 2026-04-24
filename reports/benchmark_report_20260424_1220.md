# 📊 Benchmark Report — Lab 17: Memory Systems for Agents
**Generated:** 2026-04-24 12:20  |  **Conversations:** 10  |  **Total Turns:** 61  |  **Memory Checkpoints:** 17

---

## Summary: Agent With Memory vs. Without Memory

| Metric | No Memory (Baseline) | With Memory | Delta |
|---|---|---|---|
| Response Relevance (1–5) | 3.90 | 4.16 | +0.26 (+7%) |
| Context Utilization | N/A | 32.8% | — |
| Token Efficiency (out/in) | 2.050 | 0.579 | -1.471 |
| Memory Hit Rate | N/A | 82.4% | — |
| Total Input Tokens | 2,065 | 34,353 | +32,288 |
| Total Output Tokens | 4,233 | 19,894 | +15,661 |

---

## Per-Conversation Results

| # | Conversation | Type Tested | Rel (Mem) | Rel (No Mem) | Ctx Util | Hit Rate |
|---|---|---|---|---|---|---|
| 1 | User Preference Tracking | preference | 4.80 | 4.40 | 53% | 100% |
| 2 | Multi-Session Continuity | long_term | 4.50 | 4.75 | 21% | 100% |
| 3 | Experience Recall — Cross-Turn Reference | episodic | 5.00 | 4.40 | 27% | 100% |
| 4 | Semantic Similarity Recall | semantic | 4.75 | 4.75 | 46% | 100% |
| 5 | Preference Override | preference | 1.75 | 4.50 | 38% | 50% |
| 6 | Factual Knowledge Retention | factual | 3.75 | 3.75 | 33% | 100% |
| 7 | Long Conversation — Context Trim Resilience | context_trim | 4.26 | 2.84 | 27% | 0% |
| 8 | Temporal Reasoning | episodic | 3.20 | 4.40 | 33% | 50% |
| 9 | Allergy Conflict Update (Mandatory Test) | conflict | 4.50 | 3.83 | 25% | 100% |
| 10 | Multi-Topic Context Switching | context | 4.40 | 4.80 | 37% | 100% |

---

## Memory Hit Rate Analysis

| Conv | Turn | Query (truncated) | Expected Memory | Result |
|---|---|---|---|---|
| 1 | 3 | What's the difference between a Python list a... | User is a Python engineer; prefers detailed technical a... | ✅ HIT |
| 1 | 5 | Recommend an algorithm for a multi-class clas... | User likes Python/ML + wants detailed technical answers... | ✅ HIT |
| 2 | 4 | Hello again! Can you recommend some Vietnames... | User's favorite Vietnamese dish is phở bò... | ✅ HIT |
| 2 | 5 | Any tips for picking up Vietnamese faster?... | User said they find Vietnamese tones challenging... | ✅ HIT |
| 3 | 5 | Going back to what you said about solar panel... | Turn 1 discussed solar panel efficiency types... | ✅ HIT |
| 4 | 4 | What are the advantages of consuming matcha o... | Turn 1 already discussed green tea/matcha health benefi... | ✅ HIT |
| 5 | 2 | What is machine learning?... | User wants ≤2 sentence answers... | ❌ MISS |
| 5 | 4 | What is deep learning?... | User updated preference: wants DETAILED/LONG explanatio... | ✅ HIT |
| 6 | 4 | Give me a quick company overview.... | Must include: TechViet, 250 employees, Hanoi, fintech, ... | ✅ HIT |
| 7 | 19 | What is my name and employee ID?... | Name=Mai, Employee ID=EMP-7890 stated in turn 1... | ❌ MISS |
| 8 | 4 | What was the very first question I asked you ... | First question was about the capital of France... | ❌ MISS |
| 8 | 5 | What did I ask you right before this message?... | Previous question was about recalling the first questio... | ✅ HIT |
| 9 | 3 | What foods should I avoid given my allergy?... | User is allergic to cow's milk... | ✅ HIT |
| 9 | 5 | Please update my profile. What's my allergy n... | Allergy corrected: soybeans (NOT cow's milk). Profile m... | ✅ HIT |
| 9 | 6 | Recommend some safe snacks for me based on my... | User allergy is soybeans. Must NOT mention cow's milk a... | ✅ HIT |
| 10 | 3 | Back to Japan — what are the must-see places ... | User is planning a Japan trip (turn 1)... | ✅ HIT |
| 10 | 5 | For my Japan trip, when is the best time to v... | User planning Japan trip + decided to learn JavaScript... | ✅ HIT |

---

## Token Budget Breakdown

| Conversation | In (Mem) | Out (Mem) | In (No) | Out (No) | Token Overhead |
|---|---|---|---|---|---|
| User Preference Tracking | 1,763 | 2,157 | 174 | 478 | +913% |
| Multi-Session Continuity | 1,808 | 1,193 | 133 | 833 | +1259% |
| Experience Recall — Cross-Turn Reference | 2,004 | 2,279 | 177 | 688 | +1032% |
| Semantic Similarity Recall | 1,668 | 1,985 | 138 | 751 | +1109% |
| Preference Override | 1,212 | 507 | 130 | 176 | +832% |
| Factual Knowledge Retention | 2,295 | 1,549 | 138 | 96 | +1563% |
| Long Conversation — Context Trim Resilience | 14,490 | 5,015 | 629 | 415 | +2204% |
| Temporal Reasoning | 2,663 | 641 | 160 | 97 | +1564% |
| Allergy Conflict Update (Mandatory Test) | 4,076 | 1,691 | 212 | 241 | +1823% |
| Multi-Topic Context Switching | 2,374 | 2,877 | 174 | 458 | +1264% |

---

## Architecture Summary

### 4 Memory Backends
| Backend | Technology | Persistence | Use Case |
|---|---|---|---|
| Short-term | ConversationBufferMemory | Session (RAM) | Current conversation history |
| Long-term | Redis 7 (Docker) | 30 days | User preferences & learned facts |
| Episodic | JSON JSONL file | Permanent | Ordered event journal |
| Semantic | ChromaDB (Docker) | Permanent | Cosine-similarity memory search |

### Memory Router Logic
| Intent | Backend | Trigger Keywords |
|---|---|---|
| USER_PREFERENCE | Redis | I like / I prefer / I want / sở thích |
| EXPERIENCE_RECALL | Episodic JSON | last time / you said / hôm qua / lần trước |
| FACTUAL_RECALL | ChromaDB | what is / how / why / explain / giải thích |
| CURRENT_CONTEXT | Short-term only | (all other queries) |

### Context Window — 4-Level Priority Eviction
| Priority | Content Type | Eviction Trigger |
|---|---|---|
| P1 CRITICAL | System prompt + current query | Never evicted |
| P2 HIGH | Redis preferences & facts | > 95% token limit |
| P3 MEDIUM | Recent episodic/semantic (<3 turns) | > 90% token limit |
| P4 LOW | Older conversation history | > 80% token limit |