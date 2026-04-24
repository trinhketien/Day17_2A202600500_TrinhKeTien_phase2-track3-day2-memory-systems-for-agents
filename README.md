# Lab 17 — Hệ thống bộ nhớ cho Agent (Memory Systems for Agents)

> **Khóa học:** AICB · Track 3 · Ngày 17
> **Học viên:** Trịnh Kế Tiên | **MSSV:** 2A202600500
> **Công nghệ:** Python 3.12 · LangChain 0.3 · OpenAI GPT-4o-mini · Redis · ChromaDB

---

## Cấu trúc dự án

```
Day17/
├── src/
│   ├── memory/
│   │   ├── models.py            # Các lớp dữ liệu dùng chung (MemoryEntry, ContextItem, ...)
│   │   ├── short_term.py        # Backend 1: ConversationBufferMemory
│   │   ├── long_term_redis.py   # Backend 2: Redis (lưu trữ lâu dài, TTL 30 ngày)
│   │   ├── episodic_json.py     # Backend 3: Nhật ký sự kiện dạng JSONL
│   │   ├── semantic_chroma.py   # Backend 4: ChromaDB (tìm kiếm vector ngữ nghĩa)
│   │   └── memory_router.py     # Bộ định tuyến: phân loại ý định → chọn backend
│   ├── context/
│   │   └── window_manager.py    # Quản lý cửa sổ ngữ cảnh 4 cấp ưu tiên
│   └── agent/
│       ├── langgraph_agent.py   # Agent chính: LangGraph MemoryState + graph nodes
│       ├── base_agent.py        # Agent dự phòng (pipeline cơ bản)
│       └── agent_no_memory.py   # Agent cơ sở (không có bộ nhớ, dùng làm baseline)
├── benchmark/
│   ├── conversations.py         # 10 kịch bản hội thoại đa lượt
│   ├── metrics.py               # Bộ đánh giá LLM-as-Judge
│   └── run_benchmark.py         # Chương trình chạy benchmark → xuất báo cáo
├── reports/                     # Thư mục chứa báo cáo tự động sinh
├── data/episodic/               # Thư mục dữ liệu (tạo tự động khi chạy)
├── BENCHMARK.md                 # Kết quả benchmark đầy đủ
├── REFLECTION.md                # Phân tích quyền riêng tư và giới hạn kỹ thuật
├── docker-compose.yml           # Cấu hình Redis + ChromaDB
├── requirements.txt
└── .env.example
```

---

## Hướng dẫn cài đặt và chạy

### 1. Cấu hình biến môi trường

```bash
cp .env.example .env
# Mở file .env và điền OPENAI_API_KEY
```

### 2. Khởi động hạ tầng

```bash
docker compose up -d
# Kiểm tra Redis:   docker compose ps
# Kiểm tra ChromaDB: curl http://localhost:8000/api/v1/heartbeat
```

Nếu không có Docker, hệ thống sẽ tự động sử dụng `fakeredis` (bộ nhớ trong) và `ChromaDB PersistentClient` (lưu trữ cục bộ) làm giải pháp dự phòng.

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Chạy benchmark

```bash
python benchmark/run_benchmark.py
```

Kết quả được lưu tại `reports/benchmark_report_YYYYMMDD_HHMM.md`.

---

## Kiến trúc hệ thống

### Bốn backend bộ nhớ

| Backend | Công nghệ | Thời gian lưu trữ | Chức năng |
|---|---|---|---|
| Ngắn hạn | `ConversationBufferMemory` | Trong phiên (RAM) | Lưu lịch sử hội thoại hiện tại |
| Dài hạn | Redis 7 (hoặc fakeredis) | 30 ngày (TTL) | Lưu sở thích và thông tin cá nhân |
| Sự kiện | File JSONL (append-only) | Vĩnh viễn | Nhật ký sự kiện theo thứ tự thời gian |
| Ngữ nghĩa | ChromaDB (hoặc PersistentClient) | Vĩnh viễn | Tìm kiếm tương tự bằng cosine similarity |

### Bộ định tuyến bộ nhớ (Memory Router)

Phân loại mỗi truy vấn vào một trong bốn ý định bằng mẫu regex:

| Ý định | Backend | Tín hiệu kích hoạt |
|---|---|---|
| `USER_PREFERENCE` | Redis | "tôi thích / tôi muốn / sở thích / I like / I prefer" |
| `EXPERIENCE_RECALL` | Episodic JSON | "lần trước / hôm qua / bạn đã nói / last time / you said" |
| `FACTUAL_RECALL` | ChromaDB | "là gì / như thế nào / giải thích / what is / explain" |
| `CURRENT_CONTEXT` | Short-term | Tất cả truy vấn còn lại |

### LangGraph MemoryState

Agent chính sử dụng `MemoryState(TypedDict)` với pipeline 5 node:

```
START → classify_intent → retrieve_memory → build_prompt → call_llm → save_memory → END
```

Prompt được chia thành 4 phần rõ ràng: `[USER PROFILE]`, `[EPISODIC MEMORIES]`, `[SEMANTIC MEMORIES]`, `[RECENT CONVERSATION]`.

### Quản lý cửa sổ ngữ cảnh — Cơ chế loại bỏ 4 cấp ưu tiên

| Ưu tiên | Nội dung | Điều kiện loại bỏ |
|---|---|---|
| P1 — Tối quan trọng | System prompt + truy vấn hiện tại | Không bao giờ loại bỏ |
| P2 — Cao | Sở thích và dữ kiện từ Redis | Khi vượt 95% giới hạn token |
| P3 — Trung bình | Bộ nhớ sự kiện/ngữ nghĩa gần đây | Khi vượt 90% giới hạn token |
| P4 — Thấp | Lịch sử hội thoại cũ | Khi vượt 80% giới hạn token |

---

## Các chỉ số đánh giá benchmark

| Chỉ số | Phương pháp | Ngưỡng đạt |
|---|---|---|
| Mức độ phù hợp phản hồi | LLM-as-Judge (GPT-4o-mini), thang 1–5 | ≥ 4.0 |
| Tỷ lệ sử dụng ngữ cảnh | Tỷ lệ bộ nhớ được truy xuất và thực sự sử dụng | ≥ 60% |
| Hiệu suất token | Tỷ lệ output tokens / input tokens | Càng cao càng tốt |
| Tỷ lệ nhớ đúng | Kết quả nhị phân tại 17 điểm kiểm tra | ≥ 80% |

---

## Chi phí API ước tính

| Giai đoạn | Nội dung | Chi phí |
|---|---|---|
| Benchmark | 10 kịch bản × ~6 lượt × 2 agent | ~$0.30 |
| LLM Judge | Đánh giá mức phù hợp + tỷ lệ nhớ đúng | ~$0.20 |
| Embedding | Vector ChromaDB | ~$0.01 |
| **Tổng cộng** | | **~$0.50** |
