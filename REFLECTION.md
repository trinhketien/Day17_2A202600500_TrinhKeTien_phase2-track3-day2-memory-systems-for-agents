# Phân tích phản hồi: Quyền riêng tư, Giới hạn kỹ thuật và Đánh giá hệ thống bộ nhớ

> **Bài thực hành:** Lab 17 — Memory Systems for Agents
> **Học viên:** Trịnh Kế Tiên | **MSSV:** 2A202600500

---

## 1. Loại bộ nhớ nào mang lại hiệu quả cao nhất cho agent?

**Short-term memory** (ConversationBufferMemory) đóng vai trò nền tảng vì cho phép agent duy trì mạch hội thoại liên tục trong phiên làm việc, giữ nguyên ngữ cảnh cho đến khi bị cắt bỏ do giới hạn token.

Tuy nhiên, xét ở góc độ giá trị lâu dài, **long-term profile** (Redis) là loại bộ nhớ mang lại lợi ích thiết thực nhất khi hoạt động xuyên phiên. Nó cho phép agent ghi nhớ danh tính, sở thích và các thông tin cá nhân của người dùng bất kể số lần khởi động lại hệ thống.

**Nhận xét:** Short-term phục vụ tốt trong phạm vi một phiên, long-term phục vụ tốt xuyên phiên. Cả hai đều cần thiết, song nếu chỉ được chọn một thì long-term profile sẽ được ưu tiên vì khả năng duy trì thông tin lâu dài.

---

## 2. Loại bộ nhớ nào tiềm ẩn rủi ro lớn nhất khi truy xuất sai?

**Long-term profile** là loại có rủi ro cao nhất, bởi:

1. **Sai lệch thông tin cá nhân (PII):** Nếu hồ sơ lưu `allergy = "sữa bò"` nhưng thực tế người dùng dị ứng đậu nành, agent có thể đề xuất thực phẩm gây nguy hiểm sức khỏe.
2. **Dữ liệu cũ không được cập nhật (stale facts):** Profile không tự động hết hạn, dẫn đến khả năng thông tin trở nên lỗi thời theo thời gian — ví dụ: người dùng chuyển nơi ở hoặc thay đổi nghề nghiệp.
3. **Trộn lẫn dữ liệu giữa các người dùng (cross-user contamination):** Nếu hệ thống quản lý `user_id` bị trùng, thông tin của người dùng A có thể bị truy xuất nhầm cho người dùng B.

**Semantic memory** cũng có rủi ro khi trả về các đoạn văn bản không liên quan do điểm cosine similarity cao nhưng ngữ cảnh khác biệt — hiện tượng này có thể gọi là "ảo giác truy xuất ngữ nghĩa" (semantic retrieval hallucination).

---

## 3. Khi người dùng yêu cầu xóa dữ liệu, cần xóa ở những backend nào?

Theo nguyên tắc Quyền được xóa (Right to Erasure) của GDPR — Điều 17, hệ thống cần xóa dữ liệu trên **tất cả bốn backend**:

| Backend | Phương thức xóa | Độ khó |
|---------|-----------------|--------|
| Short-term | `buffer.clear()` — tự mất khi kết thúc phiên | Thấp |
| Long-term (Redis) | `redis.delete(key)` — xóa toàn bộ hồ sơ | Thấp |
| Episodic (JSONL) | Lọc và ghi lại file loại trừ các dòng theo `user_id` | Trung bình |
| Semantic (ChromaDB) | `collection.delete(where={"user_id": ...})` | Trung bình |

**Rủi ro chính:** Episodic log sử dụng cơ chế append-only, do đó việc xóa một bản ghi cụ thể yêu cầu đọc lại toàn bộ file và ghi lại — không hỗ trợ xóa nguyên tử (atomic delete).

**Khuyến nghị:** Cần triển khai hàm `delete_all_user_data(user_id)` gọi đồng thời cả bốn backend để đảm bảo xóa triệt để.

---

## 4. Những yếu tố nào khiến hệ thống gặp vấn đề khi mở rộng quy mô?

### 4.1 Episodic memory (JSONL file)
- Mỗi người dùng tương ứng một file riêng. Với 10.000 người dùng, hệ thống cần quản lý 10.000 file — dễ chạm giới hạn file handle của hệ điều hành.
- File lớn dẫn đến `load_all()` có độ phức tạp O(n), gây suy giảm hiệu suất đáng kể.
- **Hướng khắc phục:** Chuyển sang SQLite hoặc PostgreSQL.

### 4.2 Semantic memory (ChromaDB)
- Sử dụng chung một collection cho tất cả người dùng, khiến tốc độ cosine search giảm rõ rệt khi vượt quá 1 triệu vector.
- Điều kiện lọc `where={"user_id": ...}` không được đánh chỉ mục (index), dẫn đến full scan.
- **Hướng khắc phục:** Phân tách collection theo người dùng hoặc sử dụng dịch vụ vector database có quản lý (Pinecone, Weaviate).

### 4.3 Áp lực ngân sách token
- Người dùng có hội thoại dài sẽ kích hoạt cơ chế loại bỏ P4 liên tục, có thể mất đi ngữ cảnh quan trọng.
- Giải pháp hiện tại áp dụng eviction 4 cấp độ ưu tiên, nhưng chưa có cơ chế đánh giá mức độ quan trọng (importance scoring) cho từng mục.
- **Hướng khắc phục:** Bổ sung lớp tóm tắt tự động (LLM-based summarization) cho các tin nhắn cũ trước khi loại bỏ.

### 4.4 Redis đơn thể (single-instance)
- Không có cơ chế phân cụm (clustering), tạo thành điểm lỗi duy nhất (single point of failure).
- **Hướng khắc phục:** Sử dụng Redis Cluster hoặc Redis Sentinel để đảm bảo tính sẵn sàng cao (HA).

---

## 5. Phân tích rủi ro về quyền riêng tư và thông tin cá nhân (PII)

### 5.1 Dữ liệu được lưu trữ tại từng backend

| Backend | Dữ liệu PII | Mức độ nhạy cảm |
|---------|-------------|-----------------|
| Short-term | Tên, câu hỏi, ý kiến cá nhân | Trung bình (tự xóa khi kết thúc phiên) |
| Long-term | Tên, thông tin dị ứng, sở thích, nghề nghiệp | **Cao** (lưu trữ lâu dài) |
| Episodic | Toàn bộ lịch sử hội thoại | **Cao** (lưu vĩnh viễn) |
| Semantic | Embedding vector của văn bản chứa PII | Trung bình (không đọc trực tiếp được) |

### 5.2 Các rủi ro cụ thể

1. **Không mã hóa dữ liệu lưu trữ (no encryption at rest):** Redis và JSONL đều lưu dạng plaintext. Bất kỳ ai có quyền truy cập máy chủ đều có thể đọc toàn bộ hồ sơ người dùng.

2. **Thiếu cơ chế xin phép (no consent mechanism):** Hệ thống tự động lưu PII khi người dùng phát biểu "tôi tên là X, tôi dị ứng Y" mà không hỏi xác nhận "bạn có muốn tôi ghi nhớ thông tin này không?".

3. **Thiếu thời hạn sống cho episodic log (no TTL):** Redis có TTL 30 ngày, nhưng episodic log thì tồn tại vĩnh viễn mà không có cơ chế tự động dọn dẹp.

4. **Truy xuất sai ngữ cảnh:** Semantic search có thể trả về đoạn văn bản từ một cuộc hội thoại cũ và đưa vào prompt hiện tại, khiến LLM sinh ra câu trả lời dựa trên ngữ cảnh không phù hợp.

### 5.3 Kế hoạch giảm thiểu rủi ro

| Rủi ro | Biện pháp giảm thiểu |
|--------|---------------------|
| Không mã hóa | Kích hoạt TLS cho Redis, mã hóa file JSONL khi lưu trữ |
| Thiếu cơ chế xin phép | Bổ sung luồng xác nhận tường minh trước khi lưu PII |
| Thiếu TTL cho episodic | Cấu hình `max_age_days`, tự động xóa các bản ghi quá thời hạn |
| Truy xuất sai ngữ cảnh | Tăng ngưỡng similarity (0.60 → 0.75), bổ sung điểm tin cậy (confidence score) |
| Thiếu API xóa dữ liệu | Triển khai hàm `delete_user(user_id)` hoạt động trên cả bốn backend |

---

## 6. Giới hạn kỹ thuật của giải pháp hiện tại

1. **Memory router chỉ sử dụng regex:** Không nhận diện được sắc thái ngữ nghĩa. Ví dụ: câu "Tôi không thích Python" bị phân loại là `USER_PREFERENCE` nhưng thực chất mang nghĩa phủ định, không nên lưu như một sở thích.

2. **Xử lý xung đột phụ thuộc vào khả năng trích xuất của LLM:** Nếu LLM trả về JSON không hợp lệ, hồ sơ sẽ không được cập nhật. Cần có cơ chế dự phòng (fallback) mạnh hơn.

3. **Thiếu cơ chế xếp hạng mức độ quan trọng của bộ nhớ:** Tất cả các bản ghi episodic đều có cùng mức ưu tiên. Giải pháp lý tưởng là sử dụng LLM để đánh giá mức quan trọng (1–5) tại thời điểm lưu.

4. **Thiết kế đơn người dùng (single-user):** Chưa có cơ chế cách ly dữ liệu giữa các người dùng (multi-tenant isolation). Nếu hai người dùng trùng `user_id`, dữ liệu sẽ bị gộp lẫn.

5. **Thiếu cơ chế nén bộ nhớ:** Sau 100 lượt trao đổi, episodic log trở nên rất lớn. Cần bổ sung lớp tóm tắt tự động để giảm kích thước cho các bản ghi cũ.

6. **ChromaDB fallback cục bộ không có cơ chế sao lưu:** PersistentClient lưu dữ liệu trên đĩa cục bộ nhưng không có bản sao (replication) hay sao lưu tự động (backup).
