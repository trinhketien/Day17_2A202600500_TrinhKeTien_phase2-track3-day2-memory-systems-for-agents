# Báo cáo Benchmark — Lab 17: Hệ thống bộ nhớ cho Agent

**Ngày thực hiện:** 2026-04-24 12:20
**Số kịch bản:** 10 | **Tổng số lượt trao đổi:** 61 | **Số điểm kiểm tra bộ nhớ:** 17

---

## Tổng quan: So sánh Agent có bộ nhớ và không có bộ nhớ

| Chỉ số | Không có bộ nhớ (Baseline) | Có bộ nhớ | Chênh lệch |
|---|---|---|---|
| Mức độ phù hợp của phản hồi (1–5) | 3.90 | 4.16 | +0.26 (+7%) |
| Tỷ lệ sử dụng ngữ cảnh | Không áp dụng | 32.8% | — |
| Hiệu suất token (output/input) | 2.050 | 0.579 | -1.471 |
| Tỷ lệ nhớ đúng (Memory Hit Rate) | Không áp dụng | 82.4% | — |
| Tổng input tokens | 2,065 | 34,353 | +32,288 |
| Tổng output tokens | 4,233 | 19,894 | +15,661 |

**Nhận xét:** Agent có bộ nhớ đạt mức phù hợp cao hơn 7% so với baseline, đồng thời đạt tỷ lệ nhớ đúng 82.4% (14/17 điểm kiểm tra). Chi phí token tăng đáng kể do việc đưa ngữ cảnh bộ nhớ vào prompt.

---

## Kết quả theo từng kịch bản

| # | Kịch bản | Nhóm kiểm thử | Phù hợp (Có nhớ) | Phù hợp (Không nhớ) | Sử dụng ngữ cảnh | Tỷ lệ nhớ đúng |
|---|---|---|---|---|---|---|
| 1 | Theo dõi sở thích người dùng | Ghi nhớ hồ sơ | 4.80 | 4.40 | 53% | 100% |
| 2 | Duy trì xuyên phiên | Bộ nhớ dài hạn | 4.50 | 4.75 | 21% | 100% |
| 3 | Nhớ lại trải nghiệm trước | Bộ nhớ sự kiện | 5.00 | 4.40 | 27% | 100% |
| 4 | Truy xuất ngữ nghĩa tương tự | Bộ nhớ ngữ nghĩa | 4.75 | 4.75 | 46% | 100% |
| 5 | Thay đổi sở thích giữa chừng | Xử lý xung đột | 1.75 | 4.50 | 38% | 50% |
| 6 | Ghi nhớ thông tin công ty | Ghi nhớ hồ sơ | 3.75 | 3.75 | 33% | 100% |
| 7 | Hội thoại dài — kiểm tra cắt ngữ cảnh | Ngân sách token | 4.26 | 2.84 | 27% | 0% |
| 8 | Suy luận theo thứ tự thời gian | Bộ nhớ sự kiện | 3.20 | 4.40 | 33% | 50% |
| 9 | Cập nhật xung đột dị ứng (bài kiểm tra bắt buộc) | Xử lý xung đột | 4.50 | 3.83 | 25% | 100% |
| 10 | Chuyển đổi chủ đề liên tục | Theo dõi ngữ cảnh | 4.40 | 4.80 | 37% | 100% |

---

## Phân tích chi tiết tỷ lệ nhớ đúng

| Kịch bản | Lượt | Truy vấn (rút gọn) | Thông tin cần nhớ | Kết quả |
|---|---|---|---|---|
| 1 | 3 | Sự khác biệt giữa list và tuple trong Python... | Người dùng là kỹ sư Python, thích câu trả lời chi tiết | Đúng |
| 1 | 5 | Đề xuất thuật toán cho bài toán phân loại... | Người dùng thích Python/ML, muốn câu trả lời kỹ thuật | Đúng |
| 2 | 4 | Xin chào lại! Đề xuất món Việt Nam... | Món yêu thích là phở bò | Đúng |
| 2 | 5 | Mẹo học tiếng Việt nhanh hơn... | Người dùng thấy thanh điệu khó | Đúng |
| 3 | 5 | Quay lại về tấm pin năng lượng mặt trời... | Lượt 1 đã thảo luận về hiệu suất pin mặt trời | Đúng |
| 4 | 4 | Lợi ích của việc uống matcha thường xuyên... | Lượt 1 đã thảo luận về trà xanh/matcha | Đúng |
| 5 | 2 | Machine learning là gì... | Người dùng muốn câu trả lời ngắn gọn (tối đa 2 câu) | **Sai** |
| 5 | 4 | Deep learning là gì... | Người dùng đổi ý: muốn giải thích chi tiết | Đúng |
| 6 | 4 | Tổng quan về công ty... | TechViet, 250 nhân viên, Hà Nội, fintech, PayBridge | Đúng |
| 7 | 19 | Tên tôi và mã nhân viên là gì... | Tên=Mai, Mã NV=EMP-7890 (đã nêu ở lượt 1) | **Sai** |
| 8 | 4 | Câu hỏi đầu tiên tôi hỏi là gì... | Câu hỏi đầu tiên về thủ đô nước Pháp | **Sai** |
| 8 | 5 | Câu hỏi ngay trước đây là gì... | Câu hỏi trước về việc nhớ lại câu hỏi đầu tiên | Đúng |
| 9 | 3 | Tôi nên tránh ăn gì với dị ứng này... | Người dùng dị ứng sữa bò | Đúng |
| 9 | 5 | Cập nhật hồ sơ. Dị ứng của tôi giờ là gì... | Đã sửa: dị ứng đậu nành (không phải sữa bò) | Đúng |
| 9 | 6 | Đề xuất đồ ăn vặt an toàn theo hồ sơ dị ứng... | Dị ứng đậu nành, không được nhắc đến sữa bò | Đúng |
| 10 | 3 | Quay lại Nhật Bản — địa điểm nên đến ở Tokyo... | Người dùng đang lên kế hoạch đi Nhật (lượt 1) | Đúng |
| 10 | 5 | Cho chuyến đi Nhật, khi nào nên đến Kyoto... | Đi Nhật + đã chọn học JavaScript | Đúng |

**Tổng kết:** 14/17 điểm kiểm tra đạt (82.4%).

**Phân tích các trường hợp sai:**
- **Kịch bản 5, lượt 2:** Sở thích mới ("ngắn gọn 2 câu") chưa kịp được lưu vào long-term profile trước khi truy vấn tiếp theo đến.
- **Kịch bản 7, lượt 19:** Thông tin từ lượt 1 bị loại bỏ sau 17 lượt filler do cơ chế eviction P4 hoạt động đúng thiết kế — đây là giới hạn có chủ đích của cửa sổ ngữ cảnh.
- **Kịch bản 8, lượt 4:** Agent có bộ nhớ hội thoại nhưng không có cơ chế đánh chỉ mục theo thứ tự lượt (turn index), nên khó trả lời chính xác "câu hỏi đầu tiên là gì".

---

## Phân tích ngân sách token

| Kịch bản | Input (Có nhớ) | Output (Có nhớ) | Input (Không nhớ) | Output (Không nhớ) | Chi phí thêm |
|---|---|---|---|---|---|
| Theo dõi sở thích | 1,763 | 2,157 | 174 | 478 | +913% |
| Duy trì xuyên phiên | 1,808 | 1,193 | 133 | 833 | +1,259% |
| Nhớ lại trải nghiệm | 2,004 | 2,279 | 177 | 688 | +1,032% |
| Truy xuất ngữ nghĩa | 1,668 | 1,985 | 138 | 751 | +1,109% |
| Thay đổi sở thích | 1,212 | 507 | 130 | 176 | +832% |
| Ghi nhớ thông tin | 2,295 | 1,549 | 138 | 96 | +1,563% |
| Hội thoại dài (cắt ngữ cảnh) | 14,490 | 5,015 | 629 | 415 | +2,204% |
| Suy luận thời gian | 2,663 | 641 | 160 | 97 | +1,564% |
| Xung đột dị ứng (bắt buộc) | 4,076 | 1,691 | 212 | 241 | +1,823% |
| Chuyển đổi chủ đề | 2,374 | 2,877 | 174 | 458 | +1,264% |

**Nhận xét:** Chi phí token tăng trung bình khoảng 10–15 lần so với baseline. Đây là đánh đổi cần thiết để đạt được khả năng nhớ ngữ cảnh. Trong triển khai thực tế, có thể giảm chi phí bằng cách áp dụng tóm tắt tự động (summarization) thay vì đưa toàn bộ lịch sử vào prompt.

---

## Tóm tắt kiến trúc

### Bốn backend bộ nhớ

| Backend | Công nghệ | Thời gian lưu trữ | Chức năng |
|---|---|---|---|
| Ngắn hạn (Short-term) | ConversationBufferMemory | Trong phiên (RAM) | Lưu lịch sử hội thoại hiện tại |
| Dài hạn (Long-term) | Redis 7 | 30 ngày (TTL) | Lưu sở thích và thông tin cá nhân |
| Sự kiện (Episodic) | File JSONL (append-only) | Vĩnh viễn | Nhật ký sự kiện theo thứ tự thời gian |
| Ngữ nghĩa (Semantic) | ChromaDB | Vĩnh viễn | Tìm kiếm tương tự dựa trên cosine similarity |

### Bộ định tuyến bộ nhớ (Memory Router)

| Ý định (Intent) | Backend | Tín hiệu kích hoạt |
|---|---|---|
| USER_PREFERENCE | Redis | "tôi thích / tôi muốn / sở thích / I like / I prefer" |
| EXPERIENCE_RECALL | Episodic JSON | "lần trước / hôm qua / bạn đã nói / last time / you said" |
| FACTUAL_RECALL | ChromaDB | "là gì / như thế nào / giải thích / what is / how / explain" |
| CURRENT_CONTEXT | Short-term | Tất cả truy vấn còn lại |

### Quản lý cửa sổ ngữ cảnh — Cơ chế loại bỏ 4 cấp độ ưu tiên

| Ưu tiên | Nội dung | Điều kiện loại bỏ |
|---|---|---|
| P1 — Tối quan trọng | System prompt + truy vấn hiện tại | Không bao giờ loại bỏ |
| P2 — Cao | Sở thích và dữ kiện từ Redis | Khi vượt 95% giới hạn token |
| P3 — Trung bình | Bộ nhớ sự kiện/ngữ nghĩa gần đây (< 3 lượt) | Khi vượt 90% giới hạn token |
| P4 — Thấp | Lịch sử hội thoại cũ | Khi vượt 80% giới hạn token |