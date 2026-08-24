# Bằng chứng nộp bài — Day 22: LangSmith + Prompt Versioning

| | |
|---|---|
| **Họ và tên** | Lý Nhật Huy |
| **Mã sinh viên** | 2A202601450 |
| **LLM Provider** | Google Gemini — `gemini-3.5-flash-lite` (LLM) + `gemini-embedding-001` (embeddings) |
| **Vector store** | FAISS, 107 chunks (`chunk_size=500`, `chunk_overlap=50`), retriever `k=3` |
| **LangSmith project** | `day22-lab` |

**URL LangSmith project:**
https://smith.langchain.com/o/6d62fa0e-c91b-4733-abfc-4b151614110d/projects/p/adaaab52-3648-477b-a7bd-f64c4b19acbd

**Prompt Hub:**
- V1 — https://smith.langchain.com/prompts/lynhathuy-rag-prompt-v1
- V2 — https://smith.langchain.com/prompts/lynhathuy-rag-prompt-v2

---

## Danh sách tệp bằng chứng

| Tệp | Nội dung |
|---|---|
| `01_langsmith_traces.png` | Ảnh giao diện LangSmith hiển thị danh sách traces |
| `01_rag_pipeline_log.txt` | Log console Bước 1 — 50/50 câu hỏi chạy qua RAG chain |
| `02_prompt_hub.png` | Ảnh giao diện Prompt Hub hiển thị 2 phiên bản prompt |
| `02_ab_routing_log.txt` | Log console Bước 2 — 50 câu truy vấn, mỗi dòng có nhãn `prompt-v1` / `prompt-v2` |
| `03_ragas_scores.png` | Output terminal bảng so sánh RAGAS V1 vs V2 |
| `03_ragas_scores.txt` | Bản text của ảnh trên |
| `03_ragas_report.json` | Bản sao của `data/ragas_report.json` |
| `03_ragas_run.log` | Log đầy đủ của lần chạy RAGAS (~78 phút, 400 lần đánh giá) |
| `04_pii_demo_log.txt` | Output console 6 test case PII |
| `04_json_demo_log.txt` | Output console 5 test case sửa JSON |
| `00_langsmith_verification.txt` | Kết quả `verify_langsmith.py` — đếm traces qua API, không cần trình duyệt |

Số liệu traces do LangSmith API trả về (xem `00_langsmith_verification.txt`):

```
Bước 1 (tag 'step1') : 50 traces   ✅ đạt ≥ 50
Bước 2 (tag 'step2') : 200 traces  ✅ đạt ≥ 50
Tổng Bước 1 + 2      : 250 traces  ✅ đạt ≥ 100
A/B routing: prompt-v1 = 76 | prompt-v2 = 124
```

Bước 2 có 200 traces vì script được chạy 4 lần (mỗi lần 50 câu) trong quá trình hoàn thiện
log bằng chứng; routing vẫn tất định nên tỉ lệ V1/V2 giữ nguyên 19/31 ở mỗi lần chạy.

---

## Nhiệm vụ 1 — RAG Pipeline với LangSmith

Chạy: `python 01_langsmith_rag_pipeline.py`

- Knowledge base được chia thành **107 chunks** và index vào **FAISS**.
- RAG chain viết bằng **LCEL**: `retriever | format_docs → prompt → LLM → StrOutputParser`.
- Hàm `ask()` gắn `@traceable(name="rag-query", tags=["rag", "step1"])`.
- **50/50 traces** gửi thành công lên project `day22-lab`.

Kiểm chứng nội dung trace (xem `00_langsmith_verification.txt`) — mỗi trace có đủ 3 phần
mà tiêu chí 1.4 yêu cầu:

```
── Trace mẫu: rag-query ──
Input   : {'question': 'What are common AI safety concerns with LLMs?'}
Spans   : 10 span — {'parser': 1, 'llm': 1, 'prompt': 1, 'chain': 6, 'retriever': 1}
✅ Có span retriever: 3 document được truy xuất
✅ Có span LLM (chứa câu trả lời của model)
```

---

## Nhiệm vụ 2 — Prompt Hub & A/B Routing

Chạy: `python 02_prompt_hub_ab_routing.py`

**Hai prompt khác biệt về ngữ nghĩa:**

| | V1 — `lynhathuy-rag-prompt-v1` | V2 — `lynhathuy-rag-prompt-v2` |
|---|---|---|
| Vai trò | "friendly and helpful AI assistant" | "senior domain expert and technical analyst" |
| Quy trình | trả lời thẳng | 3 bước: đọc context → liệt kê fact liên quan → viết câu trả lời |
| Độ dài | 2–4 câu, văn xuôi, không dùng danh sách | 3–5 câu, có tổ chức |
| Ràng buộc | không thêm fact ngoài context | cấm suy đoán, cấm dùng kiến thức ngoài |

**Routing tất định:** `int(md5(request_id).hexdigest(), 16) % 2` → chẵn = V1, lẻ = V2.
Cùng một `request_id` luôn cho cùng một version ở mọi lần chạy — log có phần kiểm chứng
ở cuối (`🔁 Kiểm tra tính tất định`).

**Phân bổ:** V1 = 19 câu, V2 = 31 câu (tổng 50). Không cân bằng 50/50 vì hash MD5 phân bố
ngẫu nhiên, nhưng **tất định** — đó mới là yêu cầu của tiêu chí 2.4.

**Prompt được pull từ Hub khi chạy** (không dùng biến local): log hiển thị
`↓ Đã pull 'lynhathuy-rag-prompt-v1' từ Hub (dùng bản trên Hub khi chạy)`.
Template local chỉ dùng làm fallback khi Hub không truy cập được.

Mỗi trace còn được gắn metadata `request_id` / `prompt_version` / `prompt_name`
và tag `prompt-v1` / `prompt-v2` để lọc trực tiếp trên giao diện LangSmith.

---

## Nhiệm vụ 3 — RAGAS Evaluation

Chạy: `python 03_ragas_evaluation.py` (~78 phút với Gemini free tier)
Xem lại kết quả không cần chạy lại: `python print_ragas_report.py`

50 cặp QA × 2 prompt version × 4 metrics = **400 lần đánh giá**.

| Metric | V1 (concise) | V2 (structured) | Kết luận |
|---|---|---|---|
| **faithfulness** | 0.9653 | **0.9854** | ← V2 |
| answer_relevancy | 0.9712 | 0.9711 | ≈ tương đương |
| context_recall | 0.9800 | 0.9800 | ≈ tương đương |
| context_precision | 0.9000 | 0.8967 | ≈ tương đương |

✅ Mục tiêu faithfulness ≥ 0.8: **đạt ở cả 2 version**
✅ Điểm thưởng faithfulness ≥ 0.9: **đạt ở cả 2 version**

### Phân tích: vì sao V2 cao hơn?

**V2 thắng rõ rệt ở `faithfulness` (0.9653 → 0.9854, +2.0 điểm phần trăm).**
Faithfulness đo tỉ lệ mệnh đề trong câu trả lời có thể truy vết về context được truy xuất.
Prompt V2 bắt mô hình *liệt kê fact có trong context trước khi viết* và *cấm suy đoán hoặc
dùng kiến thức ngoài context* — hai ràng buộc này trực tiếp làm giảm số mệnh đề "trôi" ra
ngoài context. V1 chỉ nói chung chung "không thêm fact ngoài context" nên mô hình đôi khi
bổ sung kiến thức nền của nó vào câu trả lời.

**Ba metric còn lại chênh nhau không đáng kể (< 0.01) và đó là điều được dự đoán trước:**

- `context_recall` và `context_precision` **chỉ phụ thuộc vào retriever**, không phụ thuộc
  vào prompt. Cả 2 version dùng chung một FAISS index và cùng `k=3`, nên context truy xuất
  được là **hoàn toàn giống nhau**. Chênh lệch 0.0033 ở `context_precision` chỉ là nhiễu
  ngẫu nhiên của LLM-judge, không phải khác biệt thật.
- `answer_relevancy` chênh 0.0001 — nhiễu thuần tuý.

Vì vậy code (`analyse()` trong `03_ragas_evaluation.py`) dùng ngưỡng `SIGNIFICANT_GAP = 0.01`:
chênh lệch nhỏ hơn ngưỡng được ghi là "≈ tương đương" thay vì tuyên bố có version thắng —
tránh kết luận sai từ sai số của LLM-judge.

**Kết luận thực tiễn:** nếu ưu tiên độ chính xác/an toàn (không bịa), chọn **V2**. Nếu ưu
tiên câu trả lời ngắn gọn cho người dùng cuối và chấp nhận faithfulness thấp hơn 2%, chọn V1.

---

## Nhiệm vụ 4 — Guardrails AI Validators

Chạy: `python 04_guardrails_validator.py`

### `PIIDetector` (`@register_validator(name="custom/pii-detector")`)

Phát hiện **4 loại PII** bằng regex (không dùng so khớp chuỗi cứng):

| Loại | Kết quả |
|---|---|
| EMAIL | `john.doe@example.com` → `[EMAIL_REDACTED]` |
| PHONE | `(555) 867-5309` → `[PHONE_REDACTED]` |
| SSN | `123-45-6789` → `[SSN_REDACTED]` |
| CREDIT_CARD | `4532 1234 5678 9010` → `[CREDIT_CARD_REDACTED]` |

6 test case: 4 loại PII riêng lẻ + 1 case nhiều PII cùng lúc + 1 case văn bản sạch
(không bị chỉnh sửa).

Hai chi tiết kỹ thuật đáng lưu ý (xem comment trong mã nguồn):

1. **Thứ tự quét** — `SCAN_ORDER` quét CREDIT_CARD và SSN *trước* PHONE, nếu không pattern
   PHONE sẽ ăn mất một phần số thẻ.
2. **Regex PHONE dùng `(?<![\d(])` thay cho `\b`** — `\b` không khớp trước dấu `(`, nên
   `(555) 867-5309` sẽ bị bỏ sót dấu ngoặc mở, kết quả thành `([PHONE_REDACTED]`.

### `JSONFormatter` (`@register_validator(name="custom/json-formatter")`)

Tự sửa **3 loại lỗi**: gỡ markdown fences, đổi nháy đơn → nháy kép, xoá dấu phẩy thừa.
Khi không sửa được thì trả về `FailResult(fix_value=FALLBACK_JSON)` — JSON dự phòng an toàn.

5 test case: JSON hợp lệ / có fences / nháy đơn / dấu phẩy thừa / hoàn toàn sai định dạng.

### `on_fail` đặt đúng chỗ

```python
Guard().use(PIIDetector(on_fail=OnFailAction.FIX))     # ĐÚNG — vào constructor
```

Ngoài ra cần **2 điều kiện nữa** thì `PassResult(value_override=...)` mới thực sự có hiệu lực
trên guardrails-ai 0.11 (đã ghi chú trong mã nguồn):

1. Validator phải khai báo `override_value_on_pass = True`.
2. Phải đặt `GUARDRAILS_RUN_SYNC=true` trước khi `import guardrails` — đường chạy validator
   bất đồng bộ của guardrails 0.11 **bỏ qua** `value_override` của `PassResult`, chỉ đường
   sequential mới áp dụng. Không đặt biến này thì PII vẫn được phát hiện và in log nhưng
   `validated_output` trả về nguyên văn bản gốc chưa che.

---

## Ghi chú kỹ thuật

**Ghim phiên bản LangChain.** `requirements.txt` ghim `langchain*` vào nhánh `0.3.x`.
`pip install "langchain>=0.3.0"` sẽ kéo về LangChain 1.x, mà RAGAS 0.4.x lại import
`langchain_community.chat_models.vertexai` — module đã bị xoá ở `langchain-community` 0.4+
→ `ModuleNotFoundError` ngay khi import RAGAS.

**Xử lý rate limit của free tier.** Gemini free tier giới hạn ~15 request/phút cho LLM và
1000 request/ngày cho embeddings, trong khi toàn bộ lab cần ~900 lần gọi LLM và ~700 lần
gọi embeddings. Các biện pháp đã áp dụng (`src/utils/rate_limit.py`):

- `InMemoryRateLimiter` giữ LLM ở 12 req/phút.
- `ThrottledEmbeddings` chia batch, exponential backoff khi gặp 429, và **cache vector
  xuống đĩa** (`data/.emb_cache/`) — chạy lại Bước 1/2/3 gần như không tốn thêm request.
- `GOOGLE_API_KEY_EVAL` tách LLM đánh giá của RAGAS sang project Gemini thứ 2 để không
  cạn quota ngày giữa chừng.
- `EMBEDDING_API_KEY` cho phép trỏ riêng embeddings sang key khác khi key chính cạn quota.

**Cách tạo `03_ragas_scores.png`.** Ảnh được render từ output thật của
`python print_ragas_report.py` bằng `src/make_evidence_png.py` (Pillow), không phải ảnh
chụp màn hình thủ công. Bản text tương ứng nằm ở `03_ragas_scores.txt` và log gốc đầy đủ
của lần chạy ở `03_ragas_run.log` để đối chiếu.

---

## Cách kiểm tra lại

```bash
pip install -r requirements.txt
cp .env.example .env          # điền LANGCHAIN_API_KEY và GOOGLE_API_KEY

cd src
python config.py              # kiểm tra cấu hình
python run_all.py             # chạy cả 4 bước
python verify_langsmith.py    # đếm traces trên LangSmith qua API
python print_ragas_report.py  # in lại bảng điểm RAGAS
```
