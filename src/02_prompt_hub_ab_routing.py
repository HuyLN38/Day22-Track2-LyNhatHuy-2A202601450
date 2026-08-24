"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com

Tác giả: Lý Nhật Huy — 2A202601450
"""
import sys
import hashlib
import warnings
from pathlib import Path

# client.pull_prompt() phát ra LangChainPendingDeprecationWarning về `allowed_objects`;
# tắt đi để log console (dùng làm bằng chứng) sạch sẽ.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
# Tên phải là duy nhất trong workspace LangSmith của mình.
PROMPT_V1_NAME = "lynhathuy-rag-prompt-v1"
PROMPT_V2_NAME = "lynhathuy-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# V1 — CONCISE: trả lời thẳng vào vấn đề, 2-4 câu, giọng thân thiện.
SYSTEM_V1 = (
    "You are a friendly and helpful AI assistant.\n"
    "Answer the user's question using ONLY the context provided below.\n"
    "Keep your answer short and direct: 2-4 sentences, plain prose, no lists.\n"
    "If the context does not contain the answer, simply say you do not know.\n"
    "Never add facts that are not present in the context. Answer in English.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# V2 — STRUCTURED / EXPERT: đọc kỹ context, trích xuất fact, trả lời có tổ chức.
SYSTEM_V2 = (
    "You are a senior domain expert and technical analyst.\n"
    "Follow this procedure before answering:\n"
    "  1) Read the context carefully and identify every fact relevant to the "
    "question.\n"
    "  2) Compose a well-organised, precise answer of 3-5 sentences that "
    "explains the concept and its purpose.\n"
    "  3) Ground every single statement in the context — do not speculate, do "
    "not use outside knowledge.\n"
    "If the context is insufficient, state explicitly that the information is "
    "not available in the provided context. Answer in English.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def _push_one(client: Client, name: str, template: ChatPromptTemplate,
              description: str, label: str):
    """
    Push một prompt lên Hub.

    Nếu nội dung không đổi so với commit mới nhất, LangSmith trả về HTTP 409
    ("Nothing to commit") — đây KHÔNG phải lỗi: prompt vẫn nằm trên Hub, chỉ là
    không tạo thêm commit trùng lặp. Ta báo cáo trường hợp này là thành công.
    """
    try:
        url = client.push_prompt(name, object=template, description=description)
        print(f"✅ Đã push {label} → {url}")
    except Exception as e:
        if "Nothing to commit" in str(e) or "409" in str(e):
            print(f"✅ {label} đã có trên Hub, nội dung không đổi → giữ nguyên commit "
                  f"hiện tại: https://smith.langchain.com/prompts/{name}")
        else:
            print(f"⚠️  {label} lỗi: {e}")


def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    Mỗi lần nội dung thay đổi sẽ tạo một commit mới của prompt trên Hub.
    """
    _push_one(client, PROMPT_V1_NAME, PROMPT_V1,
              "V1 – Concise style: câu trả lời ngắn gọn 2-4 câu, giọng thân thiện.", "V1")
    _push_one(client, PROMPT_V2_NAME, PROMPT_V2,
              "V2 – Structured expert style: phân tích context, trả lời có cấu trúc 3-5 câu.", "V2")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub (KHÔNG dùng biến local khi Hub sẵn sàng).
    Fallback về template local chỉ khi Hub không khả dụng.

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    # pull_prompt() phát ra PendingDeprecationWarning từ langsmith; tắt cục bộ
    # để log console dùng làm bằng chứng được sạch.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for name, local_fallback in ((PROMPT_V1_NAME, PROMPT_V1),
                                     (PROMPT_V2_NAME, PROMPT_V2)):
            try:
                prompts[name] = client.pull_prompt(name)
                print(f"↓ Đã pull '{name}' từ Hub (dùng bản trên Hub khi chạy)")
            except Exception as e:
                prompts[name] = local_fallback
                print(f"ℹ️  Hub không khả dụng → dùng local fallback cho '{name}' ({e})")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic) — khác hẳn
    random.choice(), nhờ đó A/B test có thể tái lập được.
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
def _ab_inputs(inputs: dict) -> dict:
    """Chỉ ghi câu hỏi + nhãn version vào trace (bỏ retriever/llm/prompt object)."""
    return {"question": inputs.get("question"), "version": inputs.get("version")}


@traceable(name="ab-rag-query", tags=["ab-test", "step2"], process_inputs=_ab_inputs)
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Bước:
      a) Retrieve top-3 docs từ retriever
      b) Ghép page_content thành context string
      c) Chạy (prompt | llm | StrOutputParser())
      d) Trả về {"question", "answer", "version"}
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  context,
        "question": question,
    })

    return {"question": question, "answer": answer, "version": version}


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    # Push → Hub, rồi pull ngược lại từ Hub để chắc chắn runtime dùng bản trên Hub
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count, failures = 0, 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        try:
            # langsmith_extra gắn thêm tag + metadata vào trace → trên LangSmith UI
            # có thể lọc theo prompt-v1 / prompt-v2 và thấy rõ request_id.
            result = ask_ab(
                retriever, llm, prompt, question, version_tag,
                langsmith_extra={
                    "tags": [f"prompt-{version_tag}"],
                    "metadata": {
                        "request_id":     request_id,
                        "prompt_version": version_tag,
                        "prompt_name":    version_key,
                    },
                },
            )
            answer = result["answer"]
        except Exception as e:                       # fallback: bỏ qua câu lỗi
            failures += 1
            answer = f"[LỖI] {e}"

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [{request_id}] [prompt-{version_tag}] {question[:55]}...")
        print(f"     → {str(answer)[:110]}")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    if failures:
        print(f"⚠️  {failures} câu hỏi bị lỗi.")

    # Chứng minh routing là tất định: chạy lại 5 request_id đầu → kết quả không đổi
    print("\n🔁 Kiểm tra tính tất định (chạy lại cùng request_id):")
    for i in range(5):
        rid = f"req-{i:04d}"
        tag = "v1" if get_prompt_version(rid) == PROMPT_V1_NAME else "v2"
        print(f"   {rid} → prompt-{tag}  (luôn giống nhau ở mọi lần chạy)")

    print("\n✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
