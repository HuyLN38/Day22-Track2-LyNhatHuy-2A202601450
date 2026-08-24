"""
Bước 1 — RAG Pipeline với LangSmith Tracing
=============================================
NHIỆM VỤ:
  1. Tải knowledge base, chia chunks, index với FAISS
  2. Xây dựng RAG chain: retriever → prompt → LLM → output parser
  3. Trang trí hàm query với @traceable để LangSmith ghi lại mỗi lần gọi
  4. Chạy 50 câu hỏi → tạo ≥ 50 traces trên LangSmith

DELIVERABLE: Mở https://smith.langchain.com → project của bạn → xác nhận ≥ 50 traces.

Tác giả: Lý Nhật Huy — 2A202601450
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ⚠️ QUAN TRỌNG: Import config TRƯỚC KHI import bất kỳ thư viện LangChain nào.
# config.py tự động đặt LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, ... vào os.environ
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Thiết lập Vectorstore ───────────────────────────────────────────────
def setup_vectorstore():
    """
    Tải knowledge base, chia chunks và tạo FAISS vectorstore.

    Luồng xử lý:
        embeddings → load text → split thành chunks → index vào FAISS
    """
    # Khởi tạo embedding model theo PROVIDER trong .env
    embeddings = get_embeddings()

    # Đọc nội dung knowledge base (data/knowledge_base.txt)
    text = load_knowledge_base()

    # Chia text thành chunks nhỏ để retrieve chính xác hơn
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"📚 Đã chia thành {len(chunks)} chunks")

    # Tạo FAISS index từ các chunks
    vectorstore = build_vectorstore(chunks, embeddings)
    return vectorstore


# ── 2. RAG Prompt Template ─────────────────────────────────────────────────
# Prompt buộc LLM chỉ trả lời dựa trên context được truy xuất (grounded answer),
# nhờ đó điểm faithfulness ở Bước 3 sẽ cao.
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. Answer the question using ONLY the "
        "context below. If the context does not contain the answer, say that "
        "you do not know. Answer in English.\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])


# ── 3. Build RAG Chain ─────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Xây dựng LCEL RAG chain theo cấu trúc pipe:
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()

    Trả về: (chain, retriever)
    """
    llm = get_llm()

    # Retriever lấy 3 chunk gần nhất với câu hỏi
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs) -> str:
        """Ghép page_content của các Document thành một chuỗi context duy nhất."""
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain — mỗi mắt xích đều xuất hiện thành một span riêng trong LangSmith
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ── 4. Hàm Query có LangSmith Tracing ─────────────────────────────────────
def _only_question(inputs: dict) -> dict:
    """Bỏ object `chain` khỏi input của trace — chỉ ghi lại câu hỏi cho gọn."""
    return {"question": inputs.get("question")}


@traceable(name="rag-query", tags=["rag", "step1"], process_inputs=_only_question)
def ask(chain, question: str) -> str:
    """
    Chạy RAG chain với một câu hỏi.
    Decorator @traceable sẽ gửi mỗi lần gọi lên LangSmith như một trace riêng,
    kèm theo input (question), các span con (retriever, prompt, LLM) và output.
    """
    return chain.invoke(question)


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    failures = 0
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        try:
            answer = ask(chain, question)
        except Exception as e:                       # fallback: không dừng cả batch
            failures += 1
            answer = f"[LỖI] {e}"
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {str(answer)[:100]}\n")

    ok = len(SAMPLE_QUESTIONS) - failures
    print(f"\n✅ {ok}/{len(SAMPLE_QUESTIONS)} traces đã gửi lên LangSmith "
          f"project '{config.LANGSMITH_PROJECT}'")
    if failures:
        print(f"⚠️  {failures} câu hỏi bị lỗi — xem log phía trên.")
    print("   Mở https://smith.langchain.com để xem traces.")


if __name__ == "__main__":
    main()
