"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!

Tác giả: Lý Nhật Huy — 2A202601450
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (giống hệt Bước 2 để kết quả so sánh được) ────────
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

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

# Số luồng RAGAS chạy song song — giảm xuống nếu provider bị rate-limit (429)
MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "4"))


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    # contexts phải là list[str] — mỗi phần tử là một chunk riêng biệt
    contexts = [doc.page_content for doc in docs]

    # ghép lại chỉ để đưa vào biến {context} của prompt
    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy {len(QA_PAIRS)} câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        try:
            out = run_rag(retriever, llm, prompt, qa["question"])
        except Exception as e:            # fallback: giữ sample lại, đánh dấu lỗi
            print(f"  ⚠️  Câu {i} lỗi: {e}")
            out = {"answer": "I do not know.", "contexts": [""]}

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],     # list[str] !
        })
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator (judge).
    # Dùng GOOGLE_API_KEY_EVAL (nếu có) để tách quota free-tier khỏi LLM sinh
    # câu trả lời — RAGAS gọi judge ~700 lần nên rất dễ cạn quota theo ngày.
    eval_key = getattr(config, "GOOGLE_API_KEY_EVAL", None)
    llm_eval = get_llm(temperature=0, api_key=eval_key)
    emb_eval = get_embeddings(api_key=eval_key)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=RunConfig(max_workers=MAX_WORKERS, timeout=300, max_retries=10),
        show_progress=True,
    )

    # result["faithfulness"] trả về list of floats (có thể lẫn None/NaN) → lấy mean
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw   = result[key]
        clean = [float(v) for v in raw if v is not None and not np.isnan(float(v))]
        scores[key] = float(np.mean(clean)) if clean else 0.0

    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Phân tích so sánh V1 vs V2 ──────────────────────────────────────────
# Chênh lệch nhỏ hơn ngưỡng này coi như bằng nhau: LLM-judge của RAGAS có tính
# ngẫu nhiên, hơn nhau 0.001 điểm KHÔNG có ý nghĩa thống kê.
SIGNIFICANT_GAP = 0.01


METRIC_ORDER = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def print_comparison(v1_scores: dict, v2_scores: dict):
    """
    In bảng so sánh V1 vs V2.

    Cột Winner chỉ ghi nhận thắng khi chênh lệch ≥ SIGNIFICANT_GAP; nhỏ hơn thì
    ghi "≈ tương đương" vì LLM-judge của RAGAS có sai số ngẫu nhiên.
    """
    print("=" * 68)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 68)
    for metric in METRIC_ORDER:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        gap = s2 - s1
        if gap >= SIGNIFICANT_GAP:
            winner = "← V2"
        elif gap <= -SIGNIFICANT_GAP:
            winner = "← V1"
        else:
            winner = "≈ tương đương"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")
    print("=" * 68)


def analyse(v1_scores: dict, v2_scores: dict) -> str:
    """
    Sinh nhận xét giải thích vì sao V1 hoặc V2 có điểm cao hơn.

    Chỉ tính là "thắng" khi chênh lệch ≥ SIGNIFICANT_GAP, tránh kết luận sai
    từ những khác biệt chỉ là nhiễu của LLM-judge.
    """
    gaps = {m: v2_scores[m] - v1_scores[m] for m in v1_scores}
    v1_wins = [m for m, g in gaps.items() if g <= -SIGNIFICANT_GAP]
    v2_wins = [m for m, g in gaps.items() if g >= SIGNIFICANT_GAP]

    parts = []

    # Nhận xét về nhóm chỉ số retrieval — luôn đúng với thiết kế của lab
    parts.append(
        "context_recall và context_precision gần như bằng nhau ở cả 2 version vì "
        "chúng chỉ phụ thuộc vào retriever (cùng FAISS index, cùng k=3) chứ không "
        "phụ thuộc vào prompt — chênh lệch còn lại chỉ là nhiễu của LLM-judge."
    )

    if v2_wins and not v1_wins:
        parts.insert(0, (
            f"V2 (structured expert) tốt hơn rõ rệt ở: {', '.join(v2_wins)} "
            f"(faithfulness {v1_scores['faithfulness']:.4f} → {v2_scores['faithfulness']:.4f}). "
            "Lý do: prompt V2 bắt mô hình liệt kê fact có trong context trước khi viết, "
            "và cấm suy đoán ngoài context, nên hầu như mọi mệnh đề đều truy vết được "
            "về context → faithfulness cao hơn."
        ))
    elif v1_wins and not v2_wins:
        parts.insert(0, (
            f"V1 (concise) tốt hơn rõ rệt ở: {', '.join(v1_wins)}. "
            "Lý do: prompt V1 giới hạn 2-4 câu nên sinh ra ít mệnh đề hơn, mỗi mệnh đề "
            "đều bám sát context → ít cơ hội 'chế' thông tin."
        ))
    elif v1_wins and v2_wins:
        parts.insert(0, (
            f"Mỗi version mạnh ở mảng khác nhau — V1 hơn ở {', '.join(v1_wins)}, "
            f"V2 hơn ở {', '.join(v2_wins)}."
        ))
    else:
        parts.insert(0, (
            "Hai version cho kết quả tương đương (mọi chênh lệch < "
            f"{SIGNIFICANT_GAP}). Cả hai prompt đều ràng buộc mô hình chỉ dùng context "
            "nên chất lượng grounding như nhau; khác biệt thực tế chỉ là độ dài và "
            "văn phong câu trả lời."
        ))

    return " ".join(parts)


# ── 7. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print()
    print_comparison(v1_scores, v2_scores)

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # ── Phân tích V1 vs V2 ────────────────────────────────────────────────
    analysis = analyse(v1_scores, v2_scores)
    print(f"\n🧠 Phân tích: {analysis}")

    # ── Lưu báo cáo ───────────────────────────────────────────────────────
    report = {
        "student":            "Lý Nhật Huy",
        "student_id":         "2A202601450",
        "provider":           config.PROVIDER,
        "langsmith_project":  config.LANGSMITH_PROJECT,
        "num_qa_pairs":       len(QA_PAIRS),
        "metrics":            ["faithfulness", "answer_relevancy",
                               "context_recall", "context_precision"],
        "prompt_v1_scores":   v1_scores,
        "prompt_v2_scores":   v2_scores,
        "target_met":         best_faith >= 0.8,
        "best_faithfulness":  best_faith,
        "analysis":           analysis,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")

    # Sao chép sang evidence/ để nộp bài
    evidence_path = Path(__file__).parent.parent / "evidence" / "03_ragas_report.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Đã sao chép sang {evidence_path}")


if __name__ == "__main__":
    main()
