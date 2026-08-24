"""
In lại báo cáo RAGAS từ data/ragas_report.json — KHÔNG chạy lại đánh giá.

Dùng để xem nhanh kết quả hoặc tạo ảnh bằng chứng mà không phải chờ 40 phút
đánh giá lại. Bảng so sánh và phần phân tích dùng đúng hàm của
03_ragas_evaluation.py nên nội dung khớp hoàn toàn với lúc chạy thật.

Cách dùng:
    python print_ragas_report.py
"""
import sys
import json
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_HERE   = Path(__file__).parent
_REPORT = _HERE.parent / "data" / "ragas_report.json"


def _load_eval_module():
    """Import 03_ragas_evaluation.py (tên module bắt đầu bằng số nên phải nạp thủ công)."""
    spec = importlib.util.spec_from_file_location(
        "ragas_evaluation", _HERE / "03_ragas_evaluation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if not _REPORT.exists():
        print(f"❌ Chưa có {_REPORT}. Hãy chạy `python 03_ragas_evaluation.py` trước.")
        sys.exit(1)

    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    v1, v2 = report["prompt_v1_scores"], report["prompt_v2_scores"]
    ev     = _load_eval_module()

    print("=" * 68)
    print("  Day 22 — Bước 3: RAGAS Evaluation")
    print(f"  {report['student']} — {report['student_id']}")
    print(f"  {report['num_qa_pairs']} QA pairs × 2 prompt version × "
          f"{len(report['metrics'])} metrics  |  provider: {report['provider']}")
    print("=" * 68)

    for label, scores in (("V1", v1), ("V2", v2)):
        print(f"\n📊 Kết quả RAGAS — Prompt {label}:")
        for k in ev.METRIC_ORDER:
            star = " ⭐" if k == "faithfulness" and scores[k] >= 0.8 else ""
            print(f"  {k:30s}: {scores[k]:.4f}{star}")

    print()
    ev.print_comparison(v1, v2)

    best = report["best_faithfulness"]
    if report["target_met"]:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu: {best:.4f} < 0.8")

    if min(v1["faithfulness"], v2["faithfulness"]) >= 0.9:
        print("✅ Điểm thưởng: faithfulness ≥ 0.9 ở CẢ HAI version")

    # Phân tích — xuống dòng cho vừa bề ngang terminal
    print("\n🧠 Phân tích:")
    line = "   "
    for word in report["analysis"].split():
        if len(line) + len(word) + 1 > 92:
            print(line)
            line = "   "
        line += (" " if line.strip() else "") + word
    if line.strip():
        print(line)


if __name__ == "__main__":
    main()
