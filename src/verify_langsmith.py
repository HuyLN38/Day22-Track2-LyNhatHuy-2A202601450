"""
Kiểm tra bằng chứng trên LangSmith bằng API (không cần mở trình duyệt).

In ra:
  - Tổng số traces trong project, tách theo tag của Bước 1 / Bước 2
  - Số traces theo từng prompt version (prompt-v1 / prompt-v2)
  - Danh sách prompt đang có trên Prompt Hub + URL để chụp màn hình
  - Một trace mẫu, chứng minh trace có đủ: câu hỏi → context → câu trả lời

Cách dùng:
    python verify_langsmith.py
"""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import config
from langsmith import Client


def count_runs(client: Client, project: str) -> tuple:
    """Đếm traces gốc (root run) trong project và thống kê theo tag."""
    roots = list(client.list_runs(project_name=project, is_root=True))
    by_name = Counter(r.name for r in roots)
    by_tag  = Counter(t for r in roots for t in (r.tags or []))
    return roots, by_name, by_tag


def describe_sample(client: Client, roots: list, name: str):
    """In chi tiết 1 trace: input, các span con, và output — chứng minh tiêu chí 1.4."""
    # Ưu tiên trace CÓ output (bỏ qua trace lỗi/đang chạy dở)
    candidates = [r for r in roots if r.name == name]
    sample = next((r for r in candidates
                   if r.outputs and any(v for v in r.outputs.values())), None)
    if sample is None:
        sample = next(iter(candidates), None)
    if sample is None:
        print(f"   (chưa có trace nào tên '{name}')")
        return

    print(f"\n   ── Trace mẫu: {sample.name} ({sample.id}) ──")
    print(f"   Input   : {str(sample.inputs)[:160]}")
    print(f"   Output  : {str(sample.outputs)[:160]}")
    print(f"   Tags    : {sample.tags}")
    print(f"   Metadata: { {k: v for k, v in (sample.extra or {}).get('metadata', {}).items() if not k.startswith('ls_')} }")

    children = list(client.list_runs(project_name=config.LANGSMITH_PROJECT,
                                     trace_id=sample.trace_id))
    kinds = Counter(c.run_type for c in children)
    print(f"   Spans   : {len(children)} span — {dict(kinds)}")

    retr = next((c for c in children if c.run_type == "retriever"), None)
    if retr:
        docs = (retr.outputs or {}).get("documents", [])
        print(f"   ✅ Có span retriever: {len(docs)} document được truy xuất")
        if docs:
            first = docs[0]
            content = first.get("page_content", "") if isinstance(first, dict) else str(first)
            print(f"      Context[0]: {content[:110]}...")
    llm = next((c for c in children if c.run_type == "llm"), None)
    if llm:
        print("   ✅ Có span LLM (chứa câu trả lời của model)")


def main():
    if not config.validate():
        sys.exit(1)

    client  = Client(api_key=config.LANGSMITH_API_KEY)
    project = config.LANGSMITH_PROJECT

    print("=" * 66)
    print(f"  Kiểm tra LangSmith — project '{project}'")
    print("=" * 66)

    roots, by_name, by_tag = count_runs(client, project)

    print(f"\n📊 Tổng số traces (root runs): {len(roots)}")
    print("\n   Theo tên trace:")
    for name, n in by_name.most_common():
        print(f"     {name:22s} : {n}")

    print("\n   Theo tag:")
    for tag, n in by_tag.most_common():
        print(f"     {tag:22s} : {n}")

    step1 = by_tag.get("step1", 0)
    step2 = by_tag.get("step2", 0)
    print(f"\n   Bước 1 (tag 'step1') : {step1} traces  "
          f"{'✅ đạt ≥ 50' if step1 >= 50 else '⚠️ chưa đủ 50'}")
    print(f"   Bước 2 (tag 'step2') : {step2} traces  "
          f"{'✅ đạt ≥ 50' if step2 >= 50 else '⚠️ chưa đủ 50'}")
    print(f"   Tổng Bước 1 + 2      : {step1 + step2} traces  "
          f"{'✅ đạt ≥ 100' if step1 + step2 >= 100 else '⚠️ chưa đủ 100'}")

    v1 = by_tag.get("prompt-v1", 0)
    v2 = by_tag.get("prompt-v2", 0)
    print(f"\n   A/B routing: prompt-v1 = {v1} | prompt-v2 = {v2}  "
          f"{'✅ cả 2 version đều nhận câu hỏi' if v1 and v2 else '⚠️ thiếu 1 version'}")

    # ── Prompt Hub ────────────────────────────────────────────────────────
    # list_prompts() trả về cả prompt public của người khác → chỉ giữ prompt
    # thuộc workspace của mình (is_public=False).
    print("\n📝 Prompt trên Prompt Hub (của workspace này):")
    try:
        prompts = [p for p in client.list_prompts(limit=100, is_public=False).repos]
        for p in prompts:
            print(f"     • {p.repo_handle}  (commits: {p.num_commits}, "
                  f"cập nhật: {p.updated_at:%Y-%m-%d %H:%M})")
            print(f"       https://smith.langchain.com/prompts/{p.repo_handle}")
        print(f"   Tổng: {len(prompts)} prompt "
              f"{'✅ có ≥ 2 version' if len(prompts) >= 2 else '⚠️ cần 2 prompt'}")
    except Exception as e:
        print(f"   ⚠️  Không liệt kê được prompt: {e}")

    # ── Trace mẫu ─────────────────────────────────────────────────────────
    print("\n🔍 Kiểm tra nội dung trace (tiêu chí 1.4):")
    describe_sample(client, roots, "rag-query")
    describe_sample(client, roots, "ab-rag-query")

    # ── URL trực tiếp tới project (dùng để chụp màn hình / nộp bài) ───────
    print("\n" + "=" * 66)
    print("  URL project để chụp màn hình / nộp bài:")
    proj = next((p for p in client.list_projects(limit=100) if p.name == project), None)
    if proj is not None:
        print(f"  https://smith.langchain.com/o/{proj.tenant_id}/projects/p/{proj.id}")
    else:
        print(f"  (mở LangSmith → Tracing Projects → '{project}')")
    print("=" * 66)


if __name__ == "__main__":
    main()
