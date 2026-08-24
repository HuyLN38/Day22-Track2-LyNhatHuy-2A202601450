"""
Tiện ích: render log console (file .txt) thành ảnh PNG kiểu cửa sổ terminal.

Dùng để tạo các file bằng chứng dạng ảnh trong evidence/ mà không cần chụp
màn hình thủ công. Nội dung ảnh chính là output thật của script đã chạy.

Cách dùng:
    python make_evidence_png.py <input.txt> <output.png> [--title "..."] [--tail N]

Ví dụ:
    python make_evidence_png.py ../evidence/03_ragas_run.log \\
                                ../evidence/03_ragas_scores.png \\
                                --title "03_ragas_evaluation.py — V1 vs V2"
"""
import sys
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Bảng màu kiểu terminal tối ────────────────────────────────────────────
BG          = (13, 17, 23)
CHROME      = (32, 38, 46)
FG          = (220, 226, 232)
DIM         = (128, 138, 150)
GREEN       = (86, 211, 100)
YELLOW      = (232, 194, 92)
CYAN        = (86, 182, 224)
RED         = (240, 113, 103)
DOTS        = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]

# SF Mono trước tiên: là font monospace duy nhất trên macOS có đủ glyph tiếng Việt
# (Menlo/Monaco thiếu các ký tự như "ế", "ộ", "ứ" → hiện ra ô vuông trống).
FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]


def _load_font(size: int):
    """Tìm font monospace có sẵn trên máy; fallback về font mặc định của PIL."""
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# Font monospace của macOS không có glyph màu cho emoji → thay bằng ký hiệu ASCII
# khi render ảnh. File .txt gốc trong evidence/ vẫn giữ nguyên emoji.
EMOJI_MAP = {
    "✅": "[OK]",  "⚠️": "[!]",  "⚠": "[!]",   "❌": "[X]",
    "⭐": "*",     "🔧": "[FIX]", "⏳": "[wait]", "ℹ️": "[i]", "ℹ": "[i]",
    "📊": "##",    "📐": "##",   "📚": "##",   "🔨": "##",  "🚀": ">>",
    "💾": "[save]", "🧠": "##",  "🔁": "<->",  "📝": "##",  "🔍": "##",
    "🖼️": "##",    "↓": "v",     "←": "<-",    "→": "->",
}


def _strip_emoji(line: str) -> str:
    for emoji, repl in EMOJI_MAP.items():
        line = line.replace(emoji, repl)
    # bỏ mọi ký tự còn lại nằm ngoài BMP (thường là emoji chưa liệt kê)
    return "".join(ch if ord(ch) < 0x2500 or ch in "─━│┌┐└┘├┤┬┴┼█▊▋▌▍▎▏" else "?"
                   for ch in line)


def _colour_for(line: str):
    """Chọn màu chữ theo nội dung dòng log (bắt chước highlight của terminal)."""
    if any(m in line for m in ("[OK]", "*", "PASS", "HOÀN THÀNH")):
        return GREEN
    if any(m in line for m in ("[!]", "[FIX]", "[wait]")):
        return YELLOW
    if any(m in line for m in ("[X]", "LỖI", "FAIL")):
        return RED
    if line.strip().startswith(("=", "─", "━")) or "Metric" in line:
        return CYAN
    if line.strip().startswith(("[", "  [")):
        return DIM
    return FG


def render(lines, out_path: Path, title: str, font_size: int = 15):
    """Vẽ danh sách dòng text thành một ảnh PNG giống cửa sổ terminal."""
    font       = _load_font(font_size)
    title_font = _load_font(font_size - 2)

    pad_x, pad_y  = 18, 14
    bar_h         = 30
    line_h        = font_size + 6

    # đo chiều rộng lớn nhất
    probe = Image.new("RGB", (10, 10))
    d     = ImageDraw.Draw(probe)
    width = max((d.textlength(l, font=font) for l in lines), default=400)
    width = int(max(width, d.textlength(title, font=title_font) + 120)) + pad_x * 2

    height = bar_h + pad_y * 2 + line_h * len(lines)

    img  = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    # thanh tiêu đề cửa sổ
    draw.rectangle([0, 0, width, bar_h], fill=CHROME)
    for i, colour in enumerate(DOTS):
        cx = 16 + i * 18
        draw.ellipse([cx - 6, bar_h // 2 - 6, cx + 6, bar_h // 2 + 6], fill=colour)
    tw = draw.textlength(title, font=title_font)
    draw.text(((width - tw) / 2, bar_h / 2 - font_size / 2 + 1), title,
              font=title_font, fill=DIM)

    # nội dung log
    y = bar_h + pad_y
    for line in lines:
        draw.text((pad_x, y), line, font=font, fill=_colour_for(line))
        y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"🖼️  Đã tạo {out_path}  ({img.width}×{img.height}px, {len(lines)} dòng)")


def clean(raw: str) -> list:
    """Bỏ các dòng nhiễu (progress bar tqdm, log gRPC) và cắt tab thành spaces."""
    out = []
    for line in raw.splitlines():
        if "\r" in line:
            line = line.split("\r")[-1]
        if "it/s]" in line or "s/it]" in line or line.startswith("Evaluating:"):
            continue
        if "ev_poll_posix" in line or "FD from fork parent" in line:
            continue
        out.append(_strip_emoji(line.replace("\t", "    ")).rstrip())
    return out


def main():
    ap = argparse.ArgumentParser(description="Render log console thành ảnh PNG")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--title", default=None, help="Tiêu đề hiển thị trên thanh cửa sổ")
    ap.add_argument("--tail",  type=int, default=None, help="Chỉ lấy N dòng cuối")
    ap.add_argument("--head",  type=int, default=None, help="Chỉ lấy N dòng đầu")
    ap.add_argument("--grep-from", default=None,
                    help="Bắt đầu từ dòng đầu tiên chứa chuỗi này")
    args = ap.parse_args()

    raw   = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    lines = clean(raw)

    if args.grep_from:
        for i, l in enumerate(lines):
            if args.grep_from in l:
                lines = lines[i:]
                break
    if args.head:
        lines = lines[:args.head]
    if args.tail:
        lines = lines[-args.tail:]

    title = args.title or Path(args.input).name
    render(lines, Path(args.output), title)


if __name__ == "__main__":
    main()
