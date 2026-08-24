"""
Bước 4 — Guardrails AI Validators
====================================
NHIỆM VỤ:
  1. Xây dựng PIIDetector: phát hiện & redact email, số điện thoại, SSN, số thẻ tín dụng
  2. Xây dựng JSONFormatter: tự động sửa JSON lỗi
  3. Bọc mỗi validator trong Guard và test với các mẫu đầu vào
  4. Chạy demo với 6 trường hợp PII và 5 trường hợp JSON

DELIVERABLE: Tất cả test cases pass (PII bị redact, JSON được sửa thành công)

CÁC KHÁI NIỆM CHÍNH:
  - @register_validator     — khai báo custom validator class
  - Validator.validate()    — implement logic kiểm tra + sửa
  - OnFailAction.FIX        — thay thế output thay vì raise error
  - Guard().use(validator)  — gắn validator instance vào guard
  - guard.validate(text)    → ValidationOutcome
      .validation_passed    — bool
      .validated_output     — output đã được xử lý

⚠️  QUAN TRỌNG: on_fail phải truyền vào CONSTRUCTOR của VALIDATOR, KHÔNG phải Guard.use()
    SAI  : Guard().use(PIIDetector, on_fail=OnFailAction.FIX)   ← TypeError
    ĐÚNG : Guard().use(PIIDetector(on_fail=OnFailAction.FIX))   ← correct

Tác giả: Lý Nhật Huy — 2A202601450
"""

import os
import re
import json

# ⚠️ PHẢI đặt TRƯỚC khi import guardrails.
# Guardrails có 2 đường chạy validator: async và sequential (sync).
# Đường async KHÔNG áp dụng PassResult(value_override=...) vào output cuối cùng,
# nên bắt buộc chạy ở chế độ sync để phần redact PII có hiệu lực.
os.environ.setdefault("GUARDRAILS_RUN_SYNC", "true")

from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult

try:
    from guardrails.hub import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction


# ── 1. PII Detector Validator ──────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Phát hiện và redact Personally Identifiable Information (PII).

    Các pattern được phát hiện (tất cả đều dùng REGEX, không so khớp chuỗi cứng):
      EMAIL       : xxx@xxx.xxx
      PHONE       : (123) 456-7890 hoặc 123-456-7890
      SSN         : 123-45-6789
      CREDIT_CARD : 1234 5678 9012 3456 (hoặc dấu gạch nối)
    """

    # ⚠️ Bắt buộc: Guardrails chỉ áp dụng PassResult(value_override=...)
    # khi validator khai báo override_value_on_pass = True.
    override_value_on_pass = True

    # Regex patterns cho từng loại PII
    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        # Dùng (?<!\d) / (?!\d) thay cho \b: \b không khớp trước dấu "(" nên
        # số dạng "(555) 867-5309" sẽ bị bỏ sót mất dấu ngoặc mở.
        "PHONE":       r"(?<![\d(])(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    # Thứ tự quét: pattern "dài/đặc thù" trước để tránh bị pattern ngắn ăn mất
    # (ví dụ CREDIT_CARD phải quét trước PHONE, SSN trước PHONE).
    SCAN_ORDER = ["EMAIL", "CREDIT_CARD", "SSN", "PHONE"]

    def validate(self, value: str, metadata: dict):
        """
        Tìm PII trong value; nếu phát hiện, redact và trả về PassResult với text đã xử lý.

        Bước:
          1. Copy value → redacted_text
          2. Với mỗi loại PII và pattern tương ứng:
             - Tìm tất cả matches bằng re.findall(pattern, redacted_text)
             - Thay thế từng match bằng "[PII_TYPE_REDACTED]"
             - Ghi lại (pii_type, match) vào found_pii
          3. Nếu found_pii không rỗng → PassResult(value_override=redacted_text)
          4. Nếu không tìm thấy PII → PassResult(value_override=value)
        """
        redacted_text = value
        found_pii     = []

        for pii_type in self.SCAN_ORDER:
            pattern = self.PII_PATTERNS[pii_type]

            # re.finditer trả về match object → lấy nguyên đoạn khớp (group 0),
            # tránh lỗi khi pattern có nhóm bắt (group) như PHONE.
            matches = [m.group(0) for m in re.finditer(pattern, redacted_text)]

            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  ⚠️  Đã redact {len(found_pii)} PII: {[p[0] for p in found_pii]}")
            return PassResult(value_override=redacted_text)

        return PassResult(value_override=value)


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validate và tự động sửa JSON lỗi.

    Các lỗi có thể sửa tự động:
      - Strip markdown code fences (``` hoặc ```json)
      - Thay single quotes → double quotes
      - Xóa trailing commas trước } hoặc ]
      - Re-serialize với json.dumps để định dạng chuẩn

    Nếu không sửa được → trả về FailResult kèm fallback JSON an toàn.
    """

    # ⚠️ Bắt buộc để PassResult(value_override=...) được áp dụng vào output.
    override_value_on_pass = True

    # JSON dự phòng khi không thể sửa được đầu vào
    FALLBACK_JSON = json.dumps(
        {"error": "invalid_json", "message": "Model output could not be parsed as JSON"},
        indent=2,
    )

    @staticmethod
    def _repair(text: str) -> str:
        """
        Cố gắng sửa chuỗi JSON lỗi.

        Bước:
          1. Strip whitespace đầu/cuối
          2. Xóa markdown fences bằng re.sub
          3. Thay single quotes → double quotes
          4. Xóa trailing commas trước } hoặc ]
        """
        text = text.strip()

        # (1) Xóa markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()

        # (2) Thay single quotes → double quotes
        text = text.replace("'", '"')

        # (3) Xóa trailing commas trước } hoặc ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def validate(self, value: str, metadata: dict):
        """
        Thử parse value thành JSON.
        Nếu thất bại, gọi _repair() rồi thử lại.

        Trả về PassResult với JSON được format đẹp nếu thành công.
        Trả về FailResult (kèm fallback JSON) nếu JSON không thể sửa được.
        """
        # (a) Thử parse trực tiếp
        try:
            parsed = json.loads(value)
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            pass

        # (b) Thử sửa rồi parse lại
        try:
            repaired_text = self._repair(value)
            parsed        = json.loads(repaired_text)
            print("  🔧 JSON đã được sửa thành công")
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            print("  ❌ Không sửa được → trả về fallback JSON")
            return FailResult(
                error_message=f"JSON không hợp lệ sau khi sửa: {e}",
                fix_value=self.FALLBACK_JSON,
            )


# ── 3. Demo: PII Guard ─────────────────────────────────────────────────────
def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  Demo: PII Detection & Redaction")
    print("=" * 55)

    # on_fail đi vào CONSTRUCTOR của validator, không phải Guard.use()
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",        "Contact John at john.doe@example.com for details."),
        ("Phone",        "Call our support line at (555) 867-5309."),
        ("SSN",          "Patient SSN is 123-45-6789 on file."),
        ("Credit Card",  "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",    "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",        "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        # In nhãn + input TRƯỚC, để các dòng log của validator nằm đúng chỗ
        print(f"\n[{label}]")
        print(f"  Input:  {text}")

        result = guard.validate(text)

        print(f"  Output: {result.validated_output}")


# ── 4. Demo: JSON Guard ────────────────────────────────────────────────────
def demo_json_guard():
    print("\n" + "=" * 55)
    print("  Demo: JSON Formatting & Repair")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",       '{"name": "Alice", "age": 30}'),
        ("Markdown fences",  '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",    "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",   '{"key": "value",}'),
        ("Truly invalid",    "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        print(f"\n[{label}]")
        print(f"  Input:  {text!r}")

        result = guard.validate(text)
        output = str(result.validated_output)

        if output.strip() == JSONFormatter.FALLBACK_JSON.strip():
            status = "⚠️  Không sửa được → dùng fallback JSON"
        elif result.validation_passed:
            status = "✅ Pass (JSON hợp lệ / đã được sửa)"
        else:
            status = "❌ Fail"

        print(f"  Status: {status}")
        print(f"  Output: {output}")


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Bước 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\n✅ Bước 4 hoàn thành!")


if __name__ == "__main__":
    main()
