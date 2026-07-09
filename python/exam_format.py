"""Định dạng toán học Unicode thống nhất cho các bài Giải tích số.

Module chỉ lo trình bày, không thực hiện thay thuật toán.  Các hàm không dùng
màu ANSI hay ký tự trang trí phụ thuộc terminal và luôn khử ``-0``.
"""

from __future__ import annotations

import math
import os
import re
import sys
import builtins
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import TextIO

SUBSCRIPT = str.maketrans("0123456789+-=()ijkn", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ᵢⱼₖₙ")
SUPERSCRIPT = str.maketrans("0123456789+-=()ijkn", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱʲᵏⁿ")


@dataclass(frozen=True)
class DisplayDigits:
    """Số chữ số chỉ dùng khi trình bày, không dùng để làm tròn tính toán."""

    matrix: int = 6
    vector: int = 6
    scalar: int = 6
    error: int = 6

    def __post_init__(self) -> None:
        for name, value in (
            ("matrix", self.matrix),
            ("vector", self.vector),
            ("scalar", self.scalar),
            ("error", self.error),
        ):
            if int(value) < 0:
                raise ValueError(f"Số chữ số {name} phải không âm.")

    @classmethod
    def common(cls, digits: int) -> "DisplayDigits":
        return cls(digits, digits, digits, digits)


def prompt_display_digits(input_func=builtins.input) -> DisplayDigits:
    """Hỏi nhanh số chữ số hiển thị chung hoặc tùy chỉnh riêng từng loại."""

    def read_nonnegative(prompt: str, default: int | None = None) -> int:
        while True:
            suffix = f" [Enter = {default}]" if default is not None else ""
            raw = input_func(prompt + suffix + ": ").strip()
            if raw == "" and default is not None:
                return default
            try:
                value = int(raw)
                if value < 0:
                    raise ValueError
                return value
            except ValueError:
                builtins.print("Lỗi: hãy nhập số nguyên không âm.")

    builtins.print("\nSố chữ số hiển thị:")
    builtins.print("1. Dùng chung một số chữ số")
    builtins.print("2. Tùy chỉnh riêng")
    choice = input_func("Chọn [Enter = 1]: ").strip() or "1"
    if choice != "2":
        return DisplayDigits.common(read_nonnegative("Số chữ số dùng chung", 7))
    return DisplayDigits(
        matrix=read_nonnegative("Số chữ số cho ma trận", 4),
        vector=read_nonnegative("Số chữ số cho vector", 7),
        scalar=read_nonnegative("Số chữ số cho đại lượng vô hướng", 7),
        error=read_nonnegative("Số chữ số cho sai số/residual", 7),
    )


def ensure_utf8() -> None:
    """Ưu tiên UTF-8 cho stdout/stderr, đặc biệt trên Windows.

    ``reconfigure`` không tồn tại ở một số stream giả trong test; khi đó hàm
    giữ nguyên stream thay vì gây lỗi lúc import.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass


def to_subscript(value: object) -> str:
    """Đổi chữ số và các ký hiệu chỉ số thông dụng thành chỉ số dưới."""

    text = str(value)
    unsupported = set(text).difference("0123456789+-=()ijkn")
    if unsupported:
        raise ValueError(f"Không thể đổi {text!r} thành chỉ số dưới.")
    return text.translate(SUBSCRIPT)


def to_superscript(value: object) -> str:
    """Đổi chữ số và các ký hiệu vòng lặp thành chỉ số trên."""

    text = str(value)
    unsupported = set(text).difference("0123456789+-=()ijkn")
    if unsupported:
        raise ValueError(f"Không thể đổi {text!r} thành chỉ số trên.")
    return text.translate(SUPERSCRIPT)


def indexed(symbol: str, index: object) -> str:
    """Trả về ký hiệu có chỉ số dưới, chẳng hạn ``indexed('λ', 1)`` → λ₁."""

    return f"{symbol}{to_subscript(index)}"


def matrix_entry(symbol: str, row: object, column: object) -> str:
    """Trả về phần tử ma trận như aᵢⱼ hoặc a₁₂."""

    return f"{symbol}{to_subscript(row)}{to_subscript(column)}"


def iteration(symbol: str, step: object, index: object | None = None) -> str:
    """Định dạng x⁽ᵏ⁾ hoặc x₁⁽ᵏ⁺¹⁾."""

    base = symbol if index is None else indexed(symbol, index)
    return f"{base}{to_superscript(f'({step})')}"


def transpose(symbol: str) -> str:
    return f"{symbol}ᵀ"


def inverse(symbol: str) -> str:
    return f"{symbol}⁻¹"


def norm(symbol: str, kind: str | int = "∞") -> str:
    """Định dạng chuẩn ‖x‖∞, ‖x‖₁ hoặc ‖x‖₂."""

    text = str(kind)
    suffix = "∞" if text in {"inf", "infinity", "∞"} else to_subscript(text)
    return f"‖{symbol}‖{suffix}"


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def format_number(value: object, decimals: int = 6, *, fraction: bool = True) -> str:
    """Định dạng số gọn, không để ``np.float64`` hay số âm bằng không."""

    if decimals < 0:
        raise ValueError("Số chữ số thập phân phải không âm.")
    if isinstance(value, Fraction) and fraction and value.denominator != 1:
        return f"{value.numerator}/{value.denominator}"
    number = _finite_float(value)
    if number is None:
        text = str(value)
        if text.lower() in {"nan", "inf", "+inf", "-inf", "infinity", "-infinity"}:
            raise ValueError("Không thể trình bày số NaN hoặc vô cực như một kết quả hữu hạn.")
        return text
    threshold = 0.5 * 10.0 ** (-decimals) if decimals else 0.5
    if abs(number) < threshold:
        if number == 0.0:
            return "0"
        return f"{number:.{max(1, decimals)}e}"
    if abs(number) >= 1.0e9:
        return f"{number:.{max(1, decimals)}e}"
    rounded = f"{number:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if rounded in {"", "-0"} else rounded


def matrix_lines(matrix: Sequence[Sequence[object]], decimals: int = 6) -> list[str]:
    """Tạo các dòng ma trận với cột thẳng hàng và ngoặc vuông Unicode."""

    rows = [list(row) for row in matrix]
    if not rows or not rows[0]:
        raise ValueError("Ma trận phải có ít nhất một hàng và một cột.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Các hàng của ma trận phải có cùng số phần tử.")
    cells = [[format_number(value, decimals) for value in row] for row in rows]
    widths = [max(len(row[j]) for row in cells) for j in range(width)]
    result: list[str] = []
    for i, row in enumerate(cells):
        if len(rows) == 1:
            left, right = "[", "]"
        else:
            left, right = ("⎡", "⎤") if i == 0 else (("⎣", "⎦") if i == len(rows) - 1 else ("⎢", "⎥"))
        body = "  ".join(value.rjust(widths[j]) for j, value in enumerate(row))
        result.append(f"{left} {body} {right}")
    return result


def format_matrix(name: str, matrix: Sequence[Sequence[object]], decimals: int = 6) -> str:
    """Định dạng ma trận nhiều dòng, nhãn chỉ xuất hiện ở dòng đầu."""

    lines = matrix_lines(matrix, decimals)
    prefix = f"{name} = "
    padding = " " * len(prefix)
    return "\n".join((prefix if i == 0 else padding) + line for i, line in enumerate(lines))


def format_vector(
    name: str,
    values: Iterable[object],
    decimals: int = 6,
    *,
    column: bool = True,
) -> str:
    """Định dạng vector cột theo mặc định; vector hàng phải được yêu cầu rõ."""

    data = list(values)
    if not data:
        raise ValueError("Vector không được rỗng.")
    matrix = [[value] for value in data] if column else [data]
    return format_matrix(name, matrix, decimals)


def format_matrix_with_digits(
    name: str,
    matrix: Sequence[Sequence[object]],
    digits: DisplayDigits,
) -> str:
    return format_matrix(name, matrix, digits.matrix)


def format_vector_with_digits(
    name: str,
    values: Iterable[object],
    digits: DisplayDigits,
    *,
    column: bool = True,
) -> str:
    return format_vector(name, values, digits.vector, column=column)


def format_scalar(value: object, digits: DisplayDigits) -> str:
    return format_number(value, digits.scalar)


def format_error(value: object, digits: DisplayDigits) -> str:
    return format_number(value, digits.error, fraction=False)


def print_matrix(
    name: str,
    matrix: Sequence[Sequence[object]],
    decimals: int = 6,
    *,
    file: TextIO | None = None,
) -> None:
    print(format_matrix(name, matrix, decimals), file=file or sys.stdout)


def safe_unicode(text: str, encoding: str | None = None) -> str:
    """Fallback ASCII rõ ràng khi một terminal thật sự không mã hóa được Unicode."""

    target = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(target)
        return text
    except (LookupError, UnicodeEncodeError):
        replacements = {
            "⁻¹": "^(-1)", "ᵀ": "^T", "∞": "inf", "≈": "~=",
            "≤": "<=", "≥": ">=", "≠": "!=", "→": "->", "⇒": "=>",
            "λ": "lambda", "σ": "sigma", "φ": "phi", "ε": "epsilon",
            "Δ": "Delta", "Σ": "Sigma", "‖": "|",
        }
        for source, replacement in replacements.items():
            text = text.replace(source, replacement)
        for source, replacement in zip("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ᵢⱼₖₙ", "0123456789+-=()ijkn"):
            text = text.replace(source, f"_{replacement}")
        for source, replacement in zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱʲᵏⁿ", "0123456789+-=()ijkn"):
            text = text.replace(source, f"^{replacement}")
        return text.encode(target, errors="replace").decode(target, errors="replace")


def prettify_math(text: str) -> str:
    """Nâng ký hiệu ASCII cũ trong nội dung trình bày lên Unicode.

    Phép đổi có chủ đích, chỉ nhận các mẫu chỉ số toán học; các nhãn kỹ thuật
    như ``step_difference`` hoặc ``x_abs`` được giữ nguyên.
    """

    replacements = {
        "A^(-1)": "A⁻¹", "A^-1": "A⁻¹", "^T": "ᵀ",
        "lambda_min": "λₘᵢₙ", "lambda_max": "λₘₐₓ",
        "epsilon": "ε", "Delta": "Δ", "sigma": "σ", "lambda": "λ",
        " <= ": " ≤ ", " >= ": " ≥ ", " != ": " ≠ ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = text.replace("<=", "≤").replace(">=", "≥").replace("!=", "≠")
    text = text.replace("~=", "≈")
    text = re.sub(r"\|\|([^|\r\n]+)\|\|_inf\b", r"‖\1‖∞", text)
    text = re.sub(
        r"\|\|([^|\r\n]+)\|\|_([0-9]+)\b",
        lambda m: "‖" + m.group(1) + "‖" + to_subscript(m.group(2)),
        text,
    )
    text = re.sub(r"\b([A-Za-z])\^\(-1\)", lambda m: m.group(1) + "⁻¹", text)
    text = re.sub(r"\b([A-Za-z])\^-1", lambda m: m.group(1) + "⁻¹", text)

    # Trong tài liệu của project, x_(k+1) ký hiệu bước lặp nên là chỉ số trên.
    text = re.sub(
        r"\bx_\(([0-9ijkn+\-=]+)\)",
        lambda m: "x" + to_superscript(f"({m.group(1)})"),
        text,
    )

    def grouped_subscript(match: re.Match[str]) -> str:
        try:
            return match.group(1) + to_subscript(match.group(2))
        except ValueError:
            return match.group(0)

    text = re.sub(r"\b([A-Za-zΑ-ω]+)_\{([0-9ijkn+\-=]+)\}", grouped_subscript, text)
    text = re.sub(r"\b([A-Za-zΑ-ω]+)_\(([0-9ijkn+\-=]+)\)", grouped_subscript, text)
    text = text.replace("^(-1)", "⁻¹")

    subscript_pattern = re.compile(r"\b([A-Za-zΑ-ω]+)_([0-9ijknpqrs+\-=]+)\b")

    def subscript_match(match: re.Match[str]) -> str:
        symbol, index = match.groups()
        greek = {"alpha": "α", "beta": "β", "theta": "θ", "tau": "τ"}.get(symbol, symbol)
        try:
            return greek + to_subscript(index)
        except ValueError:
            return match.group(0)

    text = subscript_pattern.sub(subscript_match, text)

    # Ba mẫu tách biệt là chủ ý: dấu ')' chỉ được lấy khi chính mẫu đã lấy '('.
    # Regex cũ dùng cả hai dấu ngoặc tùy chọn độc lập nên ``u_ki^2)`` bị mất
    # dấu ngoặc đóng của biểu thức Cholesky.
    def parenthesized_power(match: re.Match[str]) -> str:
        try:
            return to_superscript(f"({match.group(1)})")
        except ValueError:
            return match.group(0)

    def plain_power(match: re.Match[str]) -> str:
        try:
            return to_superscript(match.group(1))
        except ValueError:
            return match.group(0)

    text = re.sub(r"\^\(([0-9ijkn+\-=]+)\)", parenthesized_power, text)
    text = re.sub(r"\^\{([0-9ijkn+\-=]+)\}", plain_power, text)
    text = re.sub(r"\^([0-9ijkn+\-=]+)", plain_power, text)
    text = text.replace("||", "‖")
    return text


def exam_print(*values: object, **kwargs: object) -> None:
    """Thay thế tương thích cho ``print`` với ký hiệu toán được chuẩn hóa."""

    converted = tuple(prettify_math(value) if isinstance(value, str) else value for value in values)
    builtins.print(*converted, **kwargs)


if os.name == "nt":
    ensure_utf8()
