"""Đọc biểu thức toán học an toàn và thống nhất cho toàn bộ project.

Không dùng ``eval``/``exec``.  Bộ đọc chấp nhận số nguyên, thập phân, phân số,
căn thức và các hằng/hàm thông dụng; kết quả số luôn được kiểm tra là số thực
hữu hạn trước khi chuyển sang ``float``.
"""

from __future__ import annotations

import ast
import math
import re
from fractions import Fraction
from typing import Iterable, Mapping

import sympy as sp


class MathInputError(ValueError):
    """Dữ liệu người dùng không phải biểu thức toán học được hỗ trợ."""


_CONSTANTS: dict[str, sp.Expr] = {
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "tau": 2 * sp.pi,
}

_FUNCTIONS = {
    "sqrt": sp.sqrt,
    "cbrt": lambda x: sp.real_root(x, 3),
    "root": lambda x, n: sp.real_root(x, n),
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "floor": sp.floor,
    "ceil": sp.ceiling,
}

_UNICODE_FRACTIONS = {
    "½": "(1/2)", "⅓": "(1/3)", "⅔": "(2/3)", "¼": "(1/4)",
    "¾": "(3/4)", "⅕": "(1/5)", "⅖": "(2/5)", "⅗": "(3/5)",
    "⅘": "(4/5)", "⅙": "(1/6)", "⅚": "(5/6)", "⅛": "(1/8)",
    "⅜": "(3/8)", "⅝": "(5/8)", "⅞": "(7/8)",
}


def _expand_radicals(text: str) -> str:
    """Đổi √/∛ sang lời gọi hàm, kể cả với ngoặc lồng nhau."""
    output: list[str] = []
    i = 0
    while i < len(text):
        if text[i] not in {"√", "∛"}:
            output.append(text[i])
            i += 1
            continue
        function = "sqrt" if text[i] == "√" else "cbrt"
        i += 1
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            raise MathInputError("Dấu căn phải có biểu thức ở phía sau.")
        if text[i] == "(":
            start, depth = i, 0
            while i < len(text):
                depth += (text[i] == "(") - (text[i] == ")")
                i += 1
                if depth == 0:
                    break
            if depth:
                raise MathInputError("Dấu ngoặc trong căn chưa cân bằng.")
            output.append(function + text[start:i])
            continue
        match = re.match(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+|[A-Za-z_]\w*|π)", text[i:])
        if not match:
            raise MathInputError("Không đọc được biểu thức ngay sau dấu căn.")
        atom = match.group(0)
        output.append(f"{function}({atom})")
        i += len(atom)
    return "".join(output)


def normalize_math_text(text: str) -> str:
    """Chuẩn hóa ký hiệu thường gặp nhưng không làm thay đổi ý nghĩa biểu thức."""
    if not isinstance(text, str) or not text.strip():
        raise MathInputError("Không được để trống dữ liệu.")
    if len(text) > 2000:
        raise MathInputError("Biểu thức quá dài.")
    value = text.strip()
    if re.fullmatch(r"[+-]?(?:\d+,\d*|\d*,\d+)(?:[eE][+-]?\d+)?", value):
        value = value.replace(",", ".")
    for old, new in _UNICODE_FRACTIONS.items():
        value = value.replace(old, new)
    value = value.translate(str.maketrans({
        "−": "-", "–": "-", "—": "-", "×": "*", "·": "*",
        "÷": "/", "∕": "/", "^": "**", "π": "pi",
        "（": "(", "）": ")", "，": ",",
    }))
    value = re.sub(r"\bcăn\s*bậc\s*ba\b", "cbrt", value, flags=re.IGNORECASE)
    value = re.sub(r"\bcan\s*bac\s*ba\b", "cbrt", value, flags=re.IGNORECASE)
    value = re.sub(r"\bcăn\b|\bcan\b", "sqrt", value, flags=re.IGNORECASE)
    value = _expand_radicals(value)
    # Nhân ẩn thông dụng: 2pi, 2x, 2(...), (...)x và )(...).
    # Không chèn dấu nhân vào ký pháp khoa học như 1e-100 hoặc 2E6.
    value = re.sub(r"(?<=[0-9)])(?=(?![eE][+-]?\d)[A-Za-z_(])", "*", value)
    value = re.sub(r"(?<=\))(?=[0-9A-Za-z_(])", "*", value)
    return value


class _SafeMathBuilder(ast.NodeVisitor):
    def __init__(self, symbols: Mapping[str, sp.Symbol]):
        self.symbols = symbols
        self.nodes = 0

    def visit(self, node):  # type: ignore[override]
        self.nodes += 1
        if self.nodes > 300:
            raise MathInputError("Biểu thức quá phức tạp.")
        return super().visit(node)

    def generic_visit(self, node):
        raise MathInputError(f"Cú pháp '{type(node).__name__}' không được hỗ trợ.")

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise MathInputError("Chỉ chấp nhận hằng số dạng số.")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise MathInputError("Không chấp nhận NaN hoặc vô cùng.")
        return sp.Rational(str(node.value))

    def visit_Name(self, node: ast.Name):
        if node.id in self.symbols:
            return self.symbols[node.id]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise MathInputError(f"Ký hiệu '{node.id}' không được hỗ trợ.")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise MathInputError("Phép toán một ngôi không được hỗ trợ.")

    def visit_BinOp(self, node: ast.BinOp):
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise MathInputError("Không được chia cho 0.")
            return left / right
        if isinstance(node.op, ast.Pow):
            if right.is_number and right.is_real and abs(float(right)) > 10000:
                raise MathInputError("Số mũ quá lớn.")
            return left ** right
        raise MathInputError("Phép toán không được hỗ trợ.")

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise MathInputError("Hàm không được hỗ trợ.")
        if node.keywords or not (1 <= len(node.args) <= 2):
            raise MathInputError("Số đối số của hàm không hợp lệ.")
        try:
            return _FUNCTIONS[node.func.id](*(self.visit(arg) for arg in node.args))
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise MathInputError("Đối số của hàm không hợp lệ.") from exc


def parse_math_expression(
    text: str, symbols: Iterable[sp.Symbol] | Mapping[str, sp.Symbol] = ()
) -> sp.Expr:
    """Phân tích biểu thức thành SymPy mà không thực thi mã từ đầu vào."""
    symbol_map = (
        dict(symbols)
        if isinstance(symbols, Mapping)
        else {str(symbol): symbol for symbol in symbols}
    )
    normalized = normalize_math_text(text)
    try:
        tree = ast.parse(normalized, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise MathInputError("Biểu thức không đúng cú pháp.") from exc
    result = sp.sympify(_SafeMathBuilder(symbol_map).visit(tree))
    if result.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise MathInputError("Biểu thức không xác định hoặc không hữu hạn.")
    return result


def parse_real(text: str) -> float:
    """Đọc một biểu thức số thực hữu hạn và trả về ``float``."""
    expression = parse_math_expression(text)
    if expression.free_symbols or expression.is_real is not True:
        raise MathInputError("Giá trị phải là một số thực.")
    try:
        value = float(sp.N(expression, 17))
    except (TypeError, ValueError, OverflowError) as exc:
        raise MathInputError("Không tính được giá trị số thực.") from exc
    if not math.isfinite(value):
        raise MathInputError("Không chấp nhận NaN hoặc vô cùng.")
    return value


def parse_exact(text: str) -> Fraction | sp.Expr:
    """Giữ phân số/căn thức ở dạng chính xác khi thuật toán hỗ trợ đại số đúng."""
    expression = parse_math_expression(text)
    if expression.free_symbols or expression.is_real is not True:
        raise MathInputError("Giá trị phải là một số thực.")
    if expression.is_Rational:
        return Fraction(int(expression.p), int(expression.q))
    return expression


def split_number_row(text: str, expected: int) -> list[str]:
    """Tách một hàng vector/ma trận; biểu thức trong mỗi ô không chứa khoảng trắng."""
    if expected <= 0:
        raise MathInputError("Số phần tử mong đợi phải dương.")
    value = text.strip()
    if value[:1] in "[({" and value[-1:] in "])}":
        value = value[1:-1].strip()

    def by_whitespace(source: str) -> list[str]:
        return source.replace(";", " ").split()

    def by_separators(source: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        for char in source:
            if char in "([{":
                depth += 1
                current.append(char)
            elif char in ")]}":
                depth = max(0, depth - 1)
                current.append(char)
            elif depth == 0 and (char.isspace() or char in ",;"):
                token = "".join(current).strip()
                if token:
                    parts.append(token)
                    current = []
            else:
                current.append(char)
        token = "".join(current).strip()
        if token:
            parts.append(token)
        return parts

    def valid(parts: list[str]) -> bool:
        if len(parts) != expected:
            return False
        try:
            for part in parts:
                parse_real(part)
        except MathInputError:
            return False
        return True

    candidates = [by_whitespace(value), by_separators(value)]
    for parts in candidates:
        if valid(parts):
            return parts
    counts = ", ".join(str(len(parts)) for parts in candidates)
    raise MathInputError(f"Cần đúng {expected} phần tử, đã nhận {counts}.")


def parse_real_row(text: str, expected: int) -> list[float]:
    return [parse_real(item) for item in split_number_row(text, expected)]


def parse_exact_row(text: str, expected: int) -> list[Fraction | sp.Expr]:
    return [parse_exact(item) for item in split_number_row(text, expected)]


SUPPORTED_INPUT_HELP = (
    "Nhận số nguyên, thập phân, phân số và biểu thức: 1/3, sqrt(2), √2, "
    "cbrt(5), ∛5, pi, e, 2*pi, sin(pi/6)."
)
