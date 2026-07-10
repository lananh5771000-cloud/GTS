"""
PHƯƠNG PHÁP LŨY THỪA VÀ XUỐNG THANG
=====================================

Chương trình được viết theo kiểu trình bày bài thi Giải tích số:
    - in dữ kiện, công thức và điều kiện áp dụng;
    - in đầy đủ Y^(k), hệ số chuẩn hóa, X^(k) và bảng lặp;
    - chuẩn hóa vector riêng cuối cùng theo chuẩn 2;
    - kiểm tra phần dư ||Av - lambda*v||_2;
    - in rõ công thức và ma trận sau mỗi lần xuống thang.

Không dùng NumPy để có thể chạy trên máy chỉ cài Python.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from input_utils import MathInputError, parse_real as parse_input_real, split_number_row

from exam_format import exam_print as print, indexed, iteration, norm, transpose


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")


Matrix = list[list[float]]
Vector = list[float]
LINE = "=" * 100
THIN_LINE = "-" * 100
FIXED_ITERATION_WARNING = (
    "Đề yêu cầu k cố định nên kết quả chỉ là xấp xỉ sau k vòng, chưa chứng nhận đạt ε."
)


# =============================================================================
# NHẬP DỮ LIỆU
# =============================================================================

def parse_real(token: str) -> float:
    """Đọc số nguyên, số thập phân, dạng khoa học hoặc phân số a/b."""
    token = token.strip()
    if "," in token and "." not in token:
        token = token.replace(",", ".")
    value = parse_input_real(token)
    if not math.isfinite(value):
        raise ValueError
    return value


def input_int(prompt: str, minimum: int, default: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            value = int(raw)
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            suffix = f" (Enter = {default})" if default is not None else ""
            print(f"  Lỗi: hãy nhập số nguyên >= {minimum}{suffix}.")


def input_float(prompt: str, default: float | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            value = parse_real(raw)
            if value <= 0.0:
                raise ValueError
            return value
        except (ValueError, ZeroDivisionError):
            print("  Lỗi: hãy nhập số dương, ví dụ 1e-7, 0.0001 hoặc 1/1000.")


def input_row(prompt: str, size: int) -> Vector:
    while True:
        try:
            tokens = split_number_row(input(prompt), size)
            return [parse_real(token) for token in tokens]
        except (MathInputError, ValueError, ZeroDivisionError):
            print("  Lỗi: chỉ nhập số thực hoặc phân số hợp lệ, cách nhau bằng dấu cách.")


def input_matrix(n: int) -> Matrix:
    print(f"\nNhập ma trận vuông A cấp {n} (mỗi dòng gồm {n} số):")
    return [input_row(f"  Dòng {i + 1}: ", n) for i in range(n)]


def input_initial_vector(n: int) -> Vector:
    print("\nChọn vector khởi đầu X^(0):")
    print("  1. Tự nhập")
    print("  2. Dùng X^(0) = (1, 1, ..., 1)^T")
    while True:
        choice = input("Lựa chọn [Enter = 2]: ").strip() or "2"
        if choice == "1":
            vector = input_row(f"Nhập {n} phần tử của X^(0): ", n)
            break
        if choice == "2":
            vector = [1.0] * n
            break
        print("  Lỗi: chỉ chọn 1 hoặc 2.")
    if norm2(vector) == 0.0:
        raise ValueError("Vector khởi đầu không được là vector 0.")
    return vector


# =============================================================================
# PHÉP TOÁN VECTOR - MA TRẬN
# =============================================================================

def copy_matrix(a: Matrix) -> Matrix:
    return [row[:] for row in a]


def mat_vec(a: Matrix, x: Vector) -> Vector:
    return [math.fsum(a[i][j] * x[j] for j in range(len(x))) for i in range(len(a))]


def mat_mat(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("Kich thuoc ma tran khong phu hop de nhan.")
    return [
        [
            math.fsum(a[i][k] * b[k][j] for k in range(len(b)))
            for j in range(len(b[0]))
        ]
        for i in range(len(a))
    ]


def dot(x: Vector, y: Vector) -> float:
    return math.fsum(a * b for a, b in zip(x, y))


def norm2(x: Vector) -> float:
    return math.sqrt(math.fsum(value * value for value in x))


def norm_inf(x: Vector) -> float:
    return max((abs(value) for value in x), default=0.0)


def subtract_vectors(x: Vector, y: Vector) -> Vector:
    return [a - b for a, b in zip(x, y)]


def scale_vector(x: Vector, alpha: float) -> Vector:
    return [alpha * value for value in x]


def normalize_2(x: Vector) -> Vector:
    length = norm2(x)
    if length == 0.0:
        raise ArithmeticError("Không thể chuẩn hóa vector 0.")
    result = [value / length for value in x]
    # Chốt dấu để vector được trình bày nhất quán qua các lần chạy.
    pivot = max(range(len(result)), key=lambda i: abs(result[i]))
    if result[pivot] < 0.0:
        result = [-value for value in result]
    return result


def rayleigh_quotient(a: Matrix, x: Vector) -> float:
    denominator = dot(x, x)
    if denominator == 0.0:
        raise ArithmeticError("Không tính được thương Rayleigh của vector 0.")
    return dot(x, mat_vec(a, x)) / denominator


def residual_vector(a: Matrix, eigenvalue: float, x: Vector) -> Vector:
    return subtract_vectors(mat_vec(a, x), scale_vector(x, eigenvalue))


def residual_norm(a: Matrix, eigenvalue: float, x: Vector) -> float:
    return norm2(residual_vector(a, eigenvalue, x))


def is_symmetric(a: Matrix, tolerance: float = 1e-12) -> bool:
    scale = max((abs(value) for row in a for value in row), default=0.0)
    if scale == 0.0:
        return True
    return all(
        abs(a[i][j] - a[j][i]) <= tolerance * scale
        for i in range(len(a))
        for j in range(i)
    )


def matrix_frobenius_norm(a: Matrix) -> float:
    return math.sqrt(math.fsum(value * value for row in a for value in row))


def relative_residual_norm(a: Matrix, eigenvalue: float, x: Vector) -> float:
    """Residual tuong doi dung mau so max(1, ||A||_2 gan dung bang Frobenius, |lambda|)."""
    return residual_norm(a, eigenvalue, x) / max(1.0, matrix_frobenius_norm(a), abs(eigenvalue))


def transpose_matrix(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)]


def outer_product(x: Vector, y: Vector) -> Matrix:
    return [[xi * yj for yj in y] for xi in x]


def format_complex_reason(value: complex, decimals: int = 6) -> str:
    """Dinh dang tri rieng that/phuc trong phan giai thich khong hoi tu."""
    real = 0.0 if abs(value.real) < 0.5 * 10 ** (-decimals) else value.real
    imag = 0.0 if abs(value.imag) < 0.5 * 10 ** (-decimals) else value.imag
    if imag == 0.0:
        return f"{real:.{decimals}f}"
    if real == 0.0:
        return f"{imag:.{decimals}f}i"
    sign = "+" if imag >= 0.0 else "-"
    return f"{real:.{decimals}f} {sign} {abs(imag):.{decimals}f}i"


def spectral_condition(a: Matrix, tolerance: float = 1e-8) -> SpectralCondition:
    """Nhan dien cac ca phuong phap luy thua co ban can canh bao.

    Ham nay chi dung da thuc dac trung/SymPy de canh bao dieu kien ap dung,
    khong dung lam loi de tra ve vector rieng.
    """
    try:
        import sympy as sp

        matrix = sp.Matrix(a)
        roots = []
        for root, multiplicity in matrix.eigenvals().items():
            roots.extend([complex(sp.N(root, 30))] * int(multiplicity))
        if not roots:
            variable = sp.Symbol("lambda")
            roots = [
                complex(root)
                for root in sp.nroots(matrix.charpoly(variable).as_expr(), n=30, maxsteps=200)
            ]
    except Exception:
        return SpectralCondition(False, False, False, 0.0, 0, "")

    if not roots:
        return SpectralCondition(True, False, False, 0.0, 0, "Ma tran khong co pho huu han de xet.")

    moduli = [abs(root) for root in roots]
    dominant_modulus = max(moduli)
    scale = max(1.0, dominant_modulus)
    on_circle = [
        root for root in roots if abs(abs(root) - dominant_modulus) <= tolerance * scale
    ]
    complex_dominant = any(abs(root.imag) > tolerance * scale for root in on_circle)
    unique = len(on_circle) == 1 and not complex_dominant

    warning = ""
    if complex_dominant:
        warning = (
            "Không tìm được trị riêng thực trội. "
            "Phuong phap luy thua thuc co ban khong phu hop de tach truc tiep "
            "cap tri rieng phuc troi."
        )
    elif not unique:
        warning = (
            "Phuong phap luy thua co ban khong du dieu kien tach tri rieng troi "
            "duy nhat theo modun."
        )
    return SpectralCondition(
        True,
        unique,
        complex_dominant,
        dominant_modulus,
        len(on_circle),
        warning,
        tuple(roots),
        tuple(on_circle),
    )


def explain_power_failure(
    a: Matrix,
    attempts: list[PowerResult],
    epsilon: float,
    maximum_iterations: int,
    condition: SpectralCondition | None = None,
) -> str:
    """Tao loi giai thich ngan gon khi khong chung nhan duoc tri rieng troi hoi tu."""
    condition = condition or spectral_condition(a)
    lines = [
        "Không tìm được trị riêng thực trội hội tụ bằng phương pháp lũy thừa thực."
    ]

    if condition.known:
        if condition.dominant_values:
            values = ", ".join(format_complex_reason(value) for value in condition.dominant_values)
            lines.append(
                f"Phổ của ma trận có {condition.count_on_dominant_circle} trị riêng nằm trên vòng tròn "
                f"|λ| lớn nhất ≈ {condition.dominant_modulus:.6g}: {values}."
            )
        if condition.complex_dominant:
            lines.append(condition.warning)
            lines.append(
                "Các trị riêng trội là phức nên với vector thực, phép lặp nhân B không tiến về một hướng "
                "vector riêng thực cố định; dãy thường quay/dao động và phần dư không giảm về 0."
            )
        elif not condition.unique_dominant:
            lines.append(condition.warning)
            lines.append(
                "Không có một trị riêng trội duy nhất theo môđun, nên phương pháp lũy thừa cơ bản "
                "không có cơ chế tách một hướng trội ổn định; kết quả phụ thuộc vector đầu hoặc có thể dao động."
            )
        else:
            lines.append(
                "Điều kiện phổ có vẻ có trị riêng trội duy nhất, nhưng các lần thử chưa đạt sai số yêu cầu."
            )
            lines.append(
                "Nguyên nhân thường gặp: k_max còn nhỏ, |λ₂/λ₁| gần 1 làm hội tụ rất chậm, "
                "hoặc vector đầu gần trực giao với hướng riêng trội."
            )
    else:
        lines.append(
            "Chương trình không phân tích được phổ bằng SymPy, nên kết luận dựa trên phần dư của các lần lặp."
        )

    if attempts:
        best = min(attempts, key=lambda item: item.relative_residual)
        lines.append(
            f"Đã thử {len(attempts)} vector đầu; phần dư tương đối nhỏ nhất là "
            f"{best.relative_residual:.6e}, vẫn lớn hơn ε = {epsilon:.6e} sau tối đa {maximum_iterations} vòng."
        )
        lines.append(
            f"Giá trị quan sát tốt nhất: λ ≈ {best.eigenvalue:.12g}, "
            f"lý do dừng: {best.stop_reason}."
        )
    elif condition.known and condition.complex_dominant:
        lines.append(
            "Chương trình dừng trước khi thử xuống thang vì phổ đã cho thấy không tồn tại trị riêng thực trội để hội tụ."
        )
    else:
        lines.append("Không có vector đầu nào tạo được quá trình lặp hợp lệ để kiểm tra phần dư.")

    return "\n".join(lines)



# =============================================================================
# ĐỊNH DẠNG KẾT QUẢ
# =============================================================================

def clean_number(value: float, decimals: int) -> float:
    threshold = 0.5 * 10.0 ** (-decimals) if decimals > 0 else 0.5
    return 0.0 if abs(value) < threshold else value


def format_number(value: float, decimals: int) -> str:
    return f"{clean_number(value, decimals):.{decimals}f}"


def matrix_text_lines(a: Matrix, decimals: int) -> list[str]:
    if not a:
        return ["[]"]
    cells = [[format_number(value, decimals) for value in row] for row in a]
    widths = [max(len(cells[i][j]) for i in range(len(cells))) for j in range(len(cells[0]))]
    lines: list[str] = []
    for i, row in enumerate(cells):
        content = "  ".join(value.rjust(widths[j]) for j, value in enumerate(row))
        left, right = ("⎡", "⎤") if i == 0 else (("⎣", "⎦") if i == len(cells) - 1 else ("⎢", "⎥"))
        if len(cells) == 1:
            left, right = "[", "]"
        lines.append(f"{left} {content} {right}")
    return lines


def print_matrix(a: Matrix, name: str, decimals: int) -> None:
    lines = matrix_text_lines(a, decimals)
    middle = len(lines) // 2
    padding = " " * (len(name) + 3)
    for i, line in enumerate(lines):
        prefix = f"{name} = " if i == middle else padding
        print(prefix + line)


def print_vector(x: Vector, name: str, decimals: int, horizontal: bool = False) -> None:
    if horizontal:
        values = "  ".join(format_number(value, decimals) for value in x)
        print(f"{name} = [{values}]^T")
    else:
        print_matrix([[value] for value in x], name, decimals)


def print_section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def format_scientific(value: float) -> str:
    return f"{value:.7e}"


def vector_inline(x: Vector, decimals: int) -> str:
    return "[" + "  ".join(format_number(value, decimals) for value in x) + "]ᵀ"


def print_start_attempt_table(search: DominantSearchResult, decimals: int) -> None:
    print("\nBảng thử vector đầu:")
    headers = ["STT", "X⁽⁰⁾", "λ", "|λ|", "‖r‖ tương đối", "kết luận"]
    rows: list[list[str]] = []
    for index, attempt in enumerate(search.attempts, start=1):
        label = attempt.initial_label
        if attempt.initial_vector is not None and len(label) <= 12:
            label = f"{label} {vector_inline(attempt.initial_vector, min(decimals, 4))}"
        if attempt is search.result:
            conclusion = "chọn" if attempt.converged else "chọn, chưa đạt ε"
        elif attempt.converged and attempt.relative_residual <= search.result.relative_residual:
            conclusion = "đạt ε"
        elif attempt.converged:
            conclusion = "đạt ε"
        else:
            conclusion = "chưa đạt ε"
        rows.append([
            str(index),
            label,
            format_number(attempt.eigenvalue, decimals),
            format_number(abs(attempt.eigenvalue), decimals),
            f"{attempt.relative_residual:.3e}",
            conclusion,
        ])
    if not rows:
        print("  Không có lần thử hợp lệ.")
        return
    widths = [max(len(headers[j]), max(len(row[j]) for row in rows)) for j in range(len(headers))]
    print("  " + " | ".join(headers[j].rjust(widths[j]) for j in range(len(headers))))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print("  " + " | ".join(row[j].rjust(widths[j]) for j in range(len(row))))


def print_initial_vector_check(search: DominantSearchResult) -> None:
    if search.initial_projection is None and not search.initial_nearly_orthogonal:
        return
    print("\nKIỂM TRA VECTOR ĐẦU")
    if search.initial_projection is not None:
        print(
            "|(X⁽⁰⁾)ᵀv₁| / (‖X⁽⁰⁾‖₂‖v₁‖₂) "
            f"≈ {search.initial_projection:.7e}"
        )
        if search.initial_projection < 1e-6:
            print("⇒ X⁽⁰⁾ gần vuông góc với v₁.")
        else:
            print("⇒ Chưa phát hiện X⁽⁰⁾ gần vuông góc với v₁ theo chuẩn Euclid.")


# =============================================================================
# PHƯƠNG PHÁP LŨY THỪA
# =============================================================================

@dataclass
class Iteration:
    index: int
    y: Vector
    alpha: float
    x: Vector
    rayleigh: float
    delta: float
    residual: float


@dataclass
class PowerResult:
    eigenvalue: float
    eigenvector: Vector
    iterations: list[Iteration]
    converged: bool
    residual: float
    relative_residual: float
    stop_reason: str
    initial_vector: Vector | None = None
    initial_label: str = "X⁽⁰⁾"
    warning: str = ""


@dataclass
class DominantSearchResult:
    result: PowerResult
    attempts: list[PowerResult]
    dominant_certified: bool
    warning: str
    initial_projection: float | None = None
    initial_nearly_orthogonal: bool = False


@dataclass
class SpectralCondition:
    known: bool
    unique_dominant: bool
    complex_dominant: bool
    dominant_modulus: float
    count_on_dominant_circle: int
    warning: str
    eigenvalues: tuple[complex, ...] = ()
    dominant_values: tuple[complex, ...] = ()


def component_normalize(y: Vector) -> tuple[float, Vector]:
    """
    Chuẩn hóa giống bài mẫu: chọn phần tử có môđun lớn nhất làm hệ số alpha,
    rồi đặt X_mới = Y/alpha. Giữ cả dấu của alpha để phần tử trụ bằng 1.
    """
    pivot = max(range(len(y)), key=lambda i: abs(y[i]))
    alpha = y[pivot]
    scale = norm_inf(y)
    if alpha == 0.0 or not math.isfinite(scale):
        raise ArithmeticError("A*X bằng vector 0; không thể tiếp tục phương pháp lũy thừa.")
    return alpha, [value / alpha for value in y]


def power_method(
    a: Matrix,
    x0: Vector,
    fixed_iterations: int,
    epsilon: float,
    maximum_iterations: int,
    *,
    norm_kind: str = "inf",
    quotient: str = "rayleigh",
    initial_label: str = "X⁽⁰⁾",
) -> PowerResult:
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("A phải là ma trận vuông khác rỗng.")
    if len(x0) != n:
        raise ValueError("Kích thước vector đầu không phù hợp với A.")
    if fixed_iterations < 0 or maximum_iterations <= 0 or epsilon <= 0:
        raise ValueError("epsilon, max_iter phải dương và số bước cố định không được âm.")
    if any(not math.isfinite(value) for row in a for value in row) or any(
        not math.isfinite(value) for value in x0
    ):
        raise ValueError("A và vector đầu chỉ được chứa số hữu hạn.")
    x = normalize_2(x0)
    records: list[Iteration] = []
    limit = fixed_iterations if fixed_iterations > 0 else maximum_iterations
    converged = False
    stop_reason = "đã thực hiện đủ số vòng lặp đề bài yêu cầu"

    for k in range(1, limit + 1):
        y = mat_vec(a, x)
        if norm_kind == "2":
            alpha = norm2(y)
            if alpha == 0.0:
                raise ArithmeticError("A*X bang vector 0; khong the tiep tuc phuong phap luy thua.")
            x_new = [value / alpha for value in y]
        elif norm_kind == "inf":
            alpha, x_new = component_normalize(y)
        else:
            raise ValueError("norm_kind chi nhan 'inf' hoac '2'.")
        rayleigh = rayleigh_quotient(a, x_new)
        if quotient == "component":
            pivot = max(range(n), key=lambda i: abs(x[i]))
            if abs(x[pivot]) <= sys.float_info.epsilon:
                raise ArithmeticError("Khong duoc chia cho thanh phan bang 0 khi dung ti so thanh phan.")
            rayleigh = y[pivot] / x[pivot]
        elif quotient != "rayleigh":
            raise ValueError("quotient chi nhan 'rayleigh' hoac 'component'.")

        # Hai vector riêng x và -x biểu diễn cùng một hướng.
        direct_change = norm2(subtract_vectors(x_new, x))
        opposite_change = norm2(subtract_vectors(x_new, scale_vector(x, -1.0)))
        delta = min(direct_change, opposite_change)
        # Theo đúng quy ước của bài mẫu, alpha_k là xấp xỉ trị riêng dùng
        # trong kết quả và bước xuống thang. Thương Rayleigh được in thêm
        # như một giá trị đối chiếu, không thay thế alpha_k.
        residual = residual_norm(a, rayleigh, x_new)

        records.append(Iteration(k, y, alpha, x_new, rayleigh, delta, residual))
        x = x_new

        relative_residual = residual / max(1.0, matrix_frobenius_norm(a), abs(rayleigh))
        if fixed_iterations == 0 and relative_residual <= epsilon:
            converged = True
            stop_reason = f"phần dư đã thỏa ||BX - λX||₂ <= ε = {epsilon:.3e}"
            break

    unit_vector = normalize_2(x)
    if not records:
        raise ArithmeticError("Không tạo được bước lặp nào.")
    eigenvalue = rayleigh_quotient(a, unit_vector)
    residual = residual_norm(a, eigenvalue, unit_vector)
    relative_residual = residual / max(1.0, matrix_frobenius_norm(a), abs(eigenvalue))

    if fixed_iterations > 0:
        converged = relative_residual <= epsilon
    elif not converged:
        stop_reason = f"đã đạt số vòng lặp tối đa k_max = {maximum_iterations}"

    return PowerResult(
        eigenvalue=eigenvalue,
        eigenvector=unit_vector,
        iterations=records,
        converged=converged,
        residual=residual,
        relative_residual=relative_residual,
        stop_reason=stop_reason,
        initial_vector=[float(value) for value in x0],
        initial_label=initial_label,
    )


def power_method_pdf_case1(
    a: Matrix,
    x0: Vector,
    *,
    epsilon: float = 1e-10,
    max_iter: int = 5000,
    fixed_iterations: int = 0,
) -> PowerResult:
    """PDF case 1: y=A*x, x=y/||y||_2, Delta=||x-x0||_2, lambda Rayleigh."""
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("A phải là ma trận vuông khác rỗng.")
    if len(x0) != n:
        raise ValueError("Kích thước vector đầu không phù hợp với A.")
    if fixed_iterations < 0 or max_iter <= 0 or epsilon <= 0:
        raise ValueError("epsilon, max_iter phải dương và số bước cố định không được âm.")
    if any(not math.isfinite(value) for row in a for value in row) or any(
        not math.isfinite(value) for value in x0
    ):
        raise ValueError("A và vector đầu chỉ được chứa số hữu hạn.")

    x = normalize_2(x0)
    records: list[Iteration] = []
    limit = fixed_iterations if fixed_iterations > 0 else max_iter
    converged = False
    stop_reason = "đã thực hiện đủ số vòng lặp đề bài yêu cầu"

    for k in range(1, limit + 1):
        y = mat_vec(a, x)
        alpha = norm2(y)
        if alpha == 0.0:
            unit_vector = normalize_2(x)
            return PowerResult(
                eigenvalue=0.0,
                eigenvector=unit_vector,
                iterations=records,
                converged=True,
                residual=0.0,
                relative_residual=0.0,
                stop_reason="A*x = 0 nên lambda = 0 và vector hiện tại là vector riêng.",
                initial_vector=[float(value) for value in x0],
                initial_label="X^(0) PDF",
            )

        x_new = [value / alpha for value in y]
        delta = norm2(subtract_vectors(x_new, x))
        rayleigh = rayleigh_quotient(a, x_new)
        residual = residual_norm(a, rayleigh, x_new)
        records.append(Iteration(k, y, alpha, x_new, rayleigh, delta, residual))
        x = x_new

        if fixed_iterations == 0 and delta <= epsilon:
            converged = True
            stop_reason = f"đã đạt Delta <= epsilon = {epsilon:.3e}"
            break

    unit_vector = normalize_2(x)
    if not records:
        raise ArithmeticError("Không tạo được bước lặp nào.")
    eigenvalue = rayleigh_quotient(a, unit_vector)
    residual = residual_norm(a, eigenvalue, unit_vector)
    relative_residual = residual / max(1.0, matrix_frobenius_norm(a), abs(eigenvalue))
    final_delta = records[-1].delta

    if fixed_iterations > 0:
        converged = final_delta <= epsilon
        if converged:
            stop_reason = f"đã thực hiện đúng k={fixed_iterations} bước và Delta <= epsilon"
        else:
            stop_reason = f"đã thực hiện đúng k={fixed_iterations} bước, chưa xét residual để dừng"
    elif not converged:
        stop_reason = f"đã đạt số vòng lặp tối đa k_max = {max_iter}"

    return PowerResult(
        eigenvalue=eigenvalue,
        eigenvector=unit_vector,
        iterations=records,
        converged=converged,
        residual=residual,
        relative_residual=relative_residual,
        stop_reason=stop_reason,
        initial_vector=[float(value) for value in x0],
        initial_label="X^(0) PDF",
    )


def power_method_pdf_opposite_pair(
    a: Matrix,
    x0: Vector,
    *,
    epsilon: float = 1e-10,
    max_iter: int = 5000,
) -> tuple[float, float, Vector, Vector, PowerResult]:
    """PDF case 2: |lambda1|=|lambda2|, lambda2=-lambda1, dung A^2*x."""
    a2 = mat_mat(a, a)
    result = power_method_pdf_case1(a2, x0, epsilon=epsilon, max_iter=max_iter)
    s = rayleigh_quotient(a2, result.eigenvector)
    if s < 0 and abs(s) <= 100 * sys.float_info.epsilon:
        s = 0.0
    if s < 0:
        raise ArithmeticError("Rayleigh cua A^2 am; khong the lay sqrt thuc.")
    lambda1 = math.sqrt(s)
    if lambda1 == 0.0:
        raise ArithmeticError("lambda1 = 0 nen khong dung duoc cong thuc x2=A*x1/lambda1.")
    x1 = normalize_2(result.eigenvector)
    x2 = [value / lambda1 for value in mat_vec(a, x1)]
    v1 = normalize_2([left + right for left, right in zip(x1, x2)])
    v2 = normalize_2([left - right for left, right in zip(x1, x2)])
    return lambda1, -lambda1, v1, v2, result


def pdf_left_eigenvector(
    a: Matrix,
    eigenvalue: float,
    right_vector: Vector,
    *,
    epsilon: float = 1e-10,
    max_iter: int = 5000,
) -> Vector:
    """Tìm vector riêng trái bằng lũy thừa PDF trên A^T, không tự thử nhiều hướng."""
    transposed = transpose_matrix(a)
    candidate = right_vector if norm2(right_vector) > 0.0 else [1.0] * len(a)
    result = power_method_pdf_case1(
        transposed,
        candidate,
        epsilon=epsilon,
        max_iter=max_iter,
    )
    if abs(result.eigenvalue - eigenvalue) > max(1.0, abs(eigenvalue)) * 1e-6:
        raise ArithmeticError("Vector riêng trái chưa khớp trị riêng dùng để xuống thang.")
    return normalize_2(result.eigenvector)


def canonical_start_vectors(
    n: int,
    x0: Vector | None,
    *,
    check_mode: str = "fast",
    max_start_vectors: int | None = None,
) -> list[tuple[str, Vector]]:
    """Tao cac huong thu xac dinh de tranh X^(0) xau hoac gan truc giao."""
    if check_mode not in {"fast", "strict"}:
        raise ValueError("check_mode chi nhan 'fast' hoac 'strict'.")
    if max_start_vectors is None:
        # Mac dinh van thu du cac huong co ban cho bai thi nho, nhung chan lai
        # de khong bien moi lan chay thanh mot phep quet qua nang.
        max_start_vectors = 80 if check_mode == "strict" else min(max(2 * n + 4, n + 4), 40)
    max_start_vectors = max(1, max_start_vectors)

    candidates: list[tuple[str, Vector]] = []
    seen: set[tuple[float, ...]] = set()

    def key_of(vector: Vector) -> tuple[float, ...] | None:
        length = norm2(vector)
        if length == 0.0:
            return None
        normalized = [value / length for value in vector]
        pivot = max(range(n), key=lambda i: abs(normalized[i]))
        if normalized[pivot] < 0.0:
            normalized = [-value for value in normalized]
        return tuple(round(value, 12) for value in normalized)

    def add(label: str, vector: Vector, *, essential: bool = False) -> None:
        if len(vector) != n or norm2(vector) == 0.0:
            return
        if len(candidates) >= max_start_vectors:
            return
        key = key_of(vector)
        if key is None or key in seen:
            return
        seen.add(key)
        candidates.append((label, [float(value) for value in vector]))

    if x0 is not None:
        add("X⁽⁰⁾ nhập", x0, essential=True)
    for j in range(n):
        add(f"e{j + 1}", [1.0 if i == j else 0.0 for i in range(n)], essential=True)
    add("(1,1,...,1)ᵀ", [1.0] * n, essential=True)
    add("(1,2,...,n)ᵀ", [float(i + 1) for i in range(n)], essential=True)
    add("(1,-2,3,-4,...)ᵀ", [(-1.0 if i % 2 else 1.0) * (i + 1) for i in range(n)], essential=True)

    if x0 is not None:
        deltas = [1e-6] if check_mode == "fast" else [1e-3, 1e-6, 1e-9]
        for delta in deltas:
            for j in range(n):
                perturbed = [float(value) for value in x0]
                perturbed[j] += delta
                add(f"X⁽⁰⁾ + {delta:.0e}e{j + 1}", perturbed)

    return candidates


def dominant_eigenpair(
    a: Matrix,
    x0: Vector | None = None,
    *,
    epsilon: float = 1e-10,
    max_iter: int = 5000,
    fixed_iterations: int = 0,
    unique_dominant_verified: bool = False,
    check_mode: str = "fast",
    max_start_vectors: int | None = None,
    use_spectral_check: bool = True,
) -> DominantSearchResult:
    """Thử nhiều hướng xác định để tránh bỏ sót không gian riêng trội."""
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("A phải là ma trận vuông khác rỗng.")
    condition = (
        spectral_condition(a)
        if use_spectral_check and n <= 6
        else SpectralCondition(False, False, False, 0.0, 0, "")
    )
    if condition.known and condition.complex_dominant and fixed_iterations == 0:
        raise ArithmeticError(explain_power_failure(a, [], epsilon, max_iter, condition))
    candidates = canonical_start_vectors(
        n,
        x0,
        check_mode=check_mode,
        max_start_vectors=max_start_vectors,
    )

    attempts: list[PowerResult] = []
    for label, candidate in candidates:
        try:
            result = power_method(
                a,
                candidate,
                fixed_iterations,
                epsilon,
                max_iter,
                initial_label=label,
            )
        except ArithmeticError:
            continue
        if math.isfinite(result.eigenvalue) and math.isfinite(result.residual):
            attempts.append(result)
    if not attempts:
        # Ma trận 0: mọi vector khác 0 là vector riêng ứng với lambda=0.
        if matrix_frobenius_norm(a) == 0.0:
            vector = [1.0] + [0.0] * (n - 1)
            zero = PowerResult(0.0, vector, [], True, 0.0, 0.0, "A là ma trận 0")
            return DominantSearchResult(zero, [zero], unique_dominant_verified, "Trị riêng trội không duy nhất.")
        raise ArithmeticError("Không tìm được hướng khởi đầu hợp lệ.")

    if fixed_iterations > 0:
        selectable = attempts
    else:
        selectable = [
            item for item in attempts
            if item.converged and item.relative_residual <= epsilon
        ]
        if not selectable:
            raise ArithmeticError(explain_power_failure(a, attempts, epsilon, max_iter, condition))

    largest_modulus = max(abs(item.eigenvalue) for item in selectable)
    modulus_tolerance = 1e-8 * max(1.0, largest_modulus)
    near_largest = [
        item for item in selectable
        if abs(abs(item.eigenvalue) - largest_modulus) <= modulus_tolerance
    ]
    best = min(
        near_largest,
        key=lambda item: (item.relative_residual, len(item.iterations)),
    )
    values = [item.eigenvalue for item in selectable]
    spread = max((abs(abs(v) - abs(best.eigenvalue)) for v in values), default=0.0)
    initial_first_misses_best = False
    if x0 is not None and attempts:
        first = attempts[0]
        initial_first_misses_best = (
            abs(abs(first.eigenvalue) - abs(best.eigenvalue))
            > 1e-8 * max(1.0, abs(best.eigenvalue))
        )
        if fixed_iterations == 0:
            initial_first_misses_best = (
                initial_first_misses_best
                or (not first.converged)
                or first.relative_residual > epsilon
            )
    dominant_certified = unique_dominant_verified or (
        condition.known
        and condition.unique_dominant
        and not condition.complex_dominant
        and not initial_first_misses_best
    )
    warning_parts: list[str] = []
    if condition.known and condition.warning:
        warning_parts.append(condition.warning)
    elif not dominant_certified:
        warning_parts.append(
            "Đã thử nhiều vector đầu và chọn trị riêng có môđun lớn nhất quan sát được; "
            "điều này không thay thế chứng minh trị riêng trội duy nhất."
        )
    if spread > max(epsilon * max(abs(best.eigenvalue), sys.float_info.min), 1e-8 * abs(best.eigenvalue)):
        if fixed_iterations > 0:
            warning_parts.append("Các vector đầu cho những trị riêng xấp xỉ khác nhau sau k vòng.")
        else:
            warning_parts.append("Các vector đầu đã hội tụ tới những trị riêng khác nhau.")

    if fixed_iterations > 0 and not best.converged:
        warning_parts.append(FIXED_ITERATION_WARNING)

    initial_projection: float | None = None
    initial_nearly_orthogonal = False
    if x0 is not None and attempts:
        first = attempts[0]
        first_misses_best = initial_first_misses_best
        if first_misses_best:
            if fixed_iterations > 0:
                warning_parts.append(
                    "Lặp từ X⁽⁰⁾ không cho |λ| lớn nhất trong các hướng thử sau k vòng; "
                    "chương trình chọn nghiệm có |λ| lớn nhất quan sát được."
                )
            else:
                warning_parts.append(
                    "X⁽⁰⁾ gần vuông góc với hướng riêng trội ⇒ lặp từ X⁽⁰⁾ có thể bỏ sót λ trội. "
                    "Đã thử thêm các hướng eᵢ và chọn trị riêng có |λ| lớn nhất đạt sai số. "
                    "Vector dau ban dau khong phu hop."
                )
        if is_symmetric(a):
            denominator = norm2(x0) * norm2(best.eigenvector)
            if denominator > 0.0:
                initial_projection = abs(dot(x0, best.eigenvector)) / denominator
                if initial_projection < 1e-6:
                    initial_nearly_orthogonal = True
                    warning_parts.append(
                        "(X⁽⁰⁾, v*) ≈ 0 ⇒ X⁽⁰⁾ gần vuông góc với vector riêng trội."
                    )
        elif first_misses_best and fixed_iterations == 0:
            warning_parts.append(
                "Ma trận không đối xứng nên không kết luận chắc bằng tích vô hướng Euclid; "
                "cảnh báo trên dựa vào thực nghiệm nhiều vector đầu."
            )

    warning = " ".join(part.strip() for part in warning_parts if part.strip()).strip()
    best.warning = warning
    return DominantSearchResult(
        best,
        attempts,
        dominant_certified,
        warning,
        initial_projection,
        initial_nearly_orthogonal,
    )


def print_power_theory() -> None:
    print_section("A. THUẬT TOÁN PHƯƠNG PHÁP LŨY THỪA")
    print("Input:")
    print("  • Ma trận vuông B cấp n có một trị riêng trội về môđun.")
    print("  • Vector khởi đầu X⁽⁰⁾ khác 0 và có thành phần theo vector riêng trội.")
    print("  • Số vòng lặp k, hoặc sai số ε và số vòng lặp tối đa kₘₐₓ.")
    print("Output:")
    print("  • Trị riêng trội gần đúng λ và vector riêng v được đưa về chuẩn 2 bằng 1.")
    print("Các bước:")
    print("  B1. Tính Y⁽ᵏ⁾ = BX⁽ᵏ⁻¹⁾.")
    print("  B2. Chọn p sao cho |yₚ⁽ᵏ⁾| = maxᵢ|yᵢ⁽ᵏ⁾| và đặt αₖ = yₚ⁽ᵏ⁾.")
    print("  B3. Chuẩn hóa X⁽ᵏ⁾ = Y⁽ᵏ⁾/αₖ.")
    print("  B4. αₖ chỉ là hệ số chuẩn hóa, không dùng làm trị riêng cuối.")
    print("  B5. Ước lượng trị riêng bằng thương Rayleigh:")
    print("                λᴿ⁽ᵏ⁾ = (X⁽ᵏ⁾)ᵀBX⁽ᵏ⁾ / (X⁽ᵏ⁾)ᵀX⁽ᵏ⁾.")
    print("  B6. Kiểm tra phần dư r⁽ᵏ⁾ = BX⁽ᵏ⁾ − λᴿ⁽ᵏ⁾X⁽ᵏ⁾.")
    print("  B7. Cuối cùng đặt v = X⁽ᵏ⁾/‖X⁽ᵏ⁾‖₂ để ‖v‖₂ = 1.")


def print_iteration_details(record: Iteration, decimals: int) -> None:
    print(f"\nVòng lặp {record.index}:")
    yk = iteration("Y", record.index)
    xk = iteration("X", record.index)
    x_previous = iteration("X", record.index - 1)
    alpha = indexed("α", record.index)
    rayleigh = f"λᴿ{iteration('', record.index)}"
    print_vector(record.y, f"{yk} = B{x_previous}", decimals, horizontal=True)
    print(
        f"  {alpha} = phần tử của {yk} có môđun lớn nhất"
        f" = {format_number(record.alpha, decimals)}"
    )
    print_vector(record.x, f"{xk} = {yk}/{alpha}", decimals, horizontal=True)
    print(f"  {rayleigh} (thương Rayleigh) = {format_number(record.rayleigh, decimals)}")
    print(f"  Δ = ‖{xk} − {x_previous}‖₂ = {format_scientific(record.delta)}")
    print(f"  ‖B{xk} − {rayleigh}{xk}‖₂ = {format_scientific(record.residual)}")


def print_iteration_table(records: list[Iteration], decimals: int) -> None:
    print("\nBảng tổng hợp các lần lặp:")
    component_count = len(records[0].x)
    headers = ["k", "αₖ", "λᴿ⁽ᵏ⁾", "‖r‖₂"] + [iteration("x", "k", i + 1) for i in range(component_count)]
    rows: list[list[str]] = []
    for record in records:
        rows.append([
            str(record.index),
            format_number(record.alpha, decimals),
            format_number(record.rayleigh, decimals),
            f"{record.residual:.2e}",
            *[format_number(value, decimals) for value in record.x],
        ])
    widths = [max(len(headers[j]), max(len(row[j]) for row in rows)) for j in range(len(headers))]
    print("  " + " | ".join(headers[j].rjust(widths[j]) for j in range(len(headers))))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print("  " + " | ".join(row[j].rjust(widths[j]) for j in range(len(row))))


def print_power_result(
    result: PowerResult,
    stage: int,
    decimals: int,
    *,
    epsilon: float | None = None,
    fixed_iterations: int = 0,
    matrix_label: str = "B",
) -> None:
    for record in result.iterations:
        print_iteration_details(record, decimals)
    print_iteration_table(result.iterations, decimals)

    print("\nKết quả của giai đoạn:")
    print(f"  Dừng vì {result.stop_reason}.")
    eigenvalue_name, vector_name = indexed("λ", stage), indexed("v", stage)
    print(f"  Trị riêng trội gần đúng: {eigenvalue_name} = {format_number(result.eigenvalue, decimals)}")
    print_vector(result.eigenvector, vector_name, decimals)
    print(f"  {norm(vector_name, 2)} = {norm2(result.eigenvector):.{decimals}f}")
    print(
        f"  Sai số phần dư tuyệt đối ‖{matrix_label}{vector_name} − "
        f"{eigenvalue_name}{vector_name}‖₂ = {format_scientific(result.residual)}"
    )
    print(f"  Sai số phần dư tương đối = {format_scientific(result.relative_residual)}")
    if result.converged:
        print("  Đánh giá: kết quả đạt sai số ε đã nhập.")
    else:
        if fixed_iterations > 0:
            print("  Đánh giá: kết quả CHƯA đạt ε; không dùng để xuống thang tự động.")
        else:
            print("  Lưu ý: kết quả CHƯA đạt ε; nếu đề không cố định k, cần tăng số vòng lặp.")

    if fixed_iterations > 0:
        epsilon_text = "" if epsilon is None else f" với ε = {epsilon:.3e}"
        print("\nKết luận với k cố định:")
        print(f"  Sau {len(result.iterations)} vòng:")
        print(f"  {eigenvalue_name} ≈ {format_number(result.eigenvalue, decimals)}")
        print(f"  {vector_name} ≈ {vector_inline(result.eigenvector, decimals)}")
        print(
            f"  ‖{matrix_label}{vector_name} − {eigenvalue_name}{vector_name}‖₂ / (...) "
            f"= {format_scientific(result.relative_residual)}"
        )
        if result.converged:
            print(f"  Kết luận: phần dư đã đạt ε{epsilon_text}; có thể dùng kết quả để xuống thang.")
        else:
            print(
                "  Kết luận: kết quả là xấp xỉ sau k vòng, "
                "chưa đủ điều kiện dùng để xuống thang chắc chắn."
            )


def print_iteration_table_exam(records: list[Iteration], decimals: int) -> None:
    print("\nBẢNG LẶP")
    headers = ["k", "αₖ", "Δₖ", "λ⁽ᵏ⁾", "X⁽ᵏ⁾", "‖r⁽ᵏ⁾‖₂"]
    if not records:
        print("  Không có bước lặp.")
        return
    rows: list[list[str]] = []
    for record in records:
        rows.append([
            str(record.index),
            format_number(record.alpha, decimals),
            f"{record.delta:.2e}",
            format_number(record.rayleigh, decimals),
            vector_inline(record.x, decimals),
            f"{record.residual:.3e}",
        ])
    widths = [max(len(headers[j]), max(len(row[j]) for row in rows)) for j in range(len(headers))]
    print("  " + " | ".join(headers[j].rjust(widths[j]) for j in range(len(headers))))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print("  " + " | ".join(row[j].rjust(widths[j]) for j in range(len(row))))


def print_power_result_exam(
    result: PowerResult,
    stage: int,
    decimals: int,
    *,
    matrix_label: str = "A",
) -> None:
    print_iteration_table_exam(result.iterations, decimals)

    print(f"\nKẾT QUẢ GIAI ĐOẠN {stage}")
    eigenvalue_name, vector_name = indexed("λ", stage), indexed("v", stage)
    print(f"{eigenvalue_name} ≈ {format_number(result.eigenvalue, decimals)}")
    print(f"{vector_name} ≈ {vector_inline(result.eigenvector, decimals)}")
    if result.iterations:
        print(f"Δ cuối = {format_scientific(result.iterations[-1].delta)}")
    print(
        f"‖{matrix_label}{vector_name} − {eigenvalue_name}{vector_name}‖₂ "
        f"= {format_scientific(result.residual)}"
    )
    print(f"Sai số tương đối (chỉ để kiểm tra) = {format_scientific(result.relative_residual)}")
    if result.converged:
        print(f"Kết luận: nhận {eigenvalue_name}, {vector_name}.")
    else:
        print("Kết luận: chưa đạt Δ ≤ ε, không dùng để xuống thang chắc chắn.")


# =============================================================================
# PHƯƠNG PHÁP XUỐNG THANG
# =============================================================================

@dataclass
class DeflationResult:
    matrix: Matrix
    method: str
    pivot: int | None
    annihilation_error: float
    right_vector: Vector | None = None
    left_vector: Vector | None = None
    left_right_dot: float | None = None


def gaussian_solve_vector(matrix: Matrix, rhs: Vector) -> Vector:
    """Giải hệ vuông bằng Gauss pivot từng phần, dùng cho lặp ngược."""
    n = len(matrix)
    augmented = [matrix[i][:] + [rhs[i]] for i in range(n)]
    scale = max((abs(value) for row in matrix for value in row), default=0.0)
    tolerance = 100.0 * sys.float_info.epsilon * max(scale, sys.float_info.min)
    for column in range(n):
        pivot = max(range(column, n), key=lambda i: abs(augmented[i][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            raise ArithmeticError("Hệ lặp ngược suy biến số tại pivot.")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(column + 1, n):
            factor = augmented[row][column] / augmented[column][column]
            augmented[row][column] = 0.0
            for j in range(column + 1, n + 1):
                augmented[row][j] -= factor * augmented[column][j]
    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = math.fsum(augmented[row][j] * solution[j] for j in range(row + 1, n))
        solution[row] = (augmented[row][n] - total) / augmented[row][row]
    if any(not math.isfinite(value) for value in solution):
        raise ArithmeticError("Lặp ngược sinh NaN hoặc vô cùng.")
    return solution


def recover_eigenvector_on_original(
    original: Matrix,
    eigenvalue: float,
    initial: Vector | None = None,
    *,
    epsilon: float = 1e-10,
    max_iter: int = 200,
) -> tuple[float, Vector, float, float]:
    """Lặp ngược có dịch chuyển để tìm lại vector riêng trên ma trận gốc."""
    n = len(original)
    scale = matrix_frobenius_norm(original) + abs(eigenvalue)
    shift_distance = max(math.sqrt(sys.float_info.epsilon) * scale, sys.float_info.min)
    shift = eigenvalue + shift_distance
    shifted = [
        [original[i][j] - (shift if i == j else 0.0) for j in range(n)]
        for i in range(n)
    ]
    candidates: list[Vector] = []
    if initial is not None and len(initial) == n and norm2(initial) > 0.0:
        candidates.append(initial[:])
    candidates.extend([[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)])
    candidates.append([1.0] * n)

    best = None
    for candidate in candidates:
        x = normalize_2(candidate)
        for _ in range(max_iter):
            try:
                y = gaussian_solve_vector(shifted, x)
                x = normalize_2(y)
            except ArithmeticError:
                break
            value = rayleigh_quotient(original, x)
            residual = residual_norm(original, value, x)
            denominator = (
                matrix_frobenius_norm(original) * norm2(x)
                + abs(value) * norm2(x)
                + sys.float_info.min
            )
            relative = residual / denominator
            if best is None or relative < best[3]:
                best = (value, x[:], residual, relative)
            if relative <= epsilon:
                return value, x, residual, relative
    if best is None or best[3] > epsilon:
        raise ArithmeticError(
            "Không khôi phục được vector riêng đạt phần dư trên ma trận gốc."
        )
    return best


@dataclass
class EigenpairCollection:
    eigenvalues: Vector
    eigenvectors: list[Vector]
    relative_residuals: Vector
    success: bool
    reason: str


def eigenpairs_with_deflation(
    a: Matrix,
    wanted: int,
    x0: Vector | None = None,
    *,
    epsilon: float = 1e-10,
    max_iter: int = 5000,
) -> EigenpairCollection:
    """Tìm tuần tự và chỉ ghi nhận cặp đã kiểm tra trên ma trận gốc."""
    n = len(a)
    if wanted <= 0 or wanted > n:
        raise ValueError("Số cặp trị riêng cần tìm phải thuộc 1..n.")
    original = copy_matrix(a)
    current = copy_matrix(a)
    start = x0[:] if x0 is not None and norm2(x0) > 0.0 else [1.0] * n
    values: Vector = []
    vectors: list[Vector] = []
    relatives: Vector = []
    for stage in range(wanted):
        try:
            search = dominant_eigenpair(
                current, start, epsilon=epsilon, max_iter=max_iter
            )
        except ArithmeticError as error:
            return EigenpairCollection(values, vectors, relatives, False, str(error))
        current_pair = search.result
        value, vector = current_pair.eigenvalue, current_pair.eigenvector
        residual = residual_norm(original, value, vector)
        denominator = (
            matrix_frobenius_norm(original) * norm2(vector)
            + abs(value) * norm2(vector)
            + sys.float_info.min
        )
        relative = residual / denominator
        if not is_symmetric(original) or relative > epsilon:
            try:
                value, vector, _residual, relative = recover_eigenvector_on_original(
                    original,
                    current_pair.eigenvalue,
                    current_pair.eigenvector,
                    epsilon=epsilon,
                    max_iter=max_iter,
                )
            except ArithmeticError as error:
                return EigenpairCollection(values, vectors, relatives, False, str(error))
        if relative > epsilon:
            return EigenpairCollection(
                values, vectors, relatives, False,
                "Vector riêng không đạt phần dư trên ma trận gốc.",
            )
        values.append(value)
        vectors.append(vector)
        relatives.append(relative)
        if stage + 1 < wanted:
            current = deflate(
                current, current_pair.eigenvalue, current_pair.eigenvector
            ).matrix
            start = [float(i + 1) for i in range(n)]
    return EigenpairCollection(values, vectors, relatives, True, "Mọi cặp đã đạt phần dư trên A gốc.")


def _legacy_wielandt_deflate(a: Matrix, eigenvalue: float, eigenvector: Vector) -> DeflationResult:
    n = len(a)
    if is_symmetric(a):
        # Hotelling/Wielandt cho ma trận đối xứng và ||v||_2 = 1.
        result = [
            [a[i][j] - eigenvalue * eigenvector[i] * eigenvector[j] for j in range(n)]
            for i in range(n)
        ]
        # Khử sai lệch bất đối xứng chỉ do làm tròn dấu phẩy động.
        for i in range(n):
            for j in range(i):
                average = 0.5 * (result[i][j] + result[j][i])
                result[i][j] = result[j][i] = average
        method = "Hotelling cho ma trận đối xứng: Bₘớᵢ = B − λvvᵀ"
        pivot = None
    else:
        # Wielandt: chọn |v_s| lớn nhất, u = v/v_s, B_mới = B - u(e_s^T B).
        pivot = max(range(n), key=lambda i: abs(eigenvector[i]))
        if abs(eigenvector[pivot]) <= sys.float_info.epsilon:
            raise ArithmeticError("Không chọn được phần tử trụ khác 0 để xuống thang.")
        u = [value / eigenvector[pivot] for value in eigenvector]
        pivot_row = a[pivot][:]
        result = [
            [a[i][j] - u[i] * pivot_row[j] for j in range(n)]
            for i in range(n)
        ]
        method = "Wielandt tổng quát: Bₘớᵢ = B − (v/vₛ)(eₛᵀB)"

    tiny = 100.0 * sys.float_info.epsilon * matrix_frobenius_norm(a)
    result = [[0.0 if abs(value) <= tiny else value for value in row] for row in result]
    error = norm2(mat_vec(result, eigenvector))
    return DeflationResult(result, method, pivot, error)


def print_deflation(
    before: Matrix,
    after: DeflationResult,
    eigenvalue: float,
    eigenvector: Vector,
    stage: int,
    decimals: int,
) -> None:
    print_section(f"C. XUỐNG THANG LẦN {stage}")
    eigenvalue_name, vector_name = indexed("λ", stage), indexed("v", stage)
    print(f"Cặp trị riêng - vector riêng dùng để xuống thang: ({eigenvalue_name}, {vector_name}).")
    print(f"  {eigenvalue_name} = {format_number(eigenvalue, decimals)}")
    print_vector(eigenvector, vector_name, decimals, horizontal=True)
    print(f"  {norm(vector_name, 2)} = {norm2(eigenvector):.{decimals}f}")
    print(f"\nCông thức áp dụng: {after.method}.")
    if after.pivot is not None:
        print(f"  Chọn s = {after.pivot + 1} vì |({vector_name})ₛ| lớn nhất.")
    else:
        print("  B đối xứng và v đã có chuẩn 2 bằng 1, nên dùng trực tiếp λvv^T.")
    print("\nMa trận trước khi xuống thang:")
    print_matrix(before, indexed("B", stage - 1), decimals)
    print("\nMa trận sau khi xuống thang:")
    print_matrix(after.matrix, indexed("B", stage), decimals)
    print(
        f"\nKiểm tra trị riêng vừa khử: ‖{indexed('B', stage)}{vector_name}‖₂"
        f" = {format_scientific(after.annihilation_error)} (theo lý thuyết phải xấp xỉ 0)."
    )
    if after.annihilation_error > 10.0 ** (-decimals):
        print("  Nhận xét: sai lệch còn thấy rõ vì cặp (λ, v) mới là gần đúng;")
        print("  nếu đề không cố định số vòng lặp, nên lặp thêm trước khi xuống thang.")


def print_deflation_exam(
    before: Matrix,
    after: DeflationResult,
    eigenvalue: float,
    eigenvector: Vector,
    stage: int,
    decimals: int,
) -> None:
    del eigenvalue, eigenvector
    before_name = indexed("B", stage)
    after_name = indexed("B", stage + 1)
    eigenvalue_name, vector_name = indexed("λ", stage), indexed("v", stage)
    print("\nXUỐNG THANG")
    if "Hotelling" in after.method:
        print(f"{after_name} = {before_name} − {eigenvalue_name}{vector_name}{vector_name}ᵀ")
        print("(đây là trường hợp riêng của công thức PDF khi w = v)")
    elif is_symmetric(before):
        print(f"{after_name} = {before_name} − {eigenvalue_name}{vector_name}{vector_name}ᵀ")
    else:
        left_name = indexed("w", stage)
        print(
            f"{after_name} = {before_name} − "
            f"{eigenvalue_name}{vector_name}xᵀ, với x = {left_name}/({left_name}ᵀ{vector_name})"
        )
    print(f"Công thức áp dụng: {after.method}")
    print("Ma trận sau xuống thang:")
    print_matrix(after.matrix, after_name, decimals)


# =============================================================================
# BÁO CÁO VÀ CHƯƠNG TRÌNH CHÍNH
# =============================================================================

def input_print_mode() -> str:
    print("Chế độ in:")
    print("  1. Gọn để chép thi")
    print("  2. Đầy đủ để kiểm tra")
    while True:
        choice = input("Lựa chọn [Enter = 1]: ").strip() or "1"
        if choice == "1":
            return "exam"
        if choice == "2":
            return "full"
        print("  Lỗi: chỉ chọn 1 hoặc 2.")


def print_problem_data_exam(
    a: Matrix,
    x0: Vector,
    fixed_iterations: int,
    epsilon: float,
    wanted: int,
    decimals: int,
) -> None:
    print("\nDỮ KIỆN")
    print_matrix(a, "A", decimals)
    print_vector(x0, "X⁽⁰⁾", decimals, horizontal=True)
    if fixed_iterations > 0:
        print(f"k = {fixed_iterations}")
    else:
        print(f"ε = {epsilon:.3e}")
    print(f"Số trị riêng cần tìm: {wanted}")
    print("Chế độ PDF nghiêm ngặt: chỉ dùng Δ = ||X^(k) - X^(k-1)||₂ để dừng.")


def print_power_formula_exam() -> None:
    print("\nCÔNG THỨC")
    print("Y⁽ᵏ⁾ = B X⁽ᵏ⁻¹⁾")
    print("X⁽ᵏ⁾ = Y⁽ᵏ⁾ / ||Y⁽ᵏ⁾||₂")
    print("Δₖ = ||X⁽ᵏ⁾ - X⁽ᵏ⁻¹⁾||₂")
    print("Residual r⁽ᵏ⁾ = B X⁽ᵏ⁾ - λ⁽ᵏ⁾X⁽ᵏ⁾ chỉ để kiểm tra")
    print("λ = (XᵀBX)/(XᵀX) sau khi dừng theo Δₖ ≤ ε")


def print_search_warning_exam(search: DominantSearchResult) -> None:
    warning = search.warning.lower()
    message = ""
    if "phức" in warning or "phuc" in warning:
        message = "Lũy thừa thực cơ bản không phù hợp vì trị riêng trội là phức."
    elif (
        "không có một trị riêng trội duy nhất" in warning
        or "khong du dieu kien" in warning
        or "không đủ điều kiện" in warning
    ):
        message = "Không đủ điều kiện lũy thừa cơ bản vì không có trị riêng trội duy nhất theo môđun."
    elif search.initial_nearly_orthogonal or "(x⁽⁰⁾, v*)" in warning:
        message = "Chú ý: X⁽⁰⁾ gần vuông góc với hướng riêng trội, chương trình đã thử thêm các hướng eᵢ."
    if message:
        print("\n" + message)


def format_power_error_exam(error: ArithmeticError) -> str:
    text = str(error).lower()
    if "phức" in text or "phuc" in text:
        return "Lũy thừa thực cơ bản không phù hợp vì trị riêng trội là phức."
    if "trội duy nhất" in text or "troi duy nhat" in text:
        return "Không đủ điều kiện lũy thừa cơ bản vì không có trị riêng trội duy nhất theo môđun."
    return str(error).splitlines()[0]

def print_problem_data(
    a: Matrix,
    x0: Vector,
    fixed_iterations: int,
    epsilon: float,
    maximum_iterations: int,
    wanted: int,
    decimals: int,
    check_mode: str,
) -> None:
    print_section("DỮ KIỆN BÀI TOÁN")
    print(f"Cấp ma trận: n = {len(a)}.")
    print_matrix(a, "A", decimals)
    print_vector(x0, "X⁽⁰⁾", decimals, horizontal=True)
    if fixed_iterations > 0:
        print(f"Chế độ lặp: thực hiện đúng k = {fixed_iterations} vòng ở mỗi giai đoạn.")
    else:
        print(f"Chế độ lặp: đến khi ‖BX − λX‖₂ ≤ ε = {epsilon:.3e}.")
    print(f"Số vòng lặp tối đa mỗi giai đoạn: kₘₐₓ = {maximum_iterations}.")
    print(f"Số trị riêng cần tìm: {wanted}.")
    print("Chế độ thử vector đầu: " + ("kiểm tra chặt" if check_mode == "strict" else "nhanh"))
    print(f"Kết quả được trình bày với {decimals} chữ số sau dấu phẩy.")
    if is_symmetric(a):
        print("Nhận xét: A là ma trận đối xứng ⇒ các trị riêng thực và dùng được B − λvvᵀ.")
    else:
        print("Nhận xét: A không đối xứng ⇒ chương trình dùng công thức xuống thang Wielandt tổng quát.")


def print_final_summary(
    original: Matrix,
    values: Vector,
    vectors: list[Vector],
    residuals_on_original: Vector,
    final_matrix: Matrix,
    decimals: int,
    epsilon: float,
) -> None:
    print_section("D. KẾT QUẢ CUỐI CÙNG")
    for i, (value, vector, residual) in enumerate(zip(values, vectors, residuals_on_original), start=1):
        eigenvalue_name, vector_name = indexed("λ", i), indexed("v", i)
        denominator = (
            matrix_frobenius_norm(original) * norm2(vector)
            + abs(value) * norm2(vector)
            + sys.float_info.min
        )
        relative = residual / denominator
        print(f"\n{i}. {eigenvalue_name} ≈ {format_number(value, decimals)}")
        print_vector(vector, f"   {vector_name}", decimals, horizontal=True)
        print(f"   ‖A{vector_name} − {eigenvalue_name}{vector_name}‖₂ = {format_scientific(residual)}")
        print(
            f"   ‖A{vector_name} − {eigenvalue_name}{vector_name}‖₂ / "
            f"(‖A‖‖{vector_name}‖₂ + |{eigenvalue_name}|‖{vector_name}‖₂) "
            f"= {format_scientific(relative)}"
        )
        print(f"   {'Đạt' if relative <= epsilon else 'Không đạt'} ε.")
    if not values:
        print("Không tìm được cặp trị riêng - vector riêng nào đạt điều kiện để xuống thang.")
        print("Không có cặp trị riêng - vector riêng nào được chứng nhận đạt ε để ghi nhận xuống thang.")
    print("\nMa trận cuối quá trình:")
    print_matrix(final_matrix, indexed("B", len(values)), decimals)
    if is_symmetric(original) and len(vectors) > 1:
        print("\nKiểm tra tính trực giao của các vector riêng (ma trận đối xứng):")
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                vi, vj = indexed("v", i + 1), indexed("v", j + 1)
                print(f"  |{transpose(vi)}{vj}| = {abs(dot(vectors[i], vectors[j])):.7e}")
    print("\nKẾT LUẬN ĐỂ CHÉP VÀO BÀI:")
    if values:
        print(f"  Trị riêng trội đầu tiên của A là λ₁ ≈ {format_number(values[0], decimals)}.")
        print("  Vector riêng tương ứng đã được chuẩn hóa để ‖v₁‖₂ = 1.")
        if len(values) > 1:
            print(f"  Đã tìm được {len(values)} trị riêng bằng lũy thừa kết hợp xuống thang.")
    print("  Các số liệu trên được tính từ giá trị chưa làm tròn; chỉ phần hiển thị được làm tròn.")


def print_final_summary_exam(results: list[tuple[int, PowerResult]], decimals: int) -> None:
    print("\nKẾT LUẬN")
    if not results:
        print("Chưa có cặp trị riêng - vector riêng nào được chứng nhận đạt ε.")
        return
    for stage, result in results:
        eigenvalue_name, vector_name = indexed("λ", stage), indexed("v", stage)
        suffix = "" if result.converged else " (chưa đạt ε)"
        print(f"{eigenvalue_name} ≈ {format_number(result.eigenvalue, decimals)}{suffix}")
        print(f"{vector_name} ≈ {vector_inline(result.eigenvector, decimals)}")


def left_eigenvector_for(
    a: Matrix,
    eigenvalue: float,
    right_vector: Vector,
    *,
    epsilon: float = 1e-10,
    max_iter: int = 5000,
) -> Vector:
    """Tim vector rieng trai w cua A, tuc A^T w = lambda w."""
    transposed = transpose_matrix(a)
    candidates = [right_vector[:], [1.0] * len(a)]
    candidates.extend([[1.0 if i == j else 0.0 for i in range(len(a))] for j in range(len(a))])
    best: tuple[float, Vector] | None = None
    for candidate in candidates:
        try:
            _value, vector, _residual, relative = recover_eigenvector_on_original(
                transposed, eigenvalue, candidate, epsilon=epsilon, max_iter=max_iter
            )
        except ArithmeticError:
            continue
        if best is None or relative < best[0]:
            best = (relative, vector)
        if relative <= epsilon:
            return vector
    if best is None:
        raise ArithmeticError(
            "Khong thuc hien xuong thang cho ma tran khong doi xung vi chua co vector rieng trai du tin cay."
        )
    return best[1]


def deflate(a: Matrix, eigenvalue: float, eigenvector: Vector) -> DeflationResult:
    """Xuong thang dung cong thuc doi xung hoac vector rieng trai/phai."""
    n = len(a)
    if is_symmetric(a):
        result = [
            [a[i][j] - eigenvalue * eigenvector[i] * eigenvector[j] for j in range(n)]
            for i in range(n)
        ]
        for i in range(n):
            for j in range(i):
                average = 0.5 * (result[i][j] + result[j][i])
                result[i][j] = result[j][i] = average
        method = "Hotelling cho ma tran doi xung: B = A - lambda*v*v^T"
        left_vector = eigenvector[:]
        left_right_dot = dot(left_vector, eigenvector)
    else:
        left_vector = left_eigenvector_for(a, eigenvalue, eigenvector)
        left_right_dot = dot(left_vector, eigenvector)
        if abs(left_right_dot) <= 100.0 * sys.float_info.epsilon:
            raise ArithmeticError(
                "Khong thuc hien xuong thang cho ma tran khong doi xung vi w^T v = 0 hoac qua nho."
            )
        result = [
            [a[i][j] - eigenvalue * eigenvector[i] * left_vector[j] / left_right_dot for j in range(n)]
            for i in range(n)
        ]
        method = "Xuong thang khong doi xung: B = A - lambda*v*w^T/(w^T*v)"

    tiny = 100.0 * sys.float_info.epsilon * matrix_frobenius_norm(a)
    result = [[0.0 if abs(value) <= tiny else value for value in row] for row in result]
    error = norm2(mat_vec(result, eigenvector))
    return DeflationResult(result, method, None, error, eigenvector[:], left_vector, left_right_dot)


def deflate_pdf(a: Matrix, eigenvalue: float, eigenvector: Vector) -> DeflationResult:
    """Xuống thang đúng công thức PDF: A_new = A - lambda*v*x^T, x = w/(w^T v)."""
    n = len(a)
    if is_symmetric(a):
        result = [
            [a[i][j] - eigenvalue * eigenvector[i] * eigenvector[j] for j in range(n)]
            for i in range(n)
        ]
        for i in range(n):
            for j in range(i):
                average = 0.5 * (result[i][j] + result[j][i])
                result[i][j] = result[j][i] = average
        method = "Hotelling là trường hợp riêng của PDF khi w = v: B = A - lambda*v*v^T"
        left_vector = eigenvector[:]
        left_right_dot = dot(left_vector, eigenvector)
    else:
        left_vector = pdf_left_eigenvector(a, eigenvalue, eigenvector)
        left_right_dot = dot(left_vector, eigenvector)
        if abs(left_right_dot) <= 100.0 * sys.float_info.epsilon:
            raise ArithmeticError("Không thể xuống thang PDF vì w^T v = 0 hoặc quá nhỏ.")
        x = [value / left_right_dot for value in left_vector]
        result = [
            [a[i][j] - eigenvalue * eigenvector[i] * x[j] for j in range(n)]
            for i in range(n)
        ]
        method = "PDF không đối xứng: B = A - lambda*v*x^T, với x = w/(w^T*v)"

    tiny = 100.0 * sys.float_info.epsilon * matrix_frobenius_norm(a)
    result = [[0.0 if abs(value) <= tiny else value for value in row] for row in result]
    error = norm2(mat_vec(result, eigenvector))
    return DeflationResult(result, method, None, error, eigenvector[:], left_vector, left_right_dot)


def render_power_exam_report(result: PowerResult, *, stage: int = 1, decimals: int = 6) -> str:
    """Ban chep thi ngan gon, khong lo ten bien ky thuat Python."""
    eigenvalue_name = f"λ{stage}".translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
    vector_name = f"v{stage}".translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
    lines = [
        "Đề bài: tìm trị riêng trội và vector riêng tương ứng bằng phương pháp lũy thừa.",
        "Công thức: y⁽ᵏ⁺¹⁾ = Ax⁽ᵏ⁾, x⁽ᵏ⁺¹⁾ = y⁽ᵏ⁺¹⁾/‖y⁽ᵏ⁺¹⁾‖,",
        "λ⁽ᵏ⁺¹⁾ = (x⁽ᵏ⁺¹⁾)ᵀAx⁽ᵏ⁺¹⁾ / (x⁽ᵏ⁺¹⁾)ᵀx⁽ᵏ⁺¹⁾.",
        "Điều kiện áp dụng: có trị riêng trội duy nhất theo môđun và vector đầu có thành phần theo hướng riêng trội.",
        "k | x⁽ᵏ⁾ | λ⁽ᵏ⁾ | ‖x⁽ᵏ⁾-x⁽ᵏ⁻¹⁾‖ | ‖Ax⁽ᵏ⁾-λ⁽ᵏ⁾x⁽ᵏ⁾‖",
    ]
    records = result.iterations
    if len(records) > 6:
        shown = records[:3] + records[-3:]
        gap_after = records[2].index
    else:
        shown = records
        gap_after = -1
    for record in shown:
        if record.index == gap_after + 1 and len(records) > 6:
            lines.append("...")
        vector = "[" + ", ".join(format_number(value, decimals) for value in record.x) + "]ᵀ"
        lines.append(
            f"{record.index} | {vector} | {format_number(record.rayleigh, decimals)} | "
            f"{record.delta:.3e} | {record.residual:.3e}"
        )
    vector = "[" + ", ".join(format_number(value, decimals) for value in result.eigenvector) + "]ᵀ"
    lines.extend(
        [
            f"Kết quả: {eigenvalue_name} ≈ {format_number(result.eigenvalue, decimals)}, {vector_name} ≈ {vector}.",
            f"‖A{vector_name} − {eigenvalue_name}{vector_name}‖₂ = {result.residual:.6e}.",
            f"Residual tương đối = {result.relative_residual:.6e}.",
        ]
    )
    if result.converged:
        lines.append(
            f"Kết luận: Sau {len(result.iterations)} bước lặp, ta thu được trị riêng trội {eigenvalue_name} "
            f"và vector riêng tương ứng {vector_name}. Vì ‖A{vector_name} − {eigenvalue_name}{vector_name}‖₂ rất nhỏ nên kết quả được chấp nhận."
        )
    else:
        lines.append(
            "Kết luận: Phương pháp lũy thừa cơ bản chưa đủ điều kiện tách trị riêng trội duy nhất theo môđun. "
            "Kết quả trên chỉ nên xem là quan sát tính toán."
        )
    if result.warning:
        lines.append("Cảnh báo: " + result.warning)
    return "\n".join(lines)


def _assert_close(value: float, expected: float, tolerance: float, message: str) -> None:
    if abs(value - expected) > tolerance:
        raise AssertionError(f"{message}: nhận {value}, cần {expected}.")


def _diag(values: Vector) -> Matrix:
    return [
        [values[i] if i == j else 0.0 for j in range(len(values))]
        for i in range(len(values))
    ]


def run_self_tests() -> None:
    print_section("SELF-TEST LŨY THỪA VÀ XUỐNG THANG")

    search = dominant_eigenpair(_diag([5.0, 2.0, 1.0]), [0.0, 1.0, 1.0])
    _assert_close(search.result.eigenvalue, 5.0, 1e-8, "Test 1: phải tìm λ = 5")
    if "X⁽⁰⁾ gần vuông góc" not in search.warning:
        raise AssertionError("Test 1: thiếu cảnh báo vector đầu xấu.")

    search = dominant_eigenpair(_diag([5.0, 2.0, 1.0]), [1e-12, 1.0, 1.0])
    _assert_close(search.result.eigenvalue, 5.0, 1e-8, "Test 2: phải tìm λ = 5")
    if not search.initial_nearly_orthogonal:
        raise AssertionError("Test 2: thiếu phát hiện gần vuông góc.")

    collection = eigenpairs_with_deflation(
        _diag([5.0, 3.0, 1.0]),
        2,
        [1.0, 1.0, 1.0],
    )
    if not collection.success or len(collection.eigenvalues) != 2:
        raise AssertionError("Test 3: xuống thang đối xứng không tìm đủ 2 cặp.")
    if any(value > 1e-8 for value in collection.relative_residuals):
        raise AssertionError("Test 3: vector riêng trên A gốc chưa đạt phần dư.")

    search = dominant_eigenpair(_diag([3.0, -3.0, 1.0]), [1.0, 1.0, 1.0])
    if search.dominant_certified:
        raise AssertionError("Test 4: phải cảnh báo không có trị riêng trội duy nhất.")

    try:
        dominant_eigenpair([[0.0, -1.0], [1.0, 0.0]], [1.0, 0.0], max_iter=30)
    except ArithmeticError as error:
        if "phức" not in str(error) and "phuc" not in str(error):
            raise AssertionError("Test 5: lỗi phải nói rõ trị riêng phức trội.") from error
    else:
        raise AssertionError("Test 5: ma trận quay phải bị từ chối rõ ràng.")

    nonsymmetric = [[2.0, 1.0], [0.0, 1.0]]
    search = dominant_eigenpair(nonsymmetric, [1.0, 1.0])
    deflated = deflate(nonsymmetric, search.result.eigenvalue, search.result.eigenvector)
    if "Hotelling" in deflated.method:
        raise AssertionError("Test 6: ma trận không đối xứng không được dùng Hotelling.")

    search = dominant_eigenpair(
        [[2.0, 1.0], [1.0, 3.0]],
        [1.0, 0.0],
        fixed_iterations=3,
        epsilon=1e-12,
    )
    if len(search.result.iterations) != 3:
        raise AssertionError("Test 7: k cố định phải trả kết quả sau đúng 3 vòng.")
    if search.result.converged:
        raise AssertionError("Test 7: với ε rất nhỏ, kết quả sau 3 vòng chưa được đánh dấu đạt ε.")
    if FIXED_ITERATION_WARNING not in search.warning:
        raise AssertionError("Test 7: thiếu cảnh báo k cố định chưa chứng nhận đạt ε.")

    print("Đã qua 7 self-test bắt buộc.")


def main() -> None:
    print_mode = "exam" if any(arg in {"--pdf", "--exam", "--chep-thi"} for arg in sys.argv[1:]) else "full"
    if print_mode == "full":
        print(LINE)
        print("TÌM TRỊ RIÊNG TRỘI BẰNG PHƯƠNG PHÁP LŨY THỪA VÀ XUỐNG THANG")
        print(LINE)
        print("Nhập số liệu theo đề; có thể nhập phân số như 1/3. Nhấn Enter để dùng giá trị mặc định.")
    else:
        print("\nNhập số liệu theo đề; có thể nhập phân số như 1/3.")

    n = input_int("\nNhập cấp n của ma trận vuông A: ", 1)
    a = input_matrix(n)
    x0 = input_initial_vector(n)

    print("\nChế độ dừng:")
    print("  • Nhập k > 0: thực hiện đúng k vòng lặp như đề bài.")
    print("  • Nhập k = 0: tự lặp đến khi đạt sai số ε.")
    fixed_iterations = input_int("Nhập k [Enter = 0, lặp đến khi đạt ε]: ", 0, 0)
    epsilon = input_float("Nhập sai số ε dùng để kiểm tra [Enter = 1e-7]: ", 1e-7)
    maximum_iterations = 1000
    if fixed_iterations == 0 and "--ask-kmax" in sys.argv[1:]:
        maximum_iterations = input_int("Nhập kₘₐₓ [Enter = 1000]: ", 1, 1000)

    if print_mode == "full" and "--ask-start-check" in sys.argv[1:]:
        print("\nChế độ thử vector đầu:")
        print("  1. Nhanh: X⁽⁰⁾, các eᵢ và vài hướng xác định")
        print("  2. Kiểm tra chặt: thêm nhiễu 10⁻³, 10⁻⁶, 10⁻⁹ quanh X⁽⁰⁾")
        while True:
            check_choice = input("Lựa chọn [Enter = 1]: ").strip() or "1"
            if check_choice in {"1", "2"}:
                break
            print("  Lỗi: chỉ chọn 1 hoặc 2.")
        check_mode = "strict" if check_choice == "2" else "fast"
    else:
        check_mode = "fast"

    wanted = input_int(f"Số trị riêng muốn tìm, từ 1 đến {n} [Enter = 1]: ", 1, 1)
    while wanted > n:
        print(f"  Lỗi: ma trận cấp {n} nên số trị riêng cần tìm không vượt quá {n}.")
        wanted = input_int(f"Nhập lại số trị riêng (1..{n}): ", 1)
    decimals = input_int("Số chữ số sau dấu phẩy [Enter = 7]: ", 0, 7)

    original = copy_matrix(a)
    if print_mode == "exam":
        print_problem_data_exam(a, x0, fixed_iterations, epsilon, wanted, decimals)
        print_power_formula_exam()

        current = copy_matrix(a)
        exam_summary_results: list[tuple[int, PowerResult]] = []
        current_start = x0[:]

        for stage in range(1, wanted + 1):
            print_section(f"B. GIAI ĐOẠN {stage}: LŨY THỪA PDF")
            print_matrix(current, f"B_{stage - 1}", decimals)
            print_vector(current_start, "X^(0)", decimals, horizontal=True)
            try:
                result = power_method_pdf_case1(
                    current,
                    current_start,
                    fixed_iterations=fixed_iterations,
                    epsilon=epsilon,
                    max_iter=maximum_iterations,
                )
            except ArithmeticError as error:
                print(f"\nGiai đoạn {stage}: {format_power_error_exam(error)}")
                break
            print_power_result_exam(result, stage, decimals, matrix_label=f"B_{stage - 1}")
            exam_summary_results.append((stage, result))
            if stage == wanted:
                break
            if not result.converged:
                print("\nKhông xuống thang vì Δ chưa đạt ε; kết quả trên chỉ là xấp xỉ sau đúng k vòng.")
                break
            try:
                deflated = deflate_pdf(current, result.eigenvalue, result.eigenvector)
            except ArithmeticError as error:
                print(f"\nGiai đoạn {stage}: {error}")
                break
            print_deflation_exam(current, deflated, result.eigenvalue, result.eigenvector, stage, decimals)
            current = deflated.matrix
            current_start = [float(i + 1) for i in range(n)]

        print_final_summary_exam(exam_summary_results, decimals)
        return

    if print_mode == "full":
        print_problem_data(a, x0, fixed_iterations, epsilon, maximum_iterations, wanted, decimals, check_mode)
        print_power_theory()
    else:
        print_problem_data_exam(a, x0, fixed_iterations, epsilon, wanted, decimals)
        print_power_formula_exam()

    current = copy_matrix(a)
    values: Vector = []
    vectors: list[Vector] = []
    residuals_on_original: Vector = []
    exam_summary_results: list[tuple[int, PowerResult]] = []
    current_start = x0[:]

    for stage in range(1, wanted + 1):
        if print_mode == "full":
            print_section(f"B. GIAI ĐOẠN {stage}: TÌM TRỊ RIÊNG TRỘI CỦA B_{stage - 1}")
            print_matrix(current, f"B_{stage - 1}", decimals)
            print_vector(current_start, "X^(0)", decimals, horizontal=True)

        try:
            search = dominant_eigenpair(
                current,
                current_start,
                fixed_iterations=fixed_iterations,
                epsilon=epsilon,
                max_iter=maximum_iterations,
                check_mode=check_mode,
            )
            result = search.result
            if print_mode == "full":
                print_start_attempt_table(search, decimals)
                print_initial_vector_check(search)
                if search.warning:
                    print("\nCẢNH BÁO:", search.warning)
            else:
                print_search_warning_exam(search)
        except ArithmeticError as error:
            if print_mode == "full":
                print(f"\nDừng tại giai đoạn {stage}: {error}")
                print("Không thực hiện xuống thang và không ghi nhận trị riêng chưa hội tụ.")
            else:
                print(f"\nGiai đoạn {stage}: {format_power_error_exam(error)}")
            break

        if print_mode == "full":
            current_matrix_label = "A" if stage == 1 else indexed("B", stage - 1)
        else:
            current_matrix_label = "A" if stage == 1 else indexed("B", stage)
        if fixed_iterations > 0 and not result.converged:
            if print_mode == "full":
                print_power_result(
                    result,
                    stage,
                    decimals,
                    epsilon=epsilon,
                    fixed_iterations=fixed_iterations,
                    matrix_label=current_matrix_label,
                )
                print(
                    "\nKhông xuống thang tự động vì phần dư chưa đạt ε; "
                    "kết quả trên chỉ là xấp xỉ sau đúng k vòng."
                )
            else:
                print_power_result_exam(result, stage, decimals, matrix_label=current_matrix_label)
                exam_summary_results.append((stage, result))
            break

        verified_value = result.eigenvalue
        verified_vector = result.eigenvector
        verified_residual = residual_norm(original, verified_value, verified_vector)
        verified_denominator = (
            matrix_frobenius_norm(original) * norm2(verified_vector)
            + abs(verified_value) * norm2(verified_vector)
            + sys.float_info.min
        )
        verified_relative = verified_residual / verified_denominator
        if not is_symmetric(original) or verified_relative > epsilon:
            try:
                (
                    verified_value,
                    verified_vector,
                    verified_residual,
                    verified_relative,
                ) = recover_eigenvector_on_original(
                    original,
                    result.eigenvalue,
                    result.eigenvector,
                    epsilon=epsilon,
                    max_iter=maximum_iterations,
                )
            except ArithmeticError as error:
                if print_mode == "full":
                    print(f"\nDừng tại giai đoạn {stage}: {error}")
                    print("Không xuống thang và không ghi nhận cặp trị riêng chưa được kiểm tra.")
                else:
                    print(f"\nGiai đoạn {stage}: {format_power_error_exam(error)}")
                break

        verified_result = PowerResult(
            eigenvalue=verified_value,
            eigenvector=verified_vector,
            iterations=result.iterations,
            converged=verified_relative <= epsilon,
            residual=verified_residual,
            relative_residual=verified_relative,
            stop_reason=result.stop_reason + "; đã kiểm tra lại trên ma trận A gốc",
        )
        if not verified_result.converged:
            if fixed_iterations > 0:
                if print_mode == "full":
                    print_power_result(
                        verified_result,
                        stage,
                        decimals,
                        epsilon=epsilon,
                        fixed_iterations=fixed_iterations,
                        matrix_label="A",
                    )
                    print(
                        "\nKhông xuống thang tự động vì phần dư trên A gốc chưa đạt ε; "
                        "kết quả trên chỉ là xấp xỉ sau đúng k vòng."
                    )
                else:
                    print_power_result_exam(verified_result, stage, decimals, matrix_label="A")
                    exam_summary_results.append((stage, verified_result))
            else:
                if print_mode == "full":
                    print(f"\nDừng tại giai đoạn {stage}: phần dư trên A gốc chưa đạt epsilon.")
                    print("Không xuống thang và không ghi nhận kết quả.")
                else:
                    print(f"\nGiai đoạn {stage}: phần dư trên A gốc chưa đạt ε.")
            break

        if print_mode == "full":
            print_power_result(
                verified_result,
                stage,
                decimals,
                epsilon=epsilon,
                fixed_iterations=fixed_iterations,
                matrix_label="A",
            )
        else:
            print_power_result_exam(verified_result, stage, decimals, matrix_label="A")
            exam_summary_results.append((stage, verified_result))
        values.append(verified_value)
        vectors.append(verified_vector)
        residuals_on_original.append(verified_residual)

        # Xuống thang dùng vector riêng của ma trận hiện tại; vector in/kết luận
        # ở trên là vector đã được khôi phục và kiểm tra trên A gốc.
        if print_mode == "full" or stage < wanted:
            deflated = deflate(current, result.eigenvalue, result.eigenvector)
            if print_mode == "full":
                print_deflation(current, deflated, result.eigenvalue, result.eigenvector, stage, decimals)
            else:
                print_deflation_exam(current, deflated, result.eigenvalue, result.eigenvector, stage, decimals)
            current = deflated.matrix

        if stage == wanted:
            break

        # Vector toàn 1 có thể vô tình trực giao với vector riêng cần tìm tiếp theo.
        # Dãy 1,2,...,n là lựa chọn xác định và thường tránh được trường hợp đó.
        current_start = [float(i + 1) for i in range(n)]

    if print_mode == "full":
        print_final_summary(original, values, vectors, residuals_on_original, current, decimals, epsilon)
    else:
        print_final_summary_exam(exam_summary_results, decimals)


if __name__ == "__main__":
    try:
        if "--self-test" in sys.argv:
            run_self_tests()
        else:
            main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except ValueError as error:
        print(f"\nLỗi dữ liệu: {error}")
    except Exception as error:
        print(f"\nLỗi trong quá trình tính toán: {error}")
