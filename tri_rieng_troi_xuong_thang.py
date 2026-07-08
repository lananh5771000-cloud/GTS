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
from fractions import Fraction


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")


Matrix = list[list[float]]
Vector = list[float]
LINE = "=" * 100
THIN_LINE = "-" * 100


# =============================================================================
# NHẬP DỮ LIỆU
# =============================================================================

def parse_real(token: str) -> float:
    """Đọc số nguyên, số thập phân, dạng khoa học hoặc phân số a/b."""
    token = token.strip()
    if "," in token and "." not in token:
        token = token.replace(",", ".")
    value = float(Fraction(token))
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
        tokens = input(prompt).split()
        if len(tokens) != size:
            print(f"  Lỗi: dòng phải có đúng {size} số.")
            continue
        try:
            return [parse_real(token) for token in tokens]
        except (ValueError, ZeroDivisionError):
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


@dataclass
class DominantSearchResult:
    result: PowerResult
    attempts: list[PowerResult]
    dominant_certified: bool
    warning: str


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
        alpha, x_new = component_normalize(y)
        rayleigh = rayleigh_quotient(a, x_new)

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

        current_norm = norm2(x_new)
        denominator = (
            matrix_frobenius_norm(a) * current_norm
            + abs(rayleigh) * current_norm
            + sys.float_info.min
        )
        relative_residual = residual / denominator
        if fixed_iterations == 0 and relative_residual <= epsilon:
            converged = True
            stop_reason = f"phần dư đã thỏa ||BX - λX||₂ <= ε = {epsilon:.3e}"
            break

    unit_vector = normalize_2(x)
    if not records:
        raise ArithmeticError("Không tạo được bước lặp nào.")
    eigenvalue = rayleigh_quotient(a, unit_vector)
    residual = residual_norm(a, eigenvalue, unit_vector)
    denominator = (
        matrix_frobenius_norm(a) * norm2(unit_vector)
        + abs(eigenvalue) * norm2(unit_vector)
        + sys.float_info.min
    )
    relative_residual = residual / denominator

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
    )


def dominant_eigenpair(
    a: Matrix,
    x0: Vector | None = None,
    *,
    epsilon: float = 1e-10,
    max_iter: int = 5000,
    fixed_iterations: int = 0,
    unique_dominant_verified: bool = False,
) -> DominantSearchResult:
    """Thử nhiều hướng xác định để tránh bỏ sót không gian riêng trội."""
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("A phải là ma trận vuông khác rỗng.")
    candidates: list[Vector] = []
    if x0 is not None and len(x0) == n and norm2(x0) > 0:
        candidates.append([float(value) for value in x0])
    candidates.append([1.0] * n)
    candidates.extend([[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)])
    candidates.append([float(i + 1) for i in range(n)])
    candidates.append([(-1.0 if i % 2 else 1.0) * (i + 1) for i in range(n)])

    attempts: list[PowerResult] = []
    for candidate in candidates:
        try:
            result = power_method(
                a, candidate, fixed_iterations, epsilon, max_iter
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

    valid = [item for item in attempts if item.converged and item.relative_residual <= epsilon]
    if not valid:
        raise ArithmeticError(
            "Không tìm được trị riêng thực trội hội tụ bằng phương pháp lũy thừa thực."
        )
    best = max(valid, key=lambda item: (abs(item.eigenvalue), -item.relative_residual))
    values = [item.eigenvalue for item in valid]
    spread = max((abs(abs(v) - abs(best.eigenvalue)) for v in values), default=0.0)
    warning = ""
    if not unique_dominant_verified:
        warning = (
            "Đã thử nhiều vector đầu và chọn trị riêng có môđun lớn nhất quan sát được; "
            "điều này không thay thế chứng minh trị riêng trội duy nhất."
        )
    if spread > max(epsilon * max(abs(best.eigenvalue), sys.float_info.min), 1e-8 * abs(best.eigenvalue)):
        warning += " Các vector đầu đã hội tụ tới những trị riêng khác nhau."
    return DominantSearchResult(best, attempts, unique_dominant_verified, warning.strip())


def print_power_theory() -> None:
    print_section("A. THUẬT TOÁN PHƯƠNG PHÁP LŨY THỪA")
    print("Input:")
    print("  • Ma trận vuông B cấp n có một trị riêng trội về môđun.")
    print("  • Vector khởi đầu X^(0) khác 0 và có thành phần theo vector riêng trội.")
    print("  • Số vòng lặp k, hoặc sai số ε và số vòng lặp tối đa k_max.")
    print("Output:")
    print("  • Trị riêng trội gần đúng λ và vector riêng v được đưa về chuẩn 2 bằng 1.")
    print("Các bước:")
    print("  B1. Tính Y^(k) = B X^(k-1).")
    print("  B2. Chọn p sao cho |y_p^(k)| = max_i |y_i^(k)| và đặt α_k = y_p^(k).")
    print("  B3. Chuẩn hóa X^(k) = Y^(k)/α_k.")
    print("  B4. α_k chỉ là hệ số chuẩn hóa, không dùng làm trị riêng cuối.")
    print("  B5. Ước lượng trị riêng bằng thương Rayleigh:")
    print("                λ_R^(k) = (X^(k))^T B X^(k) / (X^(k))^T X^(k).")
    print("  B6. Kiểm tra phần dư r^(k) = B X^(k) - λ_R^(k) X^(k).")
    print("  B7. Cuối cùng đặt v = X^(k)/||X^(k)||₂ để ||v||₂ = 1.")


def print_iteration_details(record: Iteration, decimals: int) -> None:
    print(f"\nVòng lặp {record.index}:")
    print_vector(record.y, f"Y^({record.index}) = B X^({record.index - 1})", decimals, horizontal=True)
    print(
        f"  α_{record.index} = phần tử của Y^({record.index}) có môđun lớn nhất"
        f" = {format_number(record.alpha, decimals)}"
    )
    print_vector(record.x, f"X^({record.index}) = Y^({record.index})/α_{record.index}", decimals, horizontal=True)
    print(f"  λ_R^({record.index}) (thương Rayleigh) = {format_number(record.rayleigh, decimals)}")
    print(f"  ||X^({record.index}) - (+/-)X^({record.index - 1})||₂ = {format_scientific(record.delta)}")
    print(f"  ||B X^({record.index}) - λ_R^({record.index})X^({record.index})||₂ = {format_scientific(record.residual)}")


def print_iteration_table(records: list[Iteration], decimals: int) -> None:
    print("\nBảng tổng hợp các lần lặp:")
    component_count = len(records[0].x)
    headers = ["k", "α_k", "λ_R^(k)", "||r||₂"] + [f"x{i + 1}^(k)" for i in range(component_count)]
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


def print_power_result(result: PowerResult, stage: int, decimals: int) -> None:
    for record in result.iterations:
        print_iteration_details(record, decimals)
    print_iteration_table(result.iterations, decimals)

    print("\nKết quả của giai đoạn:")
    print(f"  Dừng vì {result.stop_reason}.")
    print(f"  Trị riêng trội gần đúng: λ_{stage} = {format_number(result.eigenvalue, decimals)}")
    print_vector(result.eigenvector, f"v_{stage}", decimals)
    print(f"  ||v_{stage}||₂ = {norm2(result.eigenvector):.{decimals}f}")
    print(f"  Sai số phần dư tuyệt đối ||Bv_{stage} - λ_{stage}v_{stage}||₂ = {format_scientific(result.residual)}")
    print(f"  Sai số phần dư tương đối = {format_scientific(result.relative_residual)}")
    if result.converged:
        print("  Đánh giá: kết quả đạt sai số ε đã nhập.")
    else:
        print("  Lưu ý: kết quả CHƯA đạt ε; nếu đề không cố định k, cần tăng số vòng lặp.")


# =============================================================================
# PHƯƠNG PHÁP XUỐNG THANG
# =============================================================================

@dataclass
class DeflationResult:
    matrix: Matrix
    method: str
    pivot: int | None
    annihilation_error: float


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


def deflate(a: Matrix, eigenvalue: float, eigenvector: Vector) -> DeflationResult:
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
        method = "Hotelling cho ma trận đối xứng: B_mới = B - λvv^T"
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
        method = "Wielandt tổng quát: B_mới = B - (v/v_s)(e_s^T B)"

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
    print(f"Cặp trị riêng - vector riêng dùng để xuống thang: (λ_{stage}, v_{stage}).")
    print(f"  λ_{stage} = {format_number(eigenvalue, decimals)}")
    print_vector(eigenvector, f"v_{stage}", decimals, horizontal=True)
    print(f"  ||v_{stage}||₂ = {norm2(eigenvector):.{decimals}f}")
    print(f"\nCông thức áp dụng: {after.method}.")
    if after.pivot is not None:
        print(f"  Chọn s = {after.pivot + 1} vì |(v_{stage})_s| lớn nhất.")
    else:
        print("  B đối xứng và v đã có chuẩn 2 bằng 1, nên dùng trực tiếp λvv^T.")
    print("\nMa trận trước khi xuống thang:")
    print_matrix(before, f"B_{stage - 1}", decimals)
    print("\nMa trận sau khi xuống thang:")
    print_matrix(after.matrix, f"B_{stage}", decimals)
    print(
        f"\nKiểm tra trị riêng vừa khử: ||B_{stage}v_{stage}||₂"
        f" = {format_scientific(after.annihilation_error)} (theo lý thuyết phải xấp xỉ 0)."
    )
    if after.annihilation_error > 10.0 ** (-decimals):
        print("  Nhận xét: sai lệch còn thấy rõ vì cặp (λ, v) mới là gần đúng;")
        print("  nếu đề không cố định số vòng lặp, nên lặp thêm trước khi xuống thang.")


# =============================================================================
# BÁO CÁO VÀ CHƯƠNG TRÌNH CHÍNH
# =============================================================================

def print_problem_data(
    a: Matrix,
    x0: Vector,
    fixed_iterations: int,
    epsilon: float,
    maximum_iterations: int,
    wanted: int,
    decimals: int,
) -> None:
    print_section("DỮ KIỆN BÀI TOÁN")
    print(f"Cấp ma trận: n = {len(a)}.")
    print_matrix(a, "A", decimals)
    print_vector(x0, "X^(0)", decimals, horizontal=True)
    if fixed_iterations > 0:
        print(f"Chế độ lặp: thực hiện đúng k = {fixed_iterations} vòng ở mỗi giai đoạn.")
    else:
        print(f"Chế độ lặp: đến khi ||BX - λX||₂ <= ε = {epsilon:.3e}.")
        print(f"Số vòng lặp tối đa mỗi giai đoạn: k_max = {maximum_iterations}.")
    print(f"Số trị riêng cần tìm: {wanted}.")
    print(f"Kết quả được trình bày với {decimals} chữ số sau dấu phẩy.")
    if is_symmetric(a):
        print("Nhận xét: A là ma trận đối xứng ⇒ các trị riêng thực và dùng được B - λvv^T.")
    else:
        print("Nhận xét: A không đối xứng ⇒ chương trình dùng công thức xuống thang Wielandt tổng quát.")


def print_final_summary(
    original: Matrix,
    values: Vector,
    vectors: list[Vector],
    residuals_on_original: Vector,
    final_matrix: Matrix,
    decimals: int,
) -> None:
    print_section("D. KẾT QUẢ CUỐI CÙNG")
    for i, (value, vector, residual) in enumerate(zip(values, vectors, residuals_on_original), start=1):
        print(f"\n{i}. λ_{i} ≈ {format_number(value, decimals)}")
        print_vector(vector, f"   v_{i}", decimals, horizontal=True)
        print(f"   ||Av_{i} - λ_{i}v_{i}||₂ = {format_scientific(residual)}")
    if not values:
        print("Không tìm được cặp trị riêng - vector riêng nào.")
    print("\nMa trận cuối quá trình:")
    print_matrix(final_matrix, f"B_{len(values)}", decimals)
    if is_symmetric(original) and len(vectors) > 1:
        print("\nKiểm tra tính trực giao của các vector riêng (ma trận đối xứng):")
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                print(f"  |v_{i + 1}^T v_{j + 1}| = {abs(dot(vectors[i], vectors[j])):.7e}")
    print("\nKẾT LUẬN ĐỂ CHÉP VÀO BÀI:")
    if values:
        print(f"  Trị riêng trội đầu tiên của A là λ_1 ≈ {format_number(values[0], decimals)}.")
        print(f"  Vector riêng tương ứng đã được chuẩn hóa để ||v_1||₂ = 1.")
        if len(values) > 1:
            print(f"  Đã tìm được {len(values)} trị riêng bằng lũy thừa kết hợp xuống thang.")
    print("  Các số liệu trên được tính từ giá trị chưa làm tròn; chỉ phần hiển thị được làm tròn.")


def main() -> None:
    print(LINE)
    print("TÌM TRỊ RIÊNG TRỘI BẰNG PHƯƠNG PHÁP LŨY THỪA VÀ XUỐNG THANG")
    print(LINE)
    print("Nhập số liệu theo đề; có thể nhập phân số như 1/3. Nhấn Enter để dùng giá trị mặc định.")

    n = input_int("\nNhập cấp n của ma trận vuông A: ", 1)
    a = input_matrix(n)
    x0 = input_initial_vector(n)

    print("\nChế độ dừng:")
    print("  • Nhập k > 0: thực hiện đúng k vòng lặp như đề bài.")
    print("  • Nhập k = 0: tự lặp đến khi đạt sai số ε.")
    fixed_iterations = input_int("Nhập k [Enter = 3]: ", 0, 3)
    epsilon = input_float("Nhập sai số ε dùng để kiểm tra [Enter = 1e-7]: ", 1e-7)
    maximum_iterations = 1000
    if fixed_iterations == 0:
        maximum_iterations = input_int("Nhập k_max [Enter = 1000]: ", 1, 1000)

    wanted = input_int(f"Số trị riêng muốn tìm, từ 1 đến {n} [Enter = 1]: ", 1, 1)
    while wanted > n:
        print(f"  Lỗi: ma trận cấp {n} nên số trị riêng cần tìm không vượt quá {n}.")
        wanted = input_int(f"Nhập lại số trị riêng (1..{n}): ", 1)
    decimals = input_int("Số chữ số sau dấu phẩy [Enter = 7]: ", 0, 7)

    original = copy_matrix(a)
    print_problem_data(a, x0, fixed_iterations, epsilon, maximum_iterations, wanted, decimals)
    print_power_theory()

    current = copy_matrix(a)
    values: Vector = []
    vectors: list[Vector] = []
    residuals_on_original: Vector = []
    current_start = x0[:]

    for stage in range(1, wanted + 1):
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
            )
            result = search.result
            if search.warning:
                print("\nCẢNH BÁO:", search.warning)
        except ArithmeticError as error:
            print(f"\nDừng tại giai đoạn {stage}: {error}")
            print("Không thực hiện xuống thang và không ghi nhận trị riêng chưa hội tụ.")
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
                print(f"\nDừng tại giai đoạn {stage}: {error}")
                print("Không xuống thang và không ghi nhận cặp trị riêng chưa được kiểm tra.")
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
            print(f"\nDừng tại giai đoạn {stage}: phần dư trên A gốc chưa đạt epsilon.")
            print("Không xuống thang và không ghi nhận kết quả.")
            break

        print_power_result(verified_result, stage, decimals)
        values.append(verified_value)
        vectors.append(verified_vector)
        residuals_on_original.append(verified_residual)

        # Xuống thang dùng vector riêng của ma trận hiện tại; vector in/kết luận
        # ở trên là vector đã được khôi phục và kiểm tra trên A gốc.
        deflated = deflate(current, result.eigenvalue, result.eigenvector)
        print_deflation(current, deflated, result.eigenvalue, result.eigenvector, stage, decimals)
        current = deflated.matrix

        if stage == wanted:
            break

        # Vector toàn 1 có thể vô tình trực giao với vector riêng cần tìm tiếp theo.
        # Dãy 1,2,...,n là lựa chọn xác định và thường tránh được trường hợp đó.
        current_start = [float(i + 1) for i in range(n)]

    print_final_summary(original, values, vectors, residuals_on_original, current, decimals)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except ValueError as error:
        print(f"\nLỗi dữ liệu: {error}")
    except Exception as error:
        print(f"\nLỗi trong quá trình tính toán: {error}")
