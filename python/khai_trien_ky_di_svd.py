"""
PHÂN TÍCH GIÁ TRỊ KỲ DỊ BẰNG LŨY THỪA VÀ XUỐNG THANG
================================================================

Chương trình trình bày theo dạng bài thi:
  1. Nêu lý thuyết SVD và thuật toán.
  2. Lập B = A^T A.
  3. Tìm từng trị riêng của B bằng phương pháp lũy thừa.
  4. Xuống thang B_i = B_(i-1) - lambda_i v_i v_i^T.
  5. Suy ra sigma_i = sqrt(lambda_i), u_i = A v_i / sigma_i.
  6. In SVD rút gọn và kiểm tra sai số tái tạo.

Mọi phép tính trung gian dùng số chưa làm tròn. Chỉ phần hiển thị được làm tròn.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")


LINE = "=" * 100
THIN_LINE = "-" * 100


# =============================================================================
# NHẬP DỮ LIỆU
# =============================================================================

def parse_number(token: str) -> float:
    token = token.strip().replace("−", "-")
    if "," in token and "." not in token:
        token = token.replace(",", ".")
    value = float(Fraction(token))
    if not math.isfinite(value):
        raise ValueError
    return value


def input_integer(prompt: str, minimum: int, default: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            suffix = f", Enter = {default}" if default is not None else ""
            print(f"  Lỗi: hãy nhập số nguyên >= {minimum}{suffix}.")


def input_positive(prompt: str, default: float | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            value = parse_number(raw)
            if value <= 0.0:
                raise ValueError
            return value
        except (ValueError, ZeroDivisionError):
            print("  Lỗi: hãy nhập số dương, ví dụ 1e-10, 0.0001 hoặc 1/1000.")


def input_row(prompt: str, length: int) -> list[float]:
    while True:
        raw = input(prompt)
        for symbol in "[](){};":
            raw = raw.replace(symbol, " ")
        tokens = raw.split()
        if len(tokens) != length:
            print(f"  Lỗi: phải nhập đúng {length} số, cách nhau bằng dấu cách.")
            continue
        try:
            return [parse_number(token) for token in tokens]
        except (ValueError, ZeroDivisionError):
            print("  Lỗi: dữ liệu không hợp lệ; có thể nhập số thập phân hoặc phân số a/b.")


def input_matrix(rows: int, columns: int) -> np.ndarray:
    print(f"\nNhập ma trận A kích thước {rows}×{columns}:")
    return np.array(
        [input_row(f"  Dòng {i + 1}: ", columns) for i in range(rows)],
        dtype=float,
    )


def input_start_vector(size: int) -> np.ndarray:
    print("\nVector khởi đầu cho phương pháp lũy thừa:")
    print("  1. Dùng y = (1, 1, ..., 1)^T")
    print("  2. Tự nhập")
    while True:
        choice = input("Chọn [Enter = 1]: ").strip() or "1"
        if choice == "1":
            return np.ones(size, dtype=float)
        if choice == "2":
            vector = np.array(input_row(f"Nhập {size} phần tử của y: ", size))
            if np.linalg.norm(vector) == 0.0:
                print("  Lỗi: vector khởi đầu không được bằng 0.")
                continue
            return vector
        print("  Lỗi: chỉ chọn 1 hoặc 2.")


# =============================================================================
# HIỂN THỊ
# =============================================================================

def clean(value: float, decimals: int) -> float:
    threshold = 0.5 * 10.0 ** (-decimals) if decimals > 0 else 0.5
    return 0.0 if abs(value) < threshold else float(value)


def number(value: float, decimals: int) -> str:
    return f"{clean(value, decimals):.{decimals}f}"


def matrix_lines(matrix: np.ndarray, decimals: int) -> list[str]:
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    cells = [[number(value, decimals) for value in row] for row in matrix]
    widths = [max(len(cells[i][j]) for i in range(len(cells))) for j in range(len(cells[0]))]
    result: list[str] = []
    for i, row in enumerate(cells):
        content = "  ".join(value.rjust(widths[j]) for j, value in enumerate(row))
        if len(cells) == 1:
            left, right = "[", "]"
        elif i == 0:
            left, right = "⎡", "⎤"
        elif i == len(cells) - 1:
            left, right = "⎣", "⎦"
        else:
            left, right = "⎢", "⎥"
        result.append(f"{left} {content} {right}")
    return result


def print_matrix(name: str, matrix: np.ndarray, decimals: int) -> None:
    lines = matrix_lines(matrix, decimals)
    middle = len(lines) // 2
    padding = " " * (len(name) + 3)
    for i, line in enumerate(lines):
        print((f"{name} = " if i == middle else padding) + line)


def print_vector(name: str, vector: np.ndarray, decimals: int, horizontal: bool = False) -> None:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if horizontal:
        text = "  ".join(number(value, decimals) for value in vector)
        print(f"{name} = [{text}]^T")
    else:
        print_matrix(name, vector[:, None], decimals)


def section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def scientific(value: float) -> str:
    return f"{float(value):.7e}"


# =============================================================================
# ĐẠI SỐ TUYẾN TÍNH PHỤ TRỢ
# =============================================================================

def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector, 2))
    if norm == 0.0:
        raise ArithmeticError("không thể chuẩn hóa vector 0")
    result = np.asarray(vector, dtype=float) / norm
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result = -result
    return result


def orthogonalize(vector: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    # Gram-Schmidt cải tiến hai lượt để giảm mất trực giao.
    for _ in range(2):
        for base in basis:
            result -= float(base @ result) * base
    return result


def independent_start(preferred: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    candidates = [np.asarray(preferred, dtype=float)]
    size = preferred.size
    candidates.extend(np.eye(size, dtype=float))
    candidates.append(np.arange(1.0, size + 1.0))
    for candidate in candidates:
        candidate = orthogonalize(candidate, basis)
        if np.linalg.norm(candidate) > 1e-12:
            return normalize(candidate)
    raise ArithmeticError("không tìm được vector khởi đầu độc lập với các vector đã có")


def complete_orthonormal_basis(existing: list[np.ndarray], dimension: int) -> list[np.ndarray]:
    # Giữ nguyên dấu của các cột đã có. Đổi dấu U và V độc lập ở đây sẽ phá
    # quan hệ Av_i=sigma_i*u_i dù từng cơ sở riêng vẫn trực chuẩn.
    basis = []
    for vector in existing:
        vector = np.asarray(vector, dtype=float)
        length = float(np.linalg.norm(vector, 2))
        if length == 0.0:
            raise ArithmeticError("không thể hoàn thiện cơ sở từ vector 0")
        basis.append(vector / length)
    if len(basis) >= dimension:
        return basis[:dimension]
    for candidate in np.eye(dimension):
        vector = orthogonalize(candidate, basis)
        if np.linalg.norm(vector) > 1e-12:
            basis.append(normalize(vector))
        if len(basis) == dimension:
            break
    return basis


# =============================================================================
# LŨY THỪA VÀ XUỐNG THANG
# =============================================================================

@dataclass
class Iteration:
    k: int
    eigenvalue: float
    vector: np.ndarray
    delta: float
    residual: float


@dataclass
class PowerResult:
    eigenvalue: float
    eigenvector: np.ndarray
    history: list[Iteration]
    converged: bool
    stop_reason: str


@dataclass
class SVDResult:
    U: np.ndarray
    singular_values: np.ndarray
    Vt: np.ndarray
    rank: int
    reconstruction_error: float
    relative_reconstruction_error: float
    left_orthogonality_error: float
    right_orthogonality_error: float
    relation_residuals: list[tuple[float, float]]
    converged: bool
    warnings: list[str]
    rank_tolerance: float = 0.0
    iteration_tolerance: float = 0.0
    orthogonality_tolerance: float = 0.0
    reconstruction_tolerance: float = 0.0
    scale_factor: float = 1.0
    power_results: list[PowerResult] | None = None


def power_method_symmetric(
    matrix: np.ndarray,
    initial: np.ndarray,
    locked_vectors: list[np.ndarray],
    epsilon: float,
    maximum_iterations: int,
    fixed_steps: int,
) -> PowerResult:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Ma trận dùng cho lũy thừa phải vuông.")
    if initial.shape != (matrix.shape[0],):
        raise ValueError("Kích thước vector đầu không phù hợp.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(initial)):
        raise ValueError("Ma trận và vector đầu phải hữu hạn.")
    if epsilon <= 0 or maximum_iterations <= 0 or fixed_steps < 0:
        raise ValueError("epsilon, max_iter phải dương; fixed_steps không âm.")

    size = matrix.shape[0]
    raw_candidates = [
        np.asarray(initial, dtype=float),
        np.ones(size, dtype=float),
        *list(np.eye(size, dtype=float)),
        np.arange(1.0, size + 1.0),
    ]
    viable: list[tuple[float, np.ndarray]] = []
    scale = float(np.linalg.norm(matrix, "fro"))
    for candidate in raw_candidates:
        candidate = orthogonalize(candidate, locked_vectors)
        if np.linalg.norm(candidate) <= np.finfo(float).tiny:
            continue
        candidate = normalize(candidate)
        image = orthogonalize(matrix @ candidate, locked_vectors)
        image_norm = float(np.linalg.norm(image, 2))
        if image_norm > np.finfo(float).tiny * max(scale, np.finfo(float).tiny):
            viable.append((image_norm, candidate))
    x = max(viable, key=lambda item: item[0])[1] if viable else independent_start(initial, locked_vectors)
    history: list[Iteration] = []
    limit = fixed_steps if fixed_steps > 0 else maximum_iterations
    matrix_scale = float(np.linalg.norm(matrix, "fro"))

    # Nếu phần ma trận còn lại bằng 0 thì trị riêng tiếp theo bằng 0.
    if matrix_scale == 0.0:
        return PowerResult(0.0, x, [], True, "ma trận còn lại bằng 0")

    converged = False
    stop_reason = "đã thực hiện đủ số vòng lặp đề bài yêu cầu"
    for k in range(1, limit + 1):
        y = matrix @ x
        y = orthogonalize(y, locked_vectors)
        if np.linalg.norm(y) <= np.finfo(float).tiny * max(matrix_scale, np.finfo(float).tiny):
            raise ArithmeticError(
                "vector khởi đầu rơi vào không gian riêng ứng với trị riêng 0; hãy đổi vector đầu"
            )
        x_new = normalize(y)
        if float(x_new @ x) < 0.0:
            x_new = -x_new

        eigenvalue = float(x_new @ matrix @ x_new)
        residual = float(np.linalg.norm(matrix @ x_new - eigenvalue * x_new, 2))
        delta = min(
            float(np.linalg.norm(x_new - x, 2)),
            float(np.linalg.norm(x_new + x, 2)),
        )
        history.append(Iteration(k, eigenvalue, x_new.copy(), delta, residual))
        x = x_new

        relative_residual = residual / max(
            matrix_scale + abs(eigenvalue), np.finfo(float).tiny
        )
        if fixed_steps == 0 and relative_residual <= epsilon:
            converged = True
            stop_reason = f"||Bv - λv||₂ <= ε = {epsilon:.3e}"
            break

    if not history:
        raise ArithmeticError("không tạo được bước lặp nào")
    if fixed_steps > 0:
        last_scale = max(matrix_scale + abs(history[-1].eigenvalue), np.finfo(float).tiny)
        converged = history[-1].residual / last_scale <= epsilon
    elif not converged:
        stop_reason = f"đã đạt k_max = {maximum_iterations} nhưng chưa đạt ε"
    last = history[-1]
    return PowerResult(last.eigenvalue, last.vector, history, converged, stop_reason)


def deflate_symmetric(matrix: np.ndarray, eigenvalue: float, vector: np.ndarray) -> np.ndarray:
    result = matrix - eigenvalue * np.outer(vector, vector)
    result = 0.5 * (result + result.T)
    tolerance = 100.0 * np.finfo(float).eps * float(np.linalg.norm(matrix, "fro"))
    result[np.abs(result) <= tolerance] = 0.0
    return result


def svd_power(
    matrix: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    epsilon: float = 1e-10,
    max_iter: int = 10000,
    fixed_steps: int = 0,
    full_matrices: bool = False,
    rank_tolerance: float | None = None,
    iteration_tolerance: float | None = None,
    orthogonality_tolerance: float | None = None,
    reconstruction_tolerance: float | None = None,
) -> SVDResult:
    """SVD bằng lũy thừa trên ma trận đã đổi thang; không gọi numpy.linalg.svd."""
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or min(A.shape) == 0:
        raise ValueError("A phải là ma trận chữ nhật khác rỗng.")
    if not np.all(np.isfinite(A)):
        raise ValueError("A chỉ được chứa số hữu hạn.")
    if epsilon <= 0 or max_iter <= 0 or fixed_steps < 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon và max_iter phải dương.")
    m, n = A.shape
    start = np.ones(n) if initial is None else np.asarray(initial, dtype=float)
    if start.shape != (n,) or not np.all(np.isfinite(start)):
        raise ValueError("Vector đầu phải có đúng n phần tử hữu hạn.")

    for name, tolerance in (
        ("rank_tolerance", rank_tolerance),
        ("iteration_tolerance", iteration_tolerance),
        ("orthogonality_tolerance", orthogonality_tolerance),
        ("reconstruction_tolerance", reconstruction_tolerance),
    ):
        if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0.0):
            raise ValueError(f"{name} phải không âm và hữu hạn.")

    # Chuẩn max-entry là một chuẩn ma trận an toàn để đổi thang: không bình
    # phương trước khi chia, nên tránh cả underflow và overflow khi lập A^T A.
    alpha = float(np.max(np.abs(A)))
    eps_machine = np.finfo(float).eps
    safe_min = np.finfo(float).tiny
    iter_tol = epsilon if iteration_tolerance is None else iteration_tolerance
    orth_tol = (
        max(100.0 * max(m, n) * eps_machine, 10.0 * epsilon)
        if orthogonality_tolerance is None
        else orthogonality_tolerance
    )
    reconstruct_tol = (
        max(100.0 * max(m, n) * eps_machine, 10.0 * epsilon)
        if reconstruction_tolerance is None
        else reconstruction_tolerance
    )

    if alpha == 0.0:
        if full_matrices:
            U, V = np.eye(m), np.eye(n)
            values = np.zeros(min(m, n))
        else:
            U, V = np.zeros((m, 0)), np.zeros((n, 0))
            values = np.zeros(0)
        return SVDResult(
            U=U,
            singular_values=values,
            Vt=V.T,
            rank=0,
            reconstruction_error=0.0,
            relative_reconstruction_error=0.0,
            left_orthogonality_error=0.0,
            right_orthogonality_error=0.0,
            relation_residuals=[],
            converged=True,
            warnings=["A là ma trận 0; sai số tái tạo tuyệt đối bằng 0."],
            rank_tolerance=0.0 if rank_tolerance is None else rank_tolerance,
            iteration_tolerance=iter_tol,
            orthogonality_tolerance=orth_tol,
            reconstruction_tolerance=reconstruct_tol,
            scale_factor=alpha,
            power_results=[],
        )

    A_scaled = A / alpha
    B = A_scaled.T @ A_scaled
    locked: list[np.ndarray] = []
    pairs: list[tuple[float, np.ndarray, PowerResult]] = []
    warnings: list[str] = []
    for _ in range(min(m, n)):
        try:
            power = power_method_symmetric(B, start, locked, iter_tol, max_iter, fixed_steps)
        except ArithmeticError:
            break
        raw_v = np.asarray(power.eigenvector, dtype=float)
        v = orthogonalize(raw_v, locked)
        # Nếu phần độc lập chỉ còn ở mức nhiễu làm tròn thì không chuẩn hóa nó
        # thành một hướng giả rồi lặp lại trị kỳ dị đã khóa.
        if np.linalg.norm(v) <= 100.0 * eps_machine * max(np.linalg.norm(raw_v), safe_min):
            break
        v = normalize(v)
        image = A_scaled @ v
        sigma_scaled = float(np.linalg.norm(image, 2))
        if sigma_scaled == 0.0:
            break
        locked.append(v)
        pairs.append((sigma_scaled, v, power))

    pairs.sort(key=lambda item: item[0], reverse=True)
    sigma_max_scaled = pairs[0][0] if pairs else 0.0
    if rank_tolerance is None:
        rank_tol_scaled = max(m, n) * eps_machine * sigma_max_scaled
        rank_tol_original = alpha * rank_tol_scaled
    else:
        rank_tol_original = rank_tolerance
        rank_tol_scaled = rank_tolerance / alpha

    resolved_pairs = [item for item in pairs if item[0] > rank_tol_scaled]
    singular_scaled = np.array([item[0] for item in resolved_pairs], dtype=float)
    singular_values = alpha * singular_scaled
    rank = len(resolved_pairs)
    V_columns = [item[1] for item in resolved_pairs]
    U_columns = [
        (A_scaled @ v) / sigma
        for sigma, v, _power in resolved_pairs
    ]
    relation_residuals: list[tuple[float, float]] = []
    scaled_norm = float(np.linalg.norm(A_scaled, "fro"))
    for sigma, u, v in zip(singular_scaled, U_columns, V_columns):
        left_abs = float(np.linalg.norm(A_scaled @ v - sigma * u, 2))
        right_abs = float(np.linalg.norm(A_scaled.T @ u - sigma * v, 2))
        relation_residuals.append((
            left_abs / max(scaled_norm * np.linalg.norm(v) + sigma * np.linalg.norm(u), safe_min),
            right_abs / max(scaled_norm * np.linalg.norm(u) + sigma * np.linalg.norm(v), safe_min),
        ))

    if full_matrices:
        U_columns = complete_orthonormal_basis(U_columns, m)
        V_columns = complete_orthonormal_basis(V_columns, n)
        U = np.column_stack(U_columns)
        V = np.column_stack(V_columns)
        displayed_values = np.zeros(min(m, n))
        displayed_values[:rank] = singular_values
    else:
        U = np.column_stack(U_columns) if rank else np.zeros((m, 0))
        V = np.column_stack(V_columns) if rank else np.zeros((n, 0))
        Sigma = np.diag(singular_scaled)
        reconstruction_scaled = U @ Sigma @ V.T if rank else np.zeros_like(A_scaled)
        displayed_values = singular_values

    # Ở nhánh full, Sigma phải ở thang của A_scaled để tái tạo ổn định.
    if full_matrices:
        Sigma_scaled = np.zeros((m, n))
        for i, sigma in enumerate(singular_scaled):
            Sigma_scaled[i, i] = sigma
        reconstruction_scaled = U @ Sigma_scaled @ V.T
    scaled_error = float(np.linalg.norm(A_scaled - reconstruction_scaled, "fro"))
    absolute_error = alpha * scaled_error
    relative_error = scaled_error / max(float(np.linalg.norm(A_scaled, "fro")), safe_min)
    left_error = float(np.linalg.norm(U.T @ U - np.eye(U.shape[1]), "fro"))
    right_error = float(np.linalg.norm(V.T @ V - np.eye(V.shape[1]), "fro"))
    required_triplets_converged = all(item[2].converged for item in resolved_pairs)
    triplets_ok = all(
        left <= iter_tol and right <= iter_tol
        for left, right in relation_residuals
    )
    converged = bool(
        required_triplets_converged
        and triplets_ok
        and left_error <= orth_tol
        and right_error <= orth_tol
        and relative_error <= reconstruct_tol
    )
    if rank < min(m, n):
        warnings.append(
            "Hạng số theo tolerance bằng "
            f"{rank} (rank_tol={rank_tol_original:.3e}); đây không phải tuyên bố hạng toán học."
        )
    if not converged:
        warnings.append(
            "SVD chưa đạt đồng thời điều kiện bộ ba kỳ dị, trực giao và tái tạo."
        )
    return SVDResult(
        U=U,
        singular_values=displayed_values,
        Vt=V.T,
        rank=rank,
        reconstruction_error=absolute_error,
        relative_reconstruction_error=relative_error,
        left_orthogonality_error=left_error,
        right_orthogonality_error=right_error,
        relation_residuals=relation_residuals,
        converged=converged,
        warnings=warnings,
        rank_tolerance=rank_tol_original,
        iteration_tolerance=iter_tol,
        orthogonality_tolerance=orth_tol,
        reconstruction_tolerance=reconstruct_tol,
        scale_factor=alpha,
        power_results=[item[2] for item in resolved_pairs],
    )


def selected_history(history: list[Iteration]) -> list[Iteration | None]:
    """Không làm ngập màn hình khi cần hàng trăm vòng lặp."""
    if len(history) <= 20:
        return list(history)
    return list(history[:5]) + [None] + list(history[-5:])


def print_iteration_table(result: PowerResult, decimals: int) -> None:
    if not result.history:
        print("  Không cần lặp vì ma trận còn lại bằng 0.")
        return
    n = result.eigenvector.size
    headers = ["k", "λ^(k)", "||Δv||₂", "||Bv-λv||₂"] + [f"v{i + 1}^(k)" for i in range(n)]
    rows: list[list[str] | None] = []
    for item in selected_history(result.history):
        if item is None:
            rows.append(None)
            continue
        rows.append([
            str(item.k),
            number(item.eigenvalue, decimals),
            f"{item.delta:.2e}",
            f"{item.residual:.2e}",
            *[number(value, decimals) for value in item.vector],
        ])
    actual_rows = [row for row in rows if row is not None]
    widths = [max(len(headers[j]), *(len(row[j]) for row in actual_rows)) for j in range(len(headers))]
    print("  " + " | ".join(headers[j].rjust(widths[j]) for j in range(len(headers))))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        if row is None:
            print("  ... các vòng lặp ở giữa được lược bớt khi hiển thị ...")
        else:
            print("  " + " | ".join(row[j].rjust(widths[j]) for j in range(len(row))))


# =============================================================================
# TRÌNH BÀY LÝ THUYẾT VÀ KẾT QUẢ
# =============================================================================

def print_theory(m: int, n: int) -> None:
    section("A. LÝ THUYẾT PHÂN TÍCH GIÁ TRỊ KỲ DỊ (SVD)")
    print(f"Với A ∈ R^({m}×{n}), tồn tại phân tích")
    print("                         A = U Σ V^T,")
    print("trong đó:")
    print("  • U là ma trận trực giao; các cột u_i là vector kỳ dị trái.")
    print("  • V là ma trận trực giao; các cột v_i là vector kỳ dị phải.")
    print("  • Σ là ma trận đường chéo chữ nhật, chứa σ_1 ≥ σ_2 ≥ ... ≥ 0.")
    print("  • v_i là vector riêng của A^T A ứng với λ_i = σ_i².")
    print("  • Nếu σ_i > 0 thì u_i = Av_i/σ_i và ||u_i||₂ = ||v_i||₂ = 1.")
    print("Nếu rank(A)=r, SVD rút gọn là")
    print("             A = U_r Σ_r V_r^T = Σ_(i=1)^r σ_i u_i v_i^T.")

    section("B. THUẬT TOÁN LŨY THỪA KẾT HỢP XUỐNG THANG")
    print("Input:")
    print("  • Ma trận A cấp m×n; vector đầu y ∈ R^n khác 0.")
    print("  • Sai số ε, số vòng lặp tối đa k_max hoặc số bước cố định k.")
    print("Output: các giá trị kỳ dị σ_i và các vector kỳ dị tương ứng.")
    print("Các bước:")
    print("  B1. Lập B_0 = A^T A (ma trận đối xứng, nửa xác định dương).")
    print("  B2. Với i = 1, 2, ..., min(m,n):")
    print("      a) Dùng lũy thừa tìm cặp trị riêng trội (λ_i,v_i) của B_(i-1):")
    print("           z^(k) = B_(i-1)v^(k-1),")
    print("           v^(k) = z^(k)/||z^(k)||₂,")
    print("           λ^(k) = (v^(k))^T B_(i-1)v^(k)  (thương Rayleigh).")
    print("      b) Suy ra σ_i = sqrt(max(λ_i,0)).")
    print("      c) Nếu σ_i>0, tính u_i = Av_i/σ_i.")
    print("      d) Nếu chưa phải bước cuối, xuống thang:")
    print("                   B_i = B_(i-1) - λ_i v_i v_i^T.")
    print("  B3. Sắp các σ_i giảm dần và ghép U_r, Σ_r, V_r.")


def print_stage(
    stage: int,
    matrix: np.ndarray,
    result: PowerResult,
    sigma: float,
    next_matrix: np.ndarray,
    decimals: int,
) -> None:
    section(f"C.{stage}. LẦN LẶP XUỐNG THANG THỨ {stage}")
    print("Ma trận dùng để tìm trị riêng trội:")
    print_matrix(f"B_{stage - 1}", matrix, decimals)
    print("\nBảng lặp lũy thừa (vector luôn được chuẩn hóa theo chuẩn 2):")
    print_iteration_table(result, decimals)
    print("\nKết quả giai đoạn:")
    print(f"  λ_{stage} ≈ {number(result.eigenvalue, decimals)}")
    print_vector(f"v_{stage}", result.eigenvector, decimals, horizontal=True)
    print(f"  ||v_{stage}||₂ = {np.linalg.norm(result.eigenvector):.{decimals}f}")
    residual = np.linalg.norm(matrix @ result.eigenvector - result.eigenvalue * result.eigenvector)
    print(f"  ||B_{stage - 1}v_{stage} - λ_{stage}v_{stage}||₂ = {scientific(residual)}")
    print(f"  σ_{stage} = sqrt(λ_{stage}) ≈ {number(sigma, decimals)}")
    print(f"  Dừng vì {result.stop_reason}.")
    if result.converged:
        print("  Đánh giá: cặp trị riêng đạt sai số ε.")
    else:
        print("  Lưu ý: chưa đạt ε; nếu đề không cố định k thì cần tăng số vòng lặp.")
    print("\nXuống thang:")
    print(f"  B_{stage} = B_{stage - 1} - λ_{stage}v_{stage}v_{stage}^T")
    print_matrix(f"B_{stage}", next_matrix, decimals)
    annihilation = np.linalg.norm(next_matrix @ result.eigenvector)
    print(f"  Kiểm tra ||B_{stage}v_{stage}||₂ = {scientific(annihilation)} (xấp xỉ 0).")


def build_reduced_svd(
    matrix: np.ndarray,
    singular_values: list[float],
    right_vectors: list[np.ndarray],
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    positive = [i for i, sigma in enumerate(singular_values) if sigma > tolerance]
    rank = len(positive)
    if rank == 0:
        return (
            np.zeros((matrix.shape[0], 0)),
            np.zeros((0, 0)),
            np.zeros((matrix.shape[1], 0)),
            0,
        )
    values = [singular_values[i] for i in positive]
    V = np.column_stack([right_vectors[i] for i in positive])
    U_columns: list[np.ndarray] = []
    for index, (sigma, vector) in enumerate(zip(values, V.T)):
        image = matrix @ vector
        image_norm = float(np.linalg.norm(image, 2))
        if image_norm <= tolerance:
            raise ArithmeticError("không dựng được vector kỳ dị trái khác 0")
        # Không Gram-Schmidt U độc lập vì sẽ phá A v_i = sigma_i u_i.
        # Dùng trực tiếp ảnh Av_i và đồng bộ lại sigma theo ||Av_i||.
        values[index] = image_norm
        U_columns.append(image / image_norm)
    U = np.column_stack(U_columns)
    Sigma = np.diag(values)
    return U, Sigma, V, rank


def print_final(
    A: np.ndarray,
    eigenvalues: list[float],
    singular_values: list[float],
    right_vectors: list[np.ndarray],
    decimals: int,
) -> None:
    section("D. KẾT QUẢ CUỐI CÙNG")
    print("Các trị riêng của B=A^T A và giá trị kỳ dị tương ứng:")
    for i, (eigenvalue, sigma) in enumerate(zip(eigenvalues, singular_values), start=1):
        print(f"  λ_{i} ≈ {number(eigenvalue, decimals)}  ⇒  σ_{i}=√λ_{i} ≈ {number(sigma, decimals)}")

    scale = float(np.linalg.norm(A, 2))
    tolerance = max(A.shape) * np.finfo(float).eps * scale
    U, Sigma, V, rank = build_reduced_svd(A, singular_values, right_vectors, tolerance)
    print(f"\nHạng số của A theo các σ_i tìm được: r = {rank}.")
    if rank == 0:
        print("A là ma trận 0; SVD rút gọn không có cột.")
        return

    print("\nSVD rút gọn A ≈ U_r Σ_r V_r^T:")
    print_matrix("U_r", U, decimals)
    print_matrix("Σ_r", Sigma, decimals)
    print_matrix("V_r", V, decimals)
    reconstruction = U @ Sigma @ V.T
    print("\nMa trận tái tạo từ SVD:")
    print_matrix("U_r Σ_r V_r^T", reconstruction, decimals)
    absolute_error = float(np.linalg.norm(A - reconstruction, "fro"))
    norm_a = float(np.linalg.norm(A, "fro"))
    relative_error = absolute_error / norm_a if norm_a > 0.0 else absolute_error
    print(f"\n||A-U_rΣ_rV_r^T||_F = {scientific(absolute_error)}")
    print(f"Sai số tái tạo tương đối = {scientific(relative_error)}")
    print("\nKiểm tra trực chuẩn:")
    print(f"  ||U_r^T U_r-I||_F = {scientific(np.linalg.norm(U.T @ U - np.eye(rank), 'fro'))}")
    print(f"  ||V_r^T V_r-I||_F = {scientific(np.linalg.norm(V.T @ V - np.eye(rank), 'fro'))}")
    print("\nKiểm tra từng cặp kỳ dị:")
    for i, sigma in enumerate(np.diag(Sigma)):
        left = np.linalg.norm(A @ V[:, i] - sigma * U[:, i], 2)
        right = np.linalg.norm(A.T @ U[:, i] - sigma * V[:, i], 2)
        print(f"  i={i + 1}: ||Av_i-sigma_i u_i||2={scientific(left)}, ||A^T u_i-sigma_i v_i||2={scientific(right)}")
    print("\nKẾT LUẬN ĐỂ CHÉP VÀO BÀI:")
    print("  Các giá trị kỳ dị của A là:")
    print("  " + ", ".join(f"σ_{i + 1} ≈ {number(value, decimals)}" for i, value in enumerate(singular_values)))
    print("  Các vector v_i ở từng bước đều đã được đưa về chuẩn 2 bằng 1.")
    print("  Mọi ma trận B_i và vector dùng để xuống thang đã được ghi ở trên.")


# =============================================================================
# CHƯƠNG TRÌNH CHÍNH
# =============================================================================

def print_svd_result(A: np.ndarray, result: SVDResult, decimals: int, count: int | None = None) -> None:
    section("D. KẾT QUẢ SVD VÀ KIỂM TRA")
    shown = len(result.singular_values) if count is None else min(count, len(result.singular_values))
    print(f"Hệ số đổi thang alpha=max|a_ij|={result.scale_factor:.7e}.")
    print("Các trị kỳ dị có thể phân giải được (đã khôi phục về thang ban đầu):")
    for i, sigma in enumerate(result.singular_values[:shown], start=1):
        print(f"  sigma_{i} = {sigma:.12e}")
    print(
        f"Hạng số theo tolerance: {result.rank}; "
        f"rank_tol={result.rank_tolerance:.7e}."
    )
    print("Không đồng nhất 'hạng số theo tolerance' với hạng toán học nếu dữ liệu kém điều kiện.")

    print_matrix("U", result.U, decimals)
    Sigma = np.zeros(A.shape)
    for i, sigma in enumerate(result.singular_values[: min(A.shape)]):
        Sigma[i, i] = sigma
    print_matrix("Sigma", Sigma, decimals)
    print_matrix("V^T", result.Vt, decimals)
    reconstruction = result.U @ Sigma @ result.Vt
    print_matrix("U Sigma V^T", reconstruction, decimals)

    print("\nKiểm tra từng bộ ba kỳ dị (phần dư tương đối):")
    for i, (left, right) in enumerate(result.relation_residuals, start=1):
        print(
            f"  i={i}: rel(Av-sigma*u)={left:.7e}; "
            f"rel(A^T u-sigma*v)={right:.7e}."
        )
    print(f"||U^T U-I||_F={result.left_orthogonality_error:.7e} (tol={result.orthogonality_tolerance:.7e})")
    print(f"||V^T V-I||_F={result.right_orthogonality_error:.7e} (tol={result.orthogonality_tolerance:.7e})")
    print(f"||A-U Sigma V^T||_F={result.reconstruction_error:.7e}")
    print(
        "Sai số tái tạo tương đối="
        f"{result.relative_reconstruction_error:.7e} "
        f"(tol={result.reconstruction_tolerance:.7e})."
    )
    for warning in result.warnings:
        print("CẢNH BÁO:", warning)
    if result.converged:
        print("KẾT LUẬN: SVD đạt đồng thời điều kiện bộ ba kỳ dị, trực giao và tái tạo.")
    else:
        print("KẾT LUẬN: SVD CHƯA HỘI TỤ; không chứng nhận kết quả cuối.")

def main() -> None:
    print(LINE)
    print("TÌM GIÁ TRỊ KỲ DỊ BẰNG PHƯƠNG PHÁP LŨY THỪA VÀ XUỐNG THANG")
    print(LINE)
    print("Có thể nhập số nguyên, thập phân hoặc phân số a/b; Enter dùng giá trị mặc định.")

    m = input_integer("\nNhập số dòng m của A: ", 1)
    n = input_integer("Nhập số cột n của A: ", 1)
    A = input_matrix(m, n)
    initial = input_start_vector(n)

    print("\nChế độ dừng phương pháp lũy thừa:")
    print("  1. Lặp đến khi đạt sai số ε")
    print("  2. Thực hiện đúng k vòng như đề yêu cầu")
    while True:
        choice = input("Chọn [Enter = 1]: ").strip() or "1"
        if choice in {"1", "2"}:
            break
        print("  Lỗi: chỉ chọn 1 hoặc 2.")
    epsilon = input_positive("Nhập ε dùng để kiểm tra [Enter = 1e-10]: ", 1e-10)
    fixed_steps = 0
    maximum_iterations = 10000
    if choice == "1":
        maximum_iterations = input_integer("Nhập k_max [Enter = 10000]: ", 1, 10000)
    else:
        fixed_steps = input_integer("Nhập số vòng lặp k: ", 1)

    count_max = min(m, n)
    count = input_integer(
        f"Số giá trị kỳ dị cần tìm, từ 1 đến {count_max} [Enter = {count_max}]: ",
        1,
        count_max,
    )
    while count > count_max:
        print(f"  Lỗi: chỉ có tối đa {count_max} giá trị kỳ dị.")
        count = input_integer(f"Nhập lại (1..{count_max}): ", 1)
    decimals = input_integer("Số chữ số sau dấu phẩy [Enter = 7]: ", 0, 7)

    print_theory(m, n)
    section("C. ÁP DỤNG CHO MA TRẬN ĐÃ NHẬP")
    print_matrix("A", A, decimals)
    print("\nTrước khi lập A^T A, chương trình đổi thang A_scaled=A/alpha để tránh tràn/ngầm 0.")
    result = svd_power(
        A,
        initial=initial,
        epsilon=epsilon,
        max_iter=maximum_iterations,
        fixed_steps=fixed_steps,
        full_matrices=True,
    )
    print_svd_result(A, result, decimals, count=count)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except Exception as error:
        print(f"\nLỗi trong quá trình tính toán: {error}")
