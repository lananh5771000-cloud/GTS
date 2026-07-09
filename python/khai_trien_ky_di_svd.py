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
from input_utils import MathInputError, parse_real, split_number_row
from stopping_utils import scaled_tolerance

import numpy as np

from exam_format import exam_print as print, indexed


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
    value = parse_real(token)
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
        try:
            tokens = split_number_row(input(prompt), length)
            return [parse_number(token) for token in tokens]
        except (MathInputError, ValueError, ZeroDivisionError):
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
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return [f"[ ma trận rỗng {matrix.shape[0]}×{matrix.shape[1]} ]"]
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


def singular_text(value: float, decimals: int) -> str:
    value = float(value)
    if value != 0.0 and (abs(value) < 10.0 ** (-decimals) or abs(value) >= 10.0 ** (decimals + 1)):
        return f"{value:.12e}"
    return number(value, decimals)


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
    lambda_delta: float = math.inf
    relative_residual: float = math.inf


@dataclass
class PowerResult:
    eigenvalue: float
    eigenvector: np.ndarray
    history: list[Iteration]
    converged: bool
    stop_reason: str
    initial_vector: np.ndarray | None = None
    raw_vector: np.ndarray | None = None
    relative_residual: float = math.inf
    stagnated: bool = False
    max_iter_reached: bool = False
    fixed_steps_completed: bool = False


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
    deflation_matrices: list[np.ndarray] | None = None
    internal_power_converged: bool = True
    singular_triplet_residual_ok: bool = True
    orthogonality_ok: bool = True
    reconstruction_ok: bool = True
    final_accepted: bool = True
    final_absolute_tolerance: float = 0.0
    final_relative_tolerance: float = 0.0
    singular_triplet_tolerance: float = 0.0


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
    initial_used = x.copy()
    history: list[Iteration] = []
    limit = fixed_steps if fixed_steps > 0 else maximum_iterations
    matrix_scale = float(np.linalg.norm(matrix, "fro"))

    # Nếu phần ma trận còn lại bằng 0 thì trị riêng tiếp theo bằng 0.
    if matrix_scale == 0.0:
        return PowerResult(
            0.0, x, [], True, "ma trận còn lại bằng 0",
            initial_vector=initial_used, raw_vector=np.zeros_like(x),
        )

    converged = False
    stop_reason = "đã thực hiện đủ số vòng lặp đề bài yêu cầu"
    raw_y = np.zeros_like(x)
    stagnation_limit = 8
    stagnation_count = 0
    no_improve_limit = 12
    no_improve_count = 0
    sign_flip_count = 0
    stagnated = False
    max_iter_reached = False
    best_relative_residual = math.inf
    previous_eigenvalue: float | None = None
    final_relative_residual = math.inf
    stagnation_tol = max(100.0 * size * np.finfo(float).eps, 0.1 * epsilon)
    lambda_tol = max(100.0 * size * np.finfo(float).eps, 0.1 * epsilon)
    for k in range(1, limit + 1):
        y = matrix @ x
        y = orthogonalize(y, locked_vectors)
        if np.linalg.norm(y) <= np.finfo(float).tiny * max(matrix_scale, np.finfo(float).tiny):
            raise ArithmeticError(
                "vector khởi đầu rơi vào không gian riêng ứng với trị riêng 0; hãy đổi vector đầu"
            )
        raw_y = y.copy()
        x_new = normalize(y)
        raw_alignment = float(x_new @ x)
        if raw_alignment < 0.0:
            sign_flip_count += 1
            x_new = -x_new
        else:
            sign_flip_count = 0

        eigenvalue = float(x_new @ matrix @ x_new)
        residual = float(np.linalg.norm(matrix @ x_new - eigenvalue * x_new, 2))
        lambda_delta = (
            math.inf
            if previous_eigenvalue is None
            else abs(eigenvalue - previous_eigenvalue) / max(1.0, abs(eigenvalue))
        )
        delta = min(
            float(np.linalg.norm(x_new - x, 2)),
            float(np.linalg.norm(x_new + x, 2)),
        )
        relative_residual = residual / max(
            1.0, matrix_scale, abs(eigenvalue), np.finfo(float).tiny
        )
        final_relative_residual = relative_residual
        history.append(Iteration(k, eigenvalue, x_new.copy(), delta, residual, lambda_delta, relative_residual))
        x = x_new
        previous_eigenvalue = eigenvalue
        if fixed_steps == 0 and relative_residual <= epsilon:
            converged = True
            stop_reason = f"||Bv - λv||₂ <= ε = {epsilon:.3e}"
            break
        if fixed_steps == 0:
            if relative_residual < best_relative_residual * (1.0 - 1e-3):
                best_relative_residual = relative_residual
                no_improve_count = 0
            else:
                no_improve_count += 1
            if delta <= stagnation_tol and lambda_delta <= lambda_tol:
                stagnation_count += 1
            else:
                stagnation_count = 0
            if (
                stagnation_count >= stagnation_limit
                or no_improve_count >= no_improve_limit
                or sign_flip_count >= stagnation_limit
            ):
                stagnated = True
                converged = relative_residual <= max(epsilon, 100.0 * size * np.finfo(float).eps)
                if stagnation_count >= stagnation_limit:
                    detail = "vector va lambda gan nhu khong doi"
                elif sign_flip_count >= stagnation_limit:
                    detail = "phat hien dao dong dau lap lai"
                else:
                    detail = "residual khong cai thien qua nhieu buoc"
                stop_reason = (
                    "vector lặp không đổi thêm đáng kể qua nhiều bước; "
                    "dừng sớm và dùng kiểm tra residual cuối để chứng nhận"
                )
                stop_reason = f"{detail}; dung som an toan va dung kiem tra cuoi de chung nhan"
                break

    if not history:
        raise ArithmeticError("không tạo được bước lặp nào")
    if fixed_steps > 0:
        last_scale = max(1.0, matrix_scale, abs(history[-1].eigenvalue), np.finfo(float).tiny)
        final_relative_residual = history[-1].residual / last_scale
        converged = final_relative_residual <= epsilon
    elif not converged:
        max_iter_reached = len(history) >= maximum_iterations
        stop_reason = f"đã đạt k_max = {maximum_iterations} nhưng chưa đạt ε"
    last = history[-1]
    return PowerResult(
        last.eigenvalue, last.vector, history, converged, stop_reason,
        initial_vector=initial_used, raw_vector=raw_y,
        relative_residual=final_relative_residual,
        stagnated=stagnated,
        max_iter_reached=max_iter_reached,
        fixed_steps_completed=fixed_steps > 0 and len(history) >= fixed_steps,
    )


def deflate_symmetric(matrix: np.ndarray, eigenvalue: float, vector: np.ndarray) -> np.ndarray:
    result = matrix - eigenvalue * np.outer(vector, vector)
    result = 0.5 * (result + result.T)
    tolerance = 100.0 * np.finfo(float).eps * float(np.linalg.norm(matrix, "fro"))
    result[np.abs(result) <= tolerance] = 0.0
    return result


def final_svd_tolerances(
    epsilon: float,
    rows: int,
    columns: int,
    alpha: float,
    scaled_norm: float,
    *,
    multiplier: float = 10000.0,
) -> tuple[float, float]:
    norm_a = abs(alpha) * max(0.0, float(scaled_norm))
    if not math.isfinite(norm_a):
        norm_a = sys.float_info.max
    absolute = scaled_tolerance(
        epsilon,
        max(rows, columns),
        max(1.0, norm_a),
        multiplier=multiplier,
    )
    relative = absolute / max(1.0, norm_a)
    relative = max(relative, 10.0 * epsilon)
    absolute = max(absolute, relative * max(1.0, norm_a))
    return absolute, relative


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
    iter_tol_input = epsilon if iteration_tolerance is None else iteration_tolerance
    iter_tol = max(iter_tol_input, 100.0 * max(m, n) * eps_machine)
    final_abs_tol, final_rel_tol = final_svd_tolerances(epsilon, m, n, max(alpha, 1.0), 1.0)
    triplet_tol = max(iter_tol, final_rel_tol)
    orth_tol = final_rel_tol if orthogonality_tolerance is None else orthogonality_tolerance
    reconstruct_tol = final_rel_tol if reconstruction_tolerance is None else reconstruction_tolerance

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
            deflation_matrices=[np.zeros((n, n))],
            internal_power_converged=True,
            singular_triplet_residual_ok=True,
            orthogonality_ok=True,
            reconstruction_ok=True,
            final_accepted=True,
            final_absolute_tolerance=final_abs_tol,
            final_relative_tolerance=final_rel_tol,
            singular_triplet_tolerance=triplet_tol,
        )

    A_scaled = A / alpha
    scaled_norm = float(np.linalg.norm(A_scaled, "fro"))
    final_abs_tol, final_rel_tol = final_svd_tolerances(epsilon, m, n, alpha, scaled_norm)
    triplet_tol = max(iter_tol, final_rel_tol)
    orth_tol = final_rel_tol if orthogonality_tolerance is None else orthogonality_tolerance
    reconstruct_tol = final_rel_tol if reconstruction_tolerance is None else reconstruction_tolerance
    B = A_scaled.T @ A_scaled
    deflation_matrices = [B.copy()]
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
        # Xuống thang Hotelling thực sự trên ma trận đang lặp.  Việc khóa
        # vector chỉ là lớp bảo vệ số; B_i vẫn được cập nhật đúng công thức.
        deflation_value = max(float(power.eigenvalue), 0.0)
        B = deflate_symmetric(B, deflation_value, v)
        deflation_matrices.append(B.copy())

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
    for sigma, u, v in zip(singular_scaled, U_columns, V_columns):
        left_abs = float(np.linalg.norm(A_scaled @ v - sigma * u, 2))
        right_abs = float(np.linalg.norm(A_scaled.T @ u - sigma * v, 2))
        denominator = max(1.0, scaled_norm, sigma)
        relation_residuals.append((
            left_abs / max(denominator, safe_min),
            right_abs / max(denominator, safe_min),
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
    internal_power_converged = all(item[2].converged for item in resolved_pairs)
    triplets_ok = all(
        left <= triplet_tol and right <= triplet_tol
        for left, right in relation_residuals
    )
    orthogonality_ok = bool(left_error <= orth_tol and right_error <= orth_tol)
    reconstruction_ok = bool(
        absolute_error <= final_abs_tol
        and relative_error <= reconstruct_tol
        and relative_error <= final_rel_tol
    )
    final_accepted = bool(
        triplets_ok
        and orthogonality_ok
        and reconstruction_ok
    )
    if rank < min(m, n):
        warnings.append(
            "Hạng số theo tolerance bằng "
            f"{rank} (rank_tol={rank_tol_original:.3e}); đây không phải tuyên bố hạng toán học."
        )
    if final_accepted and not internal_power_converged:
        warnings.append(
            "Vòng lặp nội bộ đã chạm giới hạn bước, nhưng phân tích cuối vượt qua "
            "kiểm tra nên kết quả được chấp nhận."
        )
    if any(power.stagnated for _sigma, _v, power in resolved_pairs) and not final_accepted:
        warnings.append("Dung do dinh tre, ket qua chua dat kiem tra cuoi.")
    if not final_accepted:
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
        converged=final_accepted,
        warnings=warnings,
        rank_tolerance=rank_tol_original,
        iteration_tolerance=iter_tol,
        orthogonality_tolerance=orth_tol,
        reconstruction_tolerance=reconstruct_tol,
        scale_factor=alpha,
        power_results=[item[2] for item in resolved_pairs],
        deflation_matrices=deflation_matrices,
        internal_power_converged=internal_power_converged,
        singular_triplet_residual_ok=triplets_ok,
        orthogonality_ok=orthogonality_ok,
        reconstruction_ok=reconstruction_ok,
        final_accepted=final_accepted,
        final_absolute_tolerance=final_abs_tol,
        final_relative_tolerance=final_rel_tol,
        singular_triplet_tolerance=triplet_tol,
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
    if result.final_accepted:
        if not result.internal_power_converged:
            print(
                "KET LUAN BO SUNG: Ket qua SVD cuoi cung duoc chap nhan; "
                "mot vong lap noi bo cham gioi han buoc nhung cac kiem tra cuoi deu dat."
            )
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
    print_matrix("UᵣΣᵣVᵣᵀ", reconstruction, decimals)
    absolute_error = float(np.linalg.norm(A - reconstruction, "fro"))
    norm_a = float(np.linalg.norm(A, "fro"))
    relative_error = absolute_error / norm_a if norm_a > 0.0 else absolute_error
    print(f"\n‖A − UᵣΣᵣVᵣᵀ‖ꜰ = {scientific(absolute_error)}")
    print(f"Sai số tái tạo tương đối = {scientific(relative_error)}")
    print("\nKiểm tra trực chuẩn:")
    print(f"  ‖UᵣᵀUᵣ − I‖ꜰ = {scientific(np.linalg.norm(U.T @ U - np.eye(rank), 'fro'))}")
    print(f"  ‖VᵣᵀVᵣ − I‖ꜰ = {scientific(np.linalg.norm(V.T @ V - np.eye(rank), 'fro'))}")
    print("\nKiểm tra từng cặp kỳ dị:")
    for i, sigma in enumerate(np.diag(Sigma)):
        left = np.linalg.norm(A @ V[:, i] - sigma * U[:, i], 2)
        right = np.linalg.norm(A.T @ U[:, i] - sigma * V[:, i], 2)
        ui, vi, sigma_name = indexed("u", i + 1), indexed("v", i + 1), indexed("σ", i + 1)
        print(f"  i = {i + 1}: ‖A{vi} − {sigma_name}{ui}‖₂ = {scientific(left)}, ‖Aᵀ{ui} − {sigma_name}{vi}‖₂ = {scientific(right)}")
    print("\nKẾT LUẬN ĐỂ CHÉP VÀO BÀI:")
    print("  Các giá trị kỳ dị của A là:")
    print("  " + ", ".join(f"{indexed('σ', i + 1)} ≈ {number(value, decimals)}" for i, value in enumerate(singular_values)))
    print("  Các vector vᵢ ở từng bước đều đã được đưa về chuẩn 2 bằng 1.")
    print("  Mọi ma trận Bᵢ và vector dùng để xuống thang đã được ghi ở trên.")


# =============================================================================
# CHƯƠNG TRÌNH CHÍNH
# =============================================================================

def print_deflation_history(A: np.ndarray, result: SVDResult, decimals: int) -> None:
    """In B₀,…,Bᵣ của phép xuống thang thật sự đã dùng trong lõi."""
    matrices = result.deflation_matrices or []
    powers = result.power_results or []
    if not matrices:
        return
    section("C. LŨY THỪA VÀ CÁC MA TRẬN SAU MỖI BƯỚC XUỐNG THANG")
    print("Lõi tính trên Â=A/α để tránh tràn số; Bᵢ dưới đây được khôi phục về thang AᵀA.")
    square_scale = result.scale_factor * result.scale_factor
    can_unscale = math.isfinite(square_scale) and all(
        np.all(np.isfinite(matrix * square_scale)) for matrix in matrices
    )
    if not can_unscale:
        print("α² vượt miền số máy; vì vậy in B̂ᵢ=Bᵢ/α² và ghi rõ hệ số thang.")
        square_scale = 1.0
    print_matrix("B_0=A^T A" if can_unscale else "B_hat_0", matrices[0] * square_scale, decimals)
    for index, power in enumerate(powers, start=1):
        if index >= len(matrices):
            break
        print(f"\n--- Bước xuống thang {index} ---")
        if power.initial_vector is not None:
            print_vector(f"v_{index}^(0)", power.initial_vector, decimals, horizontal=True)
        print_iteration_table(power, decimals)
        if power.raw_vector is not None:
            print_vector(f"z_{index} trước chuẩn hóa", power.raw_vector, decimals, horizontal=True)
            print(f"  ||z_{index}||₂={np.linalg.norm(power.raw_vector):.{decimals}e}")
        vector = power.eigenvector
        eigenvalue = power.eigenvalue * (
            result.scale_factor * result.scale_factor if can_unscale else 1.0
        )
        sigma = result.singular_values[index - 1]
        print(f"  λ_{index}={eigenvalue:.{decimals}e}")
        print_vector(f"v_{index} sau chuẩn hóa", vector, decimals, horizontal=True)
        print(f"  ||v_{index}||₂={np.linalg.norm(vector):.{decimals}f}")
        print(f"  σ_{index}=√max(λ_{index},0)={sigma:.{decimals}f}")
        if sigma > result.rank_tolerance:
            left = A @ vector / sigma
            print_vector(f"u_{index}=Av_{index}/σ_{index}", left, decimals, horizontal=True)
        print(
            f"  B_{index}=B_{index - 1}-λ_{index}v_{index}v_{index}^T"
            if can_unscale
            else f"  B_hat_{index}=B_hat_{index - 1}-λ_hat_{index}v_{index}v_{index}^T"
        )
        print_matrix(
            f"B_{index}" if can_unscale else f"B_hat_{index}",
            matrices[index] * square_scale,
            decimals,
        )


def _svd_display_parts(
    A: np.ndarray,
    result: SVDResult,
    form: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Trả về U, Σ, Vᵀ đúng dạng cần in mà không làm đổi dữ liệu tính toán."""
    m, n = A.shape
    if form is None:
        form = "full" if result.U.shape == (m, m) and result.Vt.shape == (n, n) else "reduced"
    if form not in {"reduced", "full", "economy"}:
        raise ValueError("form phải là 'reduced', 'full' hoặc 'economy'.")
    if form == "full":
        U_view, Vt_view = result.U, result.Vt
        Sigma = np.zeros((m, n))
        label = "đầy đủ"
        formula = "A ≈ UΣVᵀ"
    elif form == "economy":
        p = min(m, n)
        U_view, Vt_view = result.U[:, :p], result.Vt[:p, :]
        Sigma = np.zeros((p, p))
        label = "mỏng với min(m,n) bộ ba"
        formula = "A ≈ UΣVᵀ"
    else:
        U_view, Vt_view = result.U, result.Vt
        Sigma = np.zeros((result.rank, result.rank))
        label = "rút gọn theo hạng số r"
        formula = "A ≈ UᵣΣᵣVᵣᵀ"
    for i, sigma in enumerate(result.singular_values[: min(Sigma.shape)]):
        Sigma[i, i] = sigma
    return U_view, Sigma, Vt_view, label, formula


def _print_svd_exam(
    A: np.ndarray,
    result: SVDResult,
    decimals: int,
    count: int | None,
    form: str | None,
) -> None:
    """Bản để chép bài thi: chỉ in công thức, kết quả và sai số kiểm tra."""
    section("D. KẾT QUẢ SVD VÀ KIỂM TRA")
    U_view, Sigma, Vt_view, label, formula = _svd_display_parts(A, result, form)
    shown = len(result.singular_values) if count is None else min(count, len(result.singular_values))

    print("Các giá trị kỳ dị thu được:")
    for i, sigma in enumerate(result.singular_values[:shown], start=1):
        print(f"  {indexed('σ', i)} ≈ {singular_text(sigma, decimals)}")

    print(f"\nDạng SVD dùng để trình bày: {label}.")
    print(f"{formula}")
    print(f"Kích thước: U={U_view.shape}, Σ={Sigma.shape}, Vᵀ={Vt_view.shape}.")
    print_matrix("Uᵣ" if form != "full" else "U", U_view, decimals)
    print_matrix("Σᵣ" if form != "full" else "Σ", Sigma, decimals)
    print_matrix("Vᵣᵀ" if form != "full" else "Vᵀ", Vt_view, decimals)

    reconstruction = U_view @ Sigma @ Vt_view
    print("\nMa trận tái tạo từ phân tích SVD:")
    print_matrix("UᵣΣᵣVᵣᵀ" if form != "full" else "UΣVᵀ", reconstruction, decimals)

    print("\nKiểm tra lại:")
    print(f"  ‖A − UᵣΣᵣVᵣᵀ‖ꜰ ≈ {scientific(result.reconstruction_error)}")
    print(f"  Sai số tái tạo tương đối ≈ {scientific(result.relative_reconstruction_error)}")
    print(f"  ‖UᵣᵀUᵣ − I‖ꜰ ≈ {scientific(result.left_orthogonality_error)}")
    print(f"  ‖VᵣᵀVᵣ − I‖ꜰ ≈ {scientific(result.right_orthogonality_error)}")

    if result.relation_residuals:
        print("\nKiểm tra từng bộ ba kỳ dị:")
        for i, (left, right) in enumerate(result.relation_residuals, start=1):
            ui, vi, sigma_name = indexed("u", i), indexed("v", i), indexed("σ", i)
            print(
                f"  i = {i}: ‖A{vi} − {sigma_name}{ui}‖₂ ≈ {scientific(left)}, "
                f"‖Aᵀ{ui} − {sigma_name}{vi}‖₂ ≈ {scientific(right)}"
            )

    print("\nKẾT LUẬN ĐỂ CHÉP VÀO BÀI:")
    if result.converged:
        print("  SVD đạt đồng thời các kiểm tra tái tạo, trực giao và bộ ba kỳ dị.")
        print(
            "  Các sai số kiểm tra đều rất nhỏ, do đó ta nhận được phân tích SVD "
            f"{label}: {formula}."
        )
    else:
        print(
            "  Sai số kiểm tra còn lớn, vì vậy kết quả SVD này chưa nên dùng để chép bài."
        )
    if result.warnings:
        print("\nGhi chú:")
        for warning in result.warnings:
            # Ở bản chép thi chỉ giữ các cảnh báo có ý nghĩa toán học, bỏ chi tiết kỹ thuật.
            if any(keyword in warning for keyword in ("rank_tol", "tolerance", "vòng lặp nội bộ")):
                continue
            print(f"  • {warning}")


def _print_svd_full(
    A: np.ndarray,
    result: SVDResult,
    decimals: int,
    count: int | None,
    form: str | None,
) -> None:
    """Bản đầy đủ: vẫn dễ đọc như bài giải, chỉ thêm ngưỡng kiểm tra ngắn gọn."""
    section("D. KẾT QUẢ SVD VÀ KIỂM TRA")
    U_view, Sigma, Vt_view, label, formula = _svd_display_parts(A, result, form)
    shown = len(result.singular_values) if count is None else min(count, len(result.singular_values))

    print("Các trị kỳ dị tìm được:")
    for i, sigma in enumerate(result.singular_values[:shown], start=1):
        print(f"  {indexed('σ', i)} = {sigma:.12e}")
    print(f"Hạng số dùng trong phân tích: r = {result.rank}.")
    print("Ngưỡng kiểm tra được chương trình chọn tự động theo kích thước và độ lớn của A.")
    print(f"  Ngưỡng kiểm tra tuyệt đối: εₖₜ ≈ {result.final_absolute_tolerance:.7e}")
    print(f"  Ngưỡng kiểm tra tương đối: εₖₜ,rel ≈ {result.final_relative_tolerance:.7e}")

    print(f"\nDạng SVD: {label}.")
    print(f"{formula}")
    print(f"Kích thước: U={U_view.shape}, Σ={Sigma.shape}, Vᵀ={Vt_view.shape}.")
    print_matrix("Uᵣ" if form != "full" else "U", U_view, decimals)
    print_matrix("Σᵣ" if form != "full" else "Σ", Sigma, decimals)
    print_matrix("Vᵣᵀ" if form != "full" else "Vᵀ", Vt_view, decimals)

    reconstruction = U_view @ Sigma @ Vt_view
    print_matrix("UᵣΣᵣVᵣᵀ" if form != "full" else "UΣVᵀ", reconstruction, decimals)

    print("\nKiểm tra tái tạo và trực giao:")
    print(f"  ‖A − UᵣΣᵣVᵣᵀ‖ꜰ ≈ {result.reconstruction_error:.7e}")
    print(f"  Sai số tái tạo tương đối ≈ {result.relative_reconstruction_error:.7e}")
    print(f"  ‖UᵣᵀUᵣ − I‖ꜰ ≈ {result.left_orthogonality_error:.7e}")
    print(f"  ‖VᵣᵀVᵣ − I‖ꜰ ≈ {result.right_orthogonality_error:.7e}")

    print("\nKiểm tra từng bộ ba kỳ dị:")
    for i, (left, right) in enumerate(result.relation_residuals, start=1):
        print(f"  i={i}: phần dư trái ≈ {left:.7e}; phần dư phải ≈ {right:.7e}.")

    if result.warnings:
        print("\nGhi chú:")
        for warning in result.warnings:
            print("  •", warning)
    if result.converged:
        print("\nKẾT LUẬN: Các kiểm tra cuối đều đạt, kết quả SVD được dùng để chép bài.")
    else:
        print("\nKẾT LUẬN: Sai số kiểm tra còn lớn, kết quả SVD chưa được chứng nhận.")


def _print_svd_debug(
    A: np.ndarray,
    result: SVDResult,
    decimals: int,
    count: int | None,
    form: str | None,
) -> None:
    """Bản kỹ thuật: giữ toàn bộ biến kiểm tra để debug code."""
    section("D. KẾT QUẢ SVD VÀ KIỂM TRA KỸ THUẬT")
    U_view, Sigma, Vt_view, label, _formula = _svd_display_parts(A, result, form)
    shown = len(result.singular_values) if count is None else min(count, len(result.singular_values))
    print(f"Hệ số đổi thang α = max|aᵢⱼ| = {result.scale_factor:.7e}.")
    print("Các trị kỳ dị có thể phân giải được (đã khôi phục về thang ban đầu):")
    for i, sigma in enumerate(result.singular_values[:shown], start=1):
        print(f"  {indexed('σ', i)} = {sigma:.12e}")
    print(f"Hạng số theo tolerance: {result.rank}; rank_tol={result.rank_tolerance:.7e}.")
    print("Không đồng nhất 'hạng số theo tolerance' với hạng toán học nếu dữ liệu kém điều kiện.")

    print(f"\nDạng SVD: {label}.")
    print(f"Kích thước: U={U_view.shape}, Σ={Sigma.shape}, V^T={Vt_view.shape}.")
    print_matrix("Uᵣ" if form != "full" else "U", U_view, decimals)
    print_matrix("Σᵣ" if form != "full" else "Σ", Sigma, decimals)
    print_matrix("Vᵣᵀ" if form != "full" else "Vᵀ", Vt_view, decimals)
    reconstruction = U_view @ Sigma @ Vt_view
    print_matrix("U Sigma V^T", reconstruction, decimals)

    print("\nKiểm tra từng bộ ba kỳ dị (phần dư tương đối):")
    for i, (left, right) in enumerate(result.relation_residuals, start=1):
        print(
            f"  i={i}: rel(Av-sigma*u)={left:.7e}; "
            f"rel(A^T u-sigma*v)={right:.7e}."
        )
    print(f"Tolerance residual bo ba ky di = {result.singular_triplet_tolerance:.7e}.")
    print(f"Tolerance kiem tra cuoi tuyet doi = {result.final_absolute_tolerance:.7e}.")
    print(f"Tolerance kiem tra cuoi tuong doi = {result.final_relative_tolerance:.7e}.")
    print(f"internal_power_converged={result.internal_power_converged}")
    print(f"singular_triplet_residual_ok={result.singular_triplet_residual_ok}")
    print(f"orthogonality_ok={result.orthogonality_ok}")
    print(f"reconstruction_ok={result.reconstruction_ok}")
    print(f"final_accepted={result.final_accepted}")
    print(f"||U^T U-I||_F={result.left_orthogonality_error:.7e} (tol={result.orthogonality_tolerance:.7e})")
    print(f"||V^T V-I||_F={result.right_orthogonality_error:.7e} (tol={result.orthogonality_tolerance:.7e})")
    print(f"||A-U Sigma V^T||_F={result.reconstruction_error:.7e} (tol={result.final_absolute_tolerance:.7e})")
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


def print_svd_result(
    A: np.ndarray,
    result: SVDResult,
    decimals: int,
    count: int | None = None,
    form: str | None = None,
    presentation_mode: str = "exam",
) -> None:
    """In kết quả SVD theo đúng chế độ trình bày đã chọn."""
    presentation_mode = (presentation_mode or "exam").strip().lower()
    if presentation_mode in {"exam", "chep_thi", "bai_thi", "1"}:
        _print_svd_exam(A, result, decimals, count, form)
    elif presentation_mode in {"full", "day_du", "2"}:
        _print_svd_full(A, result, decimals, count, form)
    elif presentation_mode in {"debug", "ky_thuat", "technical", "3"}:
        _print_svd_debug(A, result, decimals, count, form)
    else:
        raise ValueError("presentation_mode phải là exam, full hoặc debug.")

def main() -> None:
    print(LINE)
    print("TÌM GIÁ TRỊ KỲ DỊ BẰNG PHƯƠNG PHÁP LŨY THỪA VÀ XUỐNG THANG")
    print(LINE)
    print("Có thể nhập số nguyên, thập phân hoặc phân số a/b; Enter dùng giá trị mặc định.")

    m = input_integer("\nNhập số dòng m của A: ", 1)
    n = input_integer("Nhập số cột n của A: ", 1)
    A = input_matrix(m, n)
    initial = input_start_vector(n)

    print("\nDạng phân tích cần in:")
    print("  1. SVD rút gọn theo hạng r (mặc định trong bài thi)")
    print("  2. SVD đầy đủ")
    print("  3. SVD mỏng gồm min(m,n) bộ ba, kể cả σ gần 0")
    while True:
        form_choice = input("Chọn [Enter = 1]: ").strip() or "1"
        if form_choice in {"1", "2", "3"}:
            break
        print("  Lỗi: chỉ chọn 1, 2 hoặc 3.")
    form = {"1": "reduced", "2": "full", "3": "economy"}[form_choice]


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

    decimals = input_integer("Số chữ số sau dấu phẩy [Enter = 7]: ", 0, 7)

    print("\nChọn kiểu trình bày:")
    print("  1. Bản chép thi (gọn, không in thông số kỹ thuật)")
    print("  2. Bản đầy đủ (có thêm ngưỡng kiểm tra, vẫn dễ chép)")
    print("  3. Bản kỹ thuật/debug")
    try:
        presentation_choice = input("Chọn [Enter = 1]: ").strip() or "1"
    except (EOFError, StopIteration):
        presentation_choice = "1"
    while presentation_choice not in {"1", "2", "3"}:
        print("  Lỗi: chỉ chọn 1, 2 hoặc 3.")
        try:
            presentation_choice = input("Chọn [Enter = 1]: ").strip() or "1"
        except (EOFError, StopIteration):
            presentation_choice = "1"
    presentation_mode = {"1": "exam", "2": "full", "3": "debug"}[presentation_choice]

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
        full_matrices=form in {"full", "economy"},
    )
    print_deflation_history(A, result, decimals)
    print_svd_result(A, result, decimals, form=form, presentation_mode=presentation_mode)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except Exception as error:
        print(f"\nLỗi trong quá trình tính toán: {error}")
