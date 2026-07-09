"""
GIẢI HỆ AX = B VÀ TÌM A^(-1) BẰNG PHƯƠNG PHÁP LẶP JACOBI ĐỒNG THỜI.

Mục tiêu của chương trình:
- Chỉ dùng phép lặp Jacobi đồng thời, không dùng np.linalg.solve/inv.
- Tự kiểm tra chéo trội hàng/cột; nếu cần, tự hoán vị các phương trình.
- In bài giải theo đúng các ký hiệu trong tài liệu:
      T = diag(1/a_11, ..., 1/a_nn)
      α = I - T.A,  β = T.B
      q = ||α||_∞ (chéo trội hàng)
      q = ||α||_1, λ = max|a_ii|/min|a_ii| (chéo trội cột)
      X^(k) = α.X^(k-1) + β
- Hỗ trợ một hoặc nhiều vế phải và tìm nghịch đảo bằng cách giải AX = I.
"""

from __future__ import annotations

import math
from exam_format import exam_print as print
import sys
from input_utils import MathInputError, parse_real, split_number_row
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Bảo đảm tiếng Việt và ký hiệu toán học hiển thị trên Windows Terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# -----------------------------------------------------------------------------
# 1. NHẬP DỮ LIỆU AN TOÀN
# -----------------------------------------------------------------------------


def read_number(text: str) -> float:
    """Đọc số hữu hạn dạng nguyên, thập phân, khoa học hoặc phân số."""
    s = text.strip().replace("−", "-")
    if not s:
        raise ValueError("không được để trống")

    # Cho phép nhập số thập phân theo kiểu Việt Nam: 0,25.
    if "," in s and "." not in s and "/" not in s:
        s = s.replace(",", ".")

    try:
        value = parse_real(s)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("số không hợp lệ") from exc

    if not math.isfinite(value):
        raise ValueError("số phải hữu hạn")
    return value


def ask_number(
    prompt: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    integer: bool = False,
    default: Optional[float] = None,
) -> float | int:
    while True:
        suffix = f" [mặc định {default}]" if default is not None else ""
        raw = input(prompt + suffix + ": ").strip()
        if not raw and default is not None:
            value = float(default)
        else:
            try:
                value = read_number(raw)
            except ValueError as exc:
                print(f"Dữ liệu không hợp lệ: {exc}. Vui lòng nhập lại.")
                continue

        try:
            if integer and not float(value).is_integer():
                raise ValueError("phải là số nguyên")
            if positive and value <= 0:
                raise ValueError("phải dương")
            if nonnegative and value < 0:
                raise ValueError("không được âm")
        except ValueError as exc:
            print(f"Dữ liệu không hợp lệ: {exc}. Vui lòng nhập lại.")
            continue

        return int(value) if integer else float(value)


def ask_choice(
    prompt: str, choices: Sequence[str], default: Optional[str] = None
) -> int:
    allowed = set(choices)
    while True:
        suffix = f" [mặc định {default}]" if default is not None else ""
        raw = input(prompt + suffix + ": ").strip()
        if not raw and default is not None:
            raw = default
        if raw in allowed:
            return int(raw)
        print("Lựa chọn không hợp lệ. Vui lòng nhập lại.")


def ask_row(prompt: str, length: int) -> List[float]:
    while True:
        try:
            parts = split_number_row(input(prompt), length)
            return [read_number(part) for part in parts]
        except (MathInputError, ValueError) as exc:
            print(f"Dữ liệu không hợp lệ: {exc}. Vui lòng nhập lại.")


def ask_matrix(name: str, rows: int, cols: int) -> np.ndarray:
    print(f"Nhập {name} gồm {rows} hàng, mỗi hàng {cols} phần tử:")
    data = [ask_row(f"  Hàng {i + 1}: ", cols) for i in range(rows)]
    return np.asarray(data, dtype=float)


# -----------------------------------------------------------------------------
# 2. ĐỊNH DẠNG SỐ VÀ MA TRẬN
# -----------------------------------------------------------------------------


def _is_effectively_zero(value: float, scale: float = 1.0) -> bool:
    """Chỉ khử nhiễu ở mức máy, không biến số rất nhỏ hợp lệ thành 0."""
    if value == 0.0:
        return True
    machine_level = 32.0 * np.finfo(float).eps * max(scale, np.finfo(float).tiny)
    return abs(value) <= machine_level


def format_display_number(value: float, decimals: int = 6, scale: float = 1.0) -> str:
    """Hiển thị đẹp nhưng không làm các số rất nhỏ thành 0 giả."""
    value = float(value)
    if _is_effectively_zero(value, scale):
        return "0"

    abs_value = abs(value)
    decimals = max(0, int(decimals))

    # Với số quá nhỏ/lớn, dùng khoa học để không mất thông tin.
    if abs_value < 10.0 ** (-(decimals + 1)) or abs_value >= 10.0**8:
        return f"{value:.{max(6, decimals)}e}".replace("e+", "e")

    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "+0", "") else text


def format_formula_number(value: float, precision: int = 8) -> str:
    """Định dạng số trong công thức bằng thập phân."""
    value = float(value)
    if _is_effectively_zero(value, max(abs(value), np.finfo(float).tiny)):
        return "0"

    return f"{value:.{max(6, precision)}g}"


def matrix_lines(M: np.ndarray, decimals: int = 6) -> List[str]:
    M = np.atleast_2d(np.asarray(M, dtype=float))
    scale = float(np.max(np.abs(M))) if M.size else 1.0
    strings = [
        [format_display_number(value, decimals, scale) for value in row] for row in M
    ]
    widths = [
        max(len(strings[i][j]) for i in range(len(strings))) for j in range(M.shape[1])
    ]

    lines: List[str] = []
    for i, row in enumerate(strings):
        body = "  ".join(text.rjust(widths[j]) for j, text in enumerate(row))
        if M.shape[0] == 1:
            left, right = "[", "]"
        elif i == 0:
            left, right = "⎡", "⎤"
        elif i == M.shape[0] - 1:
            left, right = "⎣", "⎦"
        else:
            left, right = "⎢", "⎥"
        lines.append(f"{left} {body} {right}")
    return lines


def print_matrix(name: str, M: np.ndarray, decimals: int = 6) -> None:
    print(f"{name} =")
    for line in matrix_lines(M, decimals):
        print("  " + line)


def print_inline_matrix(M: np.ndarray, decimals: int = 6, indent: str = "    ") -> None:
    for line in matrix_lines(M, decimals):
        print(indent + line)


def _sig(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    return f"{value:.10g}"


# -----------------------------------------------------------------------------
# 3. CHÉO TRỘI VÀ HOÁN VỊ PHƯƠNG TRÌNH
# -----------------------------------------------------------------------------


def _diagonal_tolerance(A: np.ndarray) -> float:
    scale = float(np.max(np.abs(A))) if A.size else 0.0
    if scale == 0.0:
        return 0.0
    return 32.0 * np.finfo(float).eps * max(1, A.shape[0]) * scale


def row_dominance_details(
    A: np.ndarray,
) -> Tuple[bool, List[Tuple[float, float, bool]]]:
    details: List[Tuple[float, float, bool]] = []
    for i in range(A.shape[0]):
        diagonal = abs(float(A[i, i]))
        off_sum = float(np.sum(np.abs(A[i, :])) - diagonal)
        details.append((diagonal, off_sum, diagonal > off_sum))
    return all(item[2] for item in details), details


def column_dominance_details(
    A: np.ndarray,
) -> Tuple[bool, List[Tuple[float, float, bool]]]:
    details: List[Tuple[float, float, bool]] = []
    for j in range(A.shape[1]):
        diagonal = abs(float(A[j, j]))
        off_sum = float(np.sum(np.abs(A[:, j])) - diagonal)
        details.append((diagonal, off_sum, diagonal > off_sum))
    return all(item[2] for item in details), details


def _ratio_matrix_for_row_dominance(A: np.ndarray) -> np.ndarray:
    """ratio[position i, original row r] cho chéo trội hàng sau đổi hàng."""
    n = A.shape[0]
    ratios = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        for r in range(n):
            diagonal = abs(float(A[r, i]))
            if diagonal == 0.0:
                continue
            off_sum = float(np.sum(np.abs(A[r, :])) - diagonal)
            ratios[i, r] = off_sum / diagonal
    return ratios


def _ratio_matrix_for_column_dominance(A: np.ndarray) -> np.ndarray:
    """ratio[position i, original row r] cho chéo trội cột sau đổi hàng."""
    n = A.shape[0]
    ratios = np.full((n, n), np.inf, dtype=float)
    column_sums = np.sum(np.abs(A), axis=0)
    for i in range(n):
        for r in range(n):
            diagonal = abs(float(A[r, i]))
            if diagonal == 0.0:
                continue
            off_sum = float(column_sums[i] - diagonal)
            ratios[i, r] = off_sum / diagonal
    return ratios


def _perfect_matching(
    ratios: np.ndarray, threshold: float
) -> Optional[Tuple[int, ...]]:
    """Tìm hoán vị hàng sao cho mọi ratio <= threshold."""
    n = ratios.shape[0]
    candidates = [
        [
            r
            for r in range(n)
            if math.isfinite(ratios[i, r]) and ratios[i, r] <= threshold
        ]
        for i in range(n)
    ]
    if any(not rows for rows in candidates):
        return None

    # Xét trước vị trí có ít lựa chọn để tăng khả năng tìm được ghép hoàn hảo.
    order = sorted(range(n), key=lambda i: (len(candidates[i]), i))
    row_to_position = [-1] * n

    def augment(position: int, seen_rows: List[bool]) -> bool:
        ordered_rows = sorted(candidates[position], key=lambda r: ratios[position, r])
        for row in ordered_rows:
            if seen_rows[row]:
                continue
            seen_rows[row] = True
            previous_position = row_to_position[row]
            if previous_position == -1 or augment(previous_position, seen_rows):
                row_to_position[row] = position
                return True
        return False

    for position in order:
        if not augment(position, [False] * n):
            return None

    permutation = [-1] * n
    for row, position in enumerate(row_to_position):
        if position >= 0:
            permutation[position] = row
    if any(row < 0 for row in permutation):
        return None
    return tuple(permutation)


def _best_dominant_permutation(A: np.ndarray, kind: str) -> Optional[Tuple[int, ...]]:
    ratios = (
        _ratio_matrix_for_row_dominance(A)
        if kind == "row"
        else _ratio_matrix_for_column_dominance(A)
    )
    thresholds = sorted(
        {float(x) for x in ratios.ravel() if math.isfinite(x) and x < 1.0}
    )
    for threshold in thresholds:
        permutation = _perfect_matching(ratios, threshold)
        if permutation is not None:
            return permutation
    return None


def prepare_jacobi_system(
    A: np.ndarray,
    B: np.ndarray,
    dominance_preference: str = "auto",
) -> Dict[str, object]:
    """
    Ưu tiên đúng trình tự trong tài liệu:
    1) chéo trội hàng;
    2) nếu không, chéo trội cột;
    3) nếu cả hai không đạt, thử hoán vị các phương trình.
    """
    n = A.shape[0]
    identity_perm = tuple(range(n))
    row_ok, _ = row_dominance_details(A)
    col_ok, _ = column_dominance_details(A)

    if dominance_preference not in {"auto", "row", "column"}:
        raise ValueError("kiểu chéo trội phải là auto, row hoặc column")

    if dominance_preference == "row":
        permutation = identity_perm if row_ok else _best_dominant_permutation(A, "row")
        kind = "row"
    elif dominance_preference == "column":
        permutation = identity_perm if col_ok else _best_dominant_permutation(A, "column")
        kind = "column"
    elif row_ok:
        permutation, kind = identity_perm, "row"
    elif col_ok:
        permutation, kind = identity_perm, "column"
    else:
        permutation = _best_dominant_permutation(A, "row")
        kind = "row"
        if permutation is None:
            permutation = _best_dominant_permutation(A, "column")
            kind = "column"

    if permutation is None:
        return {
            "status": "not_dominant",
            "A_original": A.copy(),
            "B_original": B.copy(),
            "original_row_dominant": row_ok,
            "original_column_dominant": col_ok,
        }

    indices = list(permutation)
    A_work = A[indices, :].copy()
    B_work = B[indices, :].copy()

    diag_tol = _diagonal_tolerance(A_work)
    if np.any(np.abs(np.diag(A_work)) <= diag_tol):
        return {
            "status": "zero_diagonal",
            "A_original": A.copy(),
            "B_original": B.copy(),
            "permutation": permutation,
        }

    T = np.diag(1.0 / np.diag(A_work))
    alpha = np.eye(n) - T @ A_work
    beta = T @ B_work

    if kind == "row":
        q = float(np.max(np.sum(np.abs(alpha), axis=1)))
        lam = 1.0
        p = "inf"
        certificate_matrix = alpha
    else:
        # Theo đúng công thức trong đáp án mẫu: vẫn dùng ma trận lặp
        # α = -D^(-1)(L+U), lấy q = ||α||_1 và thêm hệ số
        # λ = max|a_ii|/min|a_ii| trong đánh giá sai số.
        q = float(np.max(np.sum(np.abs(alpha), axis=0)))
        diagonal_abs = np.abs(np.diag(A_work))
        lam = float(np.max(diagonal_abs) / np.min(diagonal_abs))
        p = "one"
        certificate_matrix = alpha

    # Do sai số dấu phẩy động, q có thể thành 1 - 1e-16 hoặc 1 + 1e-16.
    if not (q < 1.0):
        return {
            "status": "no_contraction",
            "A_original": A.copy(),
            "B_original": B.copy(),
            "A": A_work,
            "B": B_work,
            "permutation": permutation,
            "kind": kind,
            "q": q,
        }

    return {
        "status": "ready",
        "A_original": A.copy(),
        "B_original": B.copy(),
        "A": A_work,
        "B": B_work,
        "permutation": permutation,
        "original_row_dominant": row_ok,
        "original_column_dominant": col_ok,
        "kind": kind,
        "T": T,
        "alpha": alpha,
        "beta": beta,
        "certificate_matrix": certificate_matrix,
        "q": q,
        "lambda": lam,
        "p": p,
        "dominance_preference": dominance_preference,
        "error_factor": lam * q / (1.0 - q),
    }


# -----------------------------------------------------------------------------
# 4. LÕI LẶP JACOBI ĐỒNG THỜI
# -----------------------------------------------------------------------------


def column_norms(M: np.ndarray, p: str) -> np.ndarray:
    M = np.atleast_2d(np.asarray(M, dtype=float))
    if p == "inf":
        return np.max(np.abs(M), axis=0)
    if p == "one":
        return np.sum(np.abs(M), axis=0)
    raise ValueError("chuẩn không hợp lệ")


def _relative_error_bounds(
    X: np.ndarray, absolute_bounds: np.ndarray, p: str
) -> np.ndarray:
    x_norms = column_norms(X, p)
    result = np.full_like(absolute_bounds, np.inf, dtype=float)
    for j, bound in enumerate(absolute_bounds):
        if bound == 0.0:
            result[j] = 0.0
        elif x_norms[j] > bound:
            result[j] = bound / (x_norms[j] - bound)
    return result


def required_apriori_steps(
    q: float,
    lam: float,
    first_differences: np.ndarray,
    epsilon: float,
) -> Tuple[np.ndarray, int]:
    required = np.zeros(first_differences.shape, dtype=int)

    for j, s1 in enumerate(first_differences):
        if s1 == 0.0:
            required[j] = 0
            continue

        initial_bound = lam * s1 / (1.0 - q)
        if initial_bound <= epsilon:
            required[j] = 0
            continue

        if q == 0.0:
            required[j] = 1
            continue

        target = epsilon * (1.0 - q) / (lam * s1)
        if target >= 1.0:
            required[j] = 0
        elif target <= 0.0:
            required[j] = np.iinfo(np.int32).max
        else:
            value = math.log(target) / math.log(q)
            required[j] = max(0, int(math.ceil(value - 1e-14)))

    return required, int(np.max(required))


def jacobi_solve(
    A: np.ndarray,
    B: np.ndarray,
    X0: np.ndarray,
    *,
    stop_mode: str = "posteriori_absolute",
    epsilon: float = 1e-8,
    fixed_steps: int = 0,
    max_iter: int = 10000,
    dominance_preference: str = "auto",
) -> Dict[str, object]:
    """Lõi Jacobi: mọi phần tử của X^(k) dùng duy nhất X^(k-1)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    X0 = np.asarray(X0, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A phải là ma trận vuông")
    n = A.shape[0]
    if B.ndim == 1:
        B = B[:, None]
    if X0.ndim == 1:
        X0 = X0[:, None]
    if B.ndim != 2 or B.shape[0] != n or B.shape[1] == 0:
        raise ValueError("B phải có kích thước n×m, m ≥ 1")
    if X0.shape != B.shape:
        raise ValueError("X^(0) phải có cùng kích thước với B")
    if not all(np.all(np.isfinite(M)) for M in (A, B, X0)):
        raise ValueError("mọi dữ liệu phải hữu hạn")
    if max_iter <= 0:
        raise ValueError("max_iter phải dương")
    if (
        stop_mode in ("posteriori_absolute", "posteriori_relative", "apriori")
        and epsilon <= 0
    ):
        raise ValueError("ε phải dương")
    if fixed_steps < 0:
        raise ValueError("số bước không được âm")

    prepared = prepare_jacobi_system(A, B, dominance_preference)
    prepared.update({"X0": X0.copy(), "stop_mode": stop_mode, "epsilon": epsilon})
    if prepared["status"] != "ready":
        return prepared

    A_work = np.asarray(prepared["A"], dtype=float)
    B_work = np.asarray(prepared["B"], dtype=float)
    alpha = np.asarray(prepared["alpha"], dtype=float)
    beta = np.asarray(prepared["beta"], dtype=float)
    q = float(prepared["q"])
    lam = float(prepared["lambda"])
    p = str(prepared["p"])
    factor = float(prepared["error_factor"])

    history: List[Dict[str, object]] = []
    X = X0.copy()
    residual0 = column_norms(A_work @ X - B_work, p)
    history.append(
        {
            "k": 0,
            "X": X.copy(),
            "diff_norms": None,
            "error_bounds": None,
            "relative_bounds": None,
            "apriori_bounds": None,
            "residual_norms": residual0,
            "exact_fixed_point": bool(np.array_equal(alpha @ X + beta, X)),
        }
    )

    # X^(0) đã là điểm bất động.
    if bool(history[0]["exact_fixed_point"]) and stop_mode != "fixed":
        prepared.update(
            {
                "status": "converged_exact",
                "X": X,
                "history": history,
                "required_steps_by_column": np.zeros(B.shape[1], dtype=int),
                "target_steps": 0,
            }
        )
        return prepared

    target_steps: Optional[int] = None
    required_steps_by_column: Optional[np.ndarray] = None
    first_diff_norms: Optional[np.ndarray] = None

    if stop_mode == "fixed":
        target_steps = fixed_steps
    elif stop_mode == "apriori":
        X1 = alpha @ X0 + beta
        first_diff_norms = column_norms(X1 - X0, p)
        required_steps_by_column, target_steps = required_apriori_steps(
            q, lam, first_diff_norms, epsilon
        )
        history[0]["apriori_bounds"] = lam / (1.0 - q) * first_diff_norms
        if target_steps > max_iter:
            prepared.update(
                {
                    "status": "required_steps_exceed_max_iter",
                    "history": history,
                    "required_steps_by_column": required_steps_by_column,
                    "target_steps": target_steps,
                    "first_diff_norms": first_diff_norms,
                }
            )
            return prepared

    if target_steps == 0:
        prepared.update(
            {
                "status": "fixed_steps" if stop_mode == "fixed" else "converged",
                "X": X,
                "history": history,
                "required_steps_by_column": required_steps_by_column,
                "target_steps": 0,
                "first_diff_norms": first_diff_norms,
            }
        )
        return prepared

    status: Optional[str] = None
    k = 0
    while k < max_iter and status is None:
        # Jacobi đồng thời: toàn bộ X_next chỉ dùng X của vòng trước.
        X_next = alpha @ X + beta
        k += 1

        if not np.all(np.isfinite(X_next)):
            status = "invalid_value"
            break

        diff_norms = column_norms(X_next - X, p)
        error_bounds = factor * diff_norms
        if q == 0.0:
            error_bounds[:] = 0.0

        relative_bounds = _relative_error_bounds(X_next, error_bounds, p)
        residual_norms = column_norms(A_work @ X_next - B_work, p)
        exact_fixed_point = bool(np.array_equal(alpha @ X_next + beta, X_next))

        apriori_bounds = None
        if first_diff_norms is not None:
            apriori_bounds = lam * (q**k) / (1.0 - q) * first_diff_norms

        history.append(
            {
                "k": k,
                "X": X_next.copy(),
                "diff_norms": diff_norms.copy(),
                "error_bounds": error_bounds.copy(),
                "relative_bounds": relative_bounds.copy(),
                "apriori_bounds": None
                if apriori_bounds is None
                else apriori_bounds.copy(),
                "residual_norms": residual_norms.copy(),
                "exact_fixed_point": exact_fixed_point,
            }
        )
        X = X_next

        if stop_mode == "fixed" and k >= int(target_steps):
            status = "fixed_steps"
        elif exact_fixed_point and stop_mode != "fixed":
            status = "converged_exact"
        elif stop_mode == "apriori" and k >= int(target_steps):
            status = "converged"
        elif (
            stop_mode == "posteriori_absolute"
            and float(np.max(error_bounds)) <= epsilon
        ):
            status = "converged"
        elif stop_mode == "posteriori_relative":
            max_relative = float(np.max(relative_bounds))
            if math.isfinite(max_relative) and max_relative <= epsilon:
                status = "converged"

    if status is None:
        status = "max_iter"

    prepared.update(
        {
            "status": status,
            "X": X,
            "history": history,
            "required_steps_by_column": required_steps_by_column,
            "target_steps": target_steps,
            "first_diff_norms": first_diff_norms,
        }
    )
    return prepared


def jacobi_fixed_point(
    B: np.ndarray,
    d: np.ndarray,
    x0: Optional[np.ndarray] = None,
    *,
    stop_mode: str = "epsilon",
    epsilon: float = 1e-8,
    fixed_steps: int = 0,
    max_iter: int = 10000,
    norm_kind: str = "inf",
) -> Dict[str, object]:
    """Jacobi trực tiếp cho x^(k+1)=B x^(k)+d, không chuyển về Ax=b."""
    B = np.asarray(B, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)
    if B.ndim != 2 or B.shape[0] != B.shape[1] or B.shape[0] == 0:
        raise ValueError("B phải là ma trận vuông khác rỗng")
    n = B.shape[0]
    if d.shape != (n,):
        raise ValueError("d phải là vector có đúng n phần tử")
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1)
    if x.shape != (n,):
        raise ValueError("x^(0) phải là vector có đúng n phần tử")
    if not all(np.all(np.isfinite(item)) for item in (B, d, x)):
        raise ValueError("mọi dữ liệu phải hữu hạn")
    if norm_kind not in {"inf", "one", "two"}:
        raise ValueError("chuẩn phải là inf, one hoặc two")
    if stop_mode not in {"epsilon", "fixed"}:
        raise ValueError("stop_mode phải là epsilon hoặc fixed")
    if stop_mode == "epsilon" and epsilon <= 0:
        raise ValueError("ε phải dương")
    if fixed_steps < 0 or max_iter <= 0:
        raise ValueError("số bước và max_iter không hợp lệ")

    order = {"inf": np.inf, "one": 1, "two": 2}[norm_kind]
    q = float(np.linalg.norm(B, order))

    def residual(vector: np.ndarray) -> np.ndarray:
        return vector - B @ vector - d

    history: List[Dict[str, object]] = [
        {
            "k": 0,
            "x": x.copy(),
            "diff_norm": None,
            "error_bound": None,
            "residual_norm": float(np.linalg.norm(residual(x), order)),
        }
    ]
    target = fixed_steps if stop_mode == "fixed" else max_iter
    status = "max_iter"
    certified = False

    if stop_mode == "epsilon" and history[0]["residual_norm"] <= epsilon:
        return {
            "status": "converged_exact",
            "certified": True,
            "x": x,
            "history": history,
            "q": q,
            "norm_kind": norm_kind,
            "epsilon": epsilon,
            "fixed_steps": fixed_steps,
        }

    for k in range(1, target + 1):
        x_next = B @ x + d
        if not np.all(np.isfinite(x_next)):
            status = "invalid_value"
            break
        diff_norm = float(np.linalg.norm(x_next - x, order))
        residual_norm = float(np.linalg.norm(residual(x_next), order))
        error_bound = q / (1.0 - q) * diff_norm if q < 1.0 else math.inf
        history.append(
            {
                "k": k,
                "x": x_next.copy(),
                "diff_norm": diff_norm,
                "error_bound": error_bound,
                "residual_norm": residual_norm,
            }
        )
        x = x_next
        if stop_mode == "fixed":
            if k >= fixed_steps:
                status = "fixed_steps"
                certified = q < 1.0 and error_bound <= epsilon and residual_norm <= epsilon
                break
            continue
        if q < 1.0 and error_bound <= epsilon and residual_norm <= epsilon:
            status = "converged"
            certified = True
            break

    return {
        "status": status,
        "certified": certified,
        "x": x,
        "history": history,
        "q": q,
        "norm_kind": norm_kind,
        "epsilon": epsilon,
        "fixed_steps": fixed_steps,
    }


# -----------------------------------------------------------------------------
# 5. IN BÀI GIẢI ĐÚNG KÝ HIỆU TRONG TÀI LIỆU
# -----------------------------------------------------------------------------


def _norm_symbol(p: str) -> str:
    return "∞" if p == "inf" else "1"


def _dominance_name(kind: str) -> str:
    return "chéo trội hàng" if kind == "row" else "chéo trội cột"


def _print_dominance_details(A: np.ndarray, kind: str) -> None:
    if kind == "row":
        _, details = row_dominance_details(A)
        for i, (diagonal, off_sum, ok) in enumerate(details, start=1):
            sign = ">" if ok else "≤"
            print(
                f"  Hàng {i}: |a_{i}{i}| = {_sig(diagonal)} {sign} "
                f"Σ_(j≠{i})|a_{i}j| = {_sig(off_sum)}."
            )
    else:
        _, details = column_dominance_details(A)
        for j, (diagonal, off_sum, ok) in enumerate(details, start=1):
            sign = ">" if ok else "≤"
            print(
                f"  Cột {j}: |a_{j}{j}| = {_sig(diagonal)} {sign} "
                f"Σ_(i≠{j})|a_i{j}| = {_sig(off_sum)}."
            )


def _formula_term(coefficient: float, variable: str, precision: int) -> str:
    magnitude = format_formula_number(abs(coefficient), precision)
    if magnitude == "1":
        body = variable
    elif "/" in magnitude:
        body = f"({magnitude}){variable}"
    else:
        body = f"{magnitude}{variable}"
    return (" + " if coefficient > 0 else " - ") + body


def numeric_formula_lines(
    alpha: np.ndarray, beta: np.ndarray, precision: int
) -> List[str]:
    n, m = beta.shape
    lines: List[str] = []
    if m == 1:
        for i in range(n):
            line = f"x_{i + 1}^(k) = {format_formula_number(beta[i, 0], precision)}"
            for j in range(n):
                coefficient = float(alpha[i, j])
                if _is_effectively_zero(
                    coefficient, max(float(np.max(np.abs(alpha))), np.finfo(float).tiny)
                ):
                    continue
                line += _formula_term(coefficient, f"x_{j + 1}^(k-1)", precision)
            lines.append(line)
    else:
        for i in range(n):
            beta_row = (
                "["
                + ", ".join(
                    format_formula_number(value, precision) for value in beta[i, :]
                )
                + "]"
            )
            line = f"X_({i + 1},:)^(k) = {beta_row}"
            for j in range(n):
                coefficient = float(alpha[i, j])
                if _is_effectively_zero(
                    coefficient, max(float(np.max(np.abs(alpha))), np.finfo(float).tiny)
                ):
                    continue
                line += _formula_term(coefficient, f"X_({j + 1},:)^(k-1)", precision)
            lines.append(line)
    return lines


def _component_substitution_lines(
    alpha: np.ndarray,
    beta: np.ndarray,
    X_previous: np.ndarray,
    X_current: np.ndarray,
    k: int,
    precision: int,
) -> List[str]:
    """In phép thay số từng ẩn khi B chỉ có một cột."""
    n = alpha.shape[0]
    lines: List[str] = []
    for i in range(n):
        expression = format_formula_number(beta[i, 0], precision)
        for j in range(n):
            coefficient = float(alpha[i, j])
            if _is_effectively_zero(
                coefficient, max(float(np.max(np.abs(alpha))), np.finfo(float).tiny)
            ):
                continue
            coeff_text = format_formula_number(abs(coefficient), precision)
            x_text = format_display_number(
                X_previous[j, 0],
                precision,
                max(float(np.max(np.abs(X_previous))), np.finfo(float).tiny),
            )
            term = f"{coeff_text}·({x_text})"
            expression += (" + " if coefficient > 0 else " - ") + term
        result_text = format_display_number(
            X_current[i, 0],
            precision,
            max(float(np.max(np.abs(X_current))), np.finfo(float).tiny),
        )
        lines.append(f"x_{i + 1}^({k}) = {expression} = {result_text}")
    return lines


def _print_stop_condition(result: Dict[str, object]) -> None:
    mode = str(result["stop_mode"])
    q = float(result["q"])
    lam = float(result["lambda"])
    p_symbol = _norm_symbol(str(result["p"]))
    epsilon = float(result["epsilon"])

    if mode == "posteriori_absolute":
        print(
            f"Dùng công thức sai số hậu nghiệm: E_k,j = "
            f"λq/(1-q)·‖X_j^(k)-X_j^(k-1)‖_{p_symbol}."
        )
        print(f"Dừng khi max_j E_k,j ≤ ε = {_sig(epsilon)}.")
    elif mode == "posteriori_relative":
        print(f"Trước hết E_k,j = λq/(1-q)·‖X_j^(k)-X_j^(k-1)‖_{p_symbol}.")
        print(f"Chặn sai số tương đối: r_k,j ≤ E_k,j / (‖X_j^(k)‖_{p_symbol} - E_k,j).")
        print(f"Dừng khi max_j r_k,j ≤ ε = {_sig(epsilon)}.")
    elif mode == "apriori":
        print(
            f"Dùng công thức sai số tiên nghiệm: E_k,j ≤ "
            f"λq^k/(1-q)·‖X_j^(1)-X_j^(0)‖_{p_symbol}."
        )
        first = np.asarray(result["first_diff_norms"], dtype=float)
        required = np.asarray(result["required_steps_by_column"], dtype=int)
        for j in range(len(first)):
            print(
                f"  Cột {j + 1}: ‖X_{j + 1}^(1)-X_{j + 1}^(0)‖_{p_symbol} "
                f"= {_sig(first[j])} ⇒ k ≥ {required[j]}."
            )
        print(f"Chọn k = max_j k_j = {int(result['target_steps'])}.")
    else:
        print(f"Đề yêu cầu thực hiện đúng k = {int(result['target_steps'])} bước lặp.")
        return

    print(
        f"Ở đây λ = {_sig(lam)}, q = {_sig(q)}, "
        f"λq/(1-q) = {_sig(float(result['error_factor']))}."
    )


def print_iteration_table(
    history: List[Dict[str, object]], result: Dict[str, object], decimals: int
) -> None:
    """In toàn bộ quá trình lặp Jacobi thành một bảng dễ so sánh."""
    if not history:
        return
    first_x = np.asarray(history[0]["X"], dtype=float)
    n, rhs_count = first_x.shape
    x_headers = (
        [f"x{i + 1}" for i in range(n)]
        if rhs_count == 1
        else [f"x{i + 1},{j + 1}" for j in range(rhs_count) for i in range(n)]
    )
    headers = ["k"] + x_headers + ["||ΔX||", "Sai số", "Phần dư"]
    rows: List[List[str]] = []
    mode = str(result["stop_mode"])

    def vector_text(vector: np.ndarray) -> str:
        return ", ".join(_sig(value) for value in vector)

    for item in history:
        X = np.asarray(item["X"], dtype=float)
        scale = max(float(np.max(np.abs(X))), np.finfo(float).tiny)
        values = [
            format_display_number(X[i, j], decimals, scale)
            for j in range(rhs_count)
            for i in range(n)
        ]
        if int(item["k"]) == 0:
            metrics = ["—", "—", "—"]
        else:
            diff = np.asarray(item["diff_norms"], dtype=float)
            residual = np.asarray(item["residual_norms"], dtype=float)
            if mode == "posteriori_relative":
                error = np.asarray(item["relative_bounds"], dtype=float)
            elif mode == "apriori" and item["apriori_bounds"] is not None:
                error = np.asarray(item["apriori_bounds"], dtype=float)
            else:
                error = np.asarray(item["error_bounds"], dtype=float)
            metrics = [vector_text(diff), vector_text(error), vector_text(residual)]
        rows.append([str(int(item["k"]))] + values + metrics)

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(separator)
    print("| " + " | ".join(headers[i].center(widths[i]) for i in range(len(headers))) + " |")
    print(separator)
    for row in rows:
        print("| " + " | ".join(row[i].rjust(widths[i]) for i in range(len(row))) + " |")
    print(separator)
    if rhs_count > 1:
        print("Ghi chú: chuẩn, sai số và phần dư được liệt kê theo từng cột vế phải.")


def print_solution(
    result: Dict[str, object],
    *,
    task_mode: int,
    decimals: int,
    decimal_digits_request: Optional[int] = None,
) -> None:
    line = "=" * 78
    print("\n" + line)
    print("BÀI GIẢI BẰNG PHƯƠNG PHÁP LẶP JACOBI")
    print(line)

    A0 = np.asarray(result["A_original"], dtype=float)
    B0 = np.asarray(result["B_original"], dtype=float)
    X0 = np.asarray(result["X0"], dtype=float)
    n, m = B0.shape

    print("\n1. DỮ KIỆN")
    if task_mode == 2:
        print("Để tìm A^(-1), giải đồng thời hệ AX = I_n. Khi đó X = A^(-1).")
    else:
        print("Xét hệ phương trình AX = B.")
    print_matrix("A", A0, decimals)
    print_matrix("B" if task_mode == 1 else "I_n", B0, decimals)
    print_matrix("X^(0)", X0, decimals)
    if decimal_digits_request is not None:
        print(
            f"Yêu cầu {decimal_digits_request} chữ số thập phân tin cậy nên chọn "
            f"ε = 0.5·10^(-{decimal_digits_request}) = {_sig(float(result['epsilon']))}."
        )

    status = str(result["status"])
    if status in ("not_dominant", "zero_diagonal", "no_contraction"):
        print("\n2. KIỂM TRA ĐIỀU KIỆN THỰC HIỆN")
        row_ok, _ = row_dominance_details(A0)
        col_ok, _ = column_dominance_details(A0)
        print(f"A chéo trội hàng: {'Có' if row_ok else 'Không'}.")
        print(f"A chéo trội cột: {'Có' if col_ok else 'Không'}.")
        if status == "zero_diagonal":
            print("Sau khi hoán vị vẫn xuất hiện phần tử đường chéo bằng 0.")
        elif status == "no_contraction":
            print(f"Hệ số co tính được q = {_sig(float(result['q']))} ≥ 1.")
        else:
            print("Không tìm được hoán vị phương trình để A chéo trội hàng hoặc cột.")
        print(
            "Kết luận: không đủ điều kiện bảo đảm hội tụ theo phương pháp trong tài liệu."
        )
        print("Chương trình dừng và không xác nhận nghiệm.")
        return

    A = np.asarray(result["A"], dtype=float)
    B = np.asarray(result["B"], dtype=float)
    T = np.asarray(result["T"], dtype=float)
    alpha = np.asarray(result["alpha"], dtype=float)
    beta = np.asarray(result["beta"], dtype=float)
    permutation = tuple(result["permutation"])
    kind = str(result["kind"])
    p_symbol = _norm_symbol(str(result["p"]))
    q = float(result["q"])
    lam = float(result["lambda"])

    print("\n2. KIỂM TRA VÀ ĐƯA HỆ VỀ DẠNG LẶP")
    if permutation == tuple(range(n)):
        print("Không cần đổi thứ tự các phương trình.")
    else:
        print(
            "A ban đầu chưa chéo trội theo đường chéo. Đổi thứ tự các phương trình "
            f"theo hoán vị {tuple(index + 1 for index in permutation)}."
        )
        print("Hệ tương đương sau khi đổi hàng:")
        print_matrix("A'", A, decimals)
        print_matrix("B'", B, decimals)

    print(f"Kiểm tra {_dominance_name(kind)}:")
    _print_dominance_details(A, kind)
    print(
        f"⇒ A{"'" if permutation != tuple(range(n)) else ''} là ma trận {_dominance_name(kind)}."
    )

    print("\nĐặt")
    print("  T = diag(1/a_11, 1/a_22, ..., 1/a_nn),")
    print("  α = I - T.A,    β = T.B.")
    print_matrix("T", T, decimals)
    print_matrix("α = I - T.A", alpha, decimals)
    print_matrix("β = T.B", beta, decimals)
    print("Theo ký hiệu thường dùng trong bài thi:")
    print_matrix("M_J = α = -D^(-1)(L+U)", alpha, decimals)
    print_matrix("d = β = D^(-1)B", beta, decimals)

    print("\n3. KIỂM TRA HỘI TỤ")
    if kind == "row":
        row_sums = np.sum(np.abs(alpha), axis=1)
        print("Vì A chéo trội hàng nên chọn p = ∞, λ = 1.")
        print("q = ‖α‖_∞ = max_i Σ_j|α_ij|.")
        print("Các tổng hàng của |α|: " + ", ".join(_sig(x) for x in row_sums) + ".")
    else:
        column_sums = np.sum(np.abs(alpha), axis=0)
        print("Vì A chéo trội cột nên chọn p = 1.")
        print("Với M_J = α = -D^(-1)(L+U), ta có")
        print("q = ‖M_J‖_1 = max_j Σ_i|(M_J)_ij|.")
        print(
            "Các tổng cột của |M_J|: " + ", ".join(_sig(x) for x in column_sums) + "."
        )
        print("λ = max_i|a_ii| / min_i|a_ii|.")

    print(f"q = {_sig(q)} < 1; λ = {_sig(lam)}.")
    print("⇒ Phép lặp Jacobi hội tụ tới nghiệm duy nhất với mọi X^(0).")

    print("\n4. CÔNG THỨC LẶP ĐÃ THAY SỐ")
    print("X^(k) = α.X^(k-1) + β,    k = 1, 2, ...")
    for formula in numeric_formula_lines(alpha, beta, max(decimals, 8)):
        print("  " + formula)

    print("\n5. ĐIỀU KIỆN DỪNG")
    _print_stop_condition(result)

    print("\n6. CÁC BƯỚC LẶP")
    history: List[Dict[str, object]] = list(result["history"])
    print_iteration_table(history, result, decimals)
    # In lại từng bước dưới dạng thế số để có thể chép thẳng vào bài thi.
    # Trước đây đoạn này vô tình duyệt enumerate([]), nên toàn bộ phần diễn
    # giải chi tiết không bao giờ được hiển thị.
    for index, row in enumerate(history):
        k = int(row["k"])
        X = np.asarray(row["X"], dtype=float)
        print(f"\nBước lặp k = {k}:")
        if k > 0 and m == 1:
            X_previous = np.asarray(history[index - 1]["X"], dtype=float)
            for substitution in _component_substitution_lines(
                alpha, beta, X_previous, X, k, max(decimals, 8)
            ):
                print("  " + substitution)
        print_matrix(f"X^({k})", X, decimals)

        if k == 0:
            continue

        diff_norms = np.asarray(row["diff_norms"], dtype=float)
        error_bounds = np.asarray(row["error_bounds"], dtype=float)
        relative_bounds = np.asarray(row["relative_bounds"], dtype=float)
        residual_norms = np.asarray(row["residual_norms"], dtype=float)
        apriori_bounds = row["apriori_bounds"]

        for j in range(m):
            pieces = [f"‖ΔX_{j + 1}‖_{p_symbol} = {_sig(diff_norms[j])}"]
            stop_mode = str(result["stop_mode"])
            if stop_mode == "posteriori_absolute":
                pieces.append(f"E_{k},{j + 1} = {_sig(error_bounds[j])}")
            elif stop_mode == "posteriori_relative":
                rel_text = (
                    _sig(relative_bounds[j])
                    if math.isfinite(relative_bounds[j])
                    else "chưa xác định"
                )
                pieces.append(f"E_{k},{j + 1} = {_sig(error_bounds[j])}")
                pieces.append(f"r_{k},{j + 1} = {rel_text}")
            elif stop_mode == "apriori" and apriori_bounds is not None:
                prior = np.asarray(apriori_bounds, dtype=float)
                pieces.append(f"E_tiên nghiệm ≤ {_sig(prior[j])}")
            pieces.append(
                f"‖A.X_(:,{j + 1})^({k})-B_(:,{j + 1})‖_{p_symbol} = {_sig(residual_norms[j])}"
            )
            print("  Cột " + str(j + 1) + ": " + "; ".join(pieces) + ".")

        if str(result["stop_mode"]) == "posteriori_absolute":
            print(f"  max_j E_{k},j = {_sig(float(np.max(error_bounds)))}.")
        elif str(result["stop_mode"]) == "posteriori_relative":
            max_rel = float(np.max(relative_bounds))
            text = _sig(max_rel) if math.isfinite(max_rel) else "chưa xác định"
            print(f"  max_j r_{k},j = {text}.")

    last = history[-1]
    last_k = int(last["k"])
    print("\n7. KIỂM TRA DỪNG VÀ KẾT LUẬN")
    if status == "converged_exact":
        print(f"X^({last_k}) là điểm bất động: α.X^({last_k}) + β = X^({last_k}).")
        print("Do đó đã thu được nghiệm đúng trong số học máy.")
    elif status == "converged":
        stop_mode = str(result["stop_mode"])
        if stop_mode == "posteriori_absolute":
            last_error = float(np.max(np.asarray(last["error_bounds"], dtype=float)))
            print(
                f"max_j E_{last_k},j = {_sig(last_error)} ≤ ε = {_sig(float(result['epsilon']))}."
            )
        elif stop_mode == "posteriori_relative":
            last_relative = float(
                np.max(np.asarray(last["relative_bounds"], dtype=float))
            )
            print(
                f"max_j r_{last_k},j = {_sig(last_relative)} ≤ ε = {_sig(float(result['epsilon']))}."
            )
        else:
            prior = np.asarray(last["apriori_bounds"], dtype=float)
            print(
                f"Sau k = {last_k}, max_j E_tiên nghiệm ≤ "
                f"{_sig(float(np.max(prior)))} ≤ ε = {_sig(float(result['epsilon']))}."
            )
    elif status == "fixed_steps":
        print(f"Đã thực hiện đúng {last_k} bước lặp theo yêu cầu của đề.")
        if last_k > 0:
            last_error = np.asarray(last["error_bounds"], dtype=float)
            print(
                f"Ước lượng sai số hậu nghiệm: E_{last_k} ≤ "
                f"λq/(1-q)·‖X^({last_k})-X^({last_k - 1})‖_{p_symbol}."
            )
            if m == 1:
                print(f"Suy ra E_{last_k} ≤ {_sig(float(last_error[0]))}.")
            else:
                print(
                    "Theo từng cột: "
                    + ", ".join(
                        f"E_{last_k},{j + 1} ≤ {_sig(value)}"
                        for j, value in enumerate(last_error)
                    )
                    + "."
                )
    elif status == "required_steps_exceed_max_iter":
        print(
            f"Công thức tiên nghiệm yêu cầu k = {int(result['target_steps'])} > max_iter. "
            "Chưa thực hiện phép lặp."
        )
        return
    elif status == "max_iter":
        print("Đã đạt max_iter nhưng chưa thỏa điều kiện dừng.")
        print("Không xác nhận nghiệm với độ chính xác đã yêu cầu.")
        return
    else:
        print("Xuất hiện giá trị số không hợp lệ trong quá trình lặp.")
        print("Không xác nhận nghiệm.")
        return

    X_final = np.asarray(result["X"], dtype=float)
    if task_mode == 2:
        inverse_label = (
            f"A^(-1) ≈ X^({last_k})"
            if status != "fixed_steps"
            else f"A^(-1)_xấp_xỉ sau {last_k} bước = X^({last_k})"
        )
        print_matrix(inverse_label, X_final, decimals)
        identity = np.eye(n)
        left = A0 @ X_final
        right = X_final @ A0
        print("Kiểm tra:")
        print_matrix("A.A^(-1)_xấp_xỉ", left, decimals)
        print_matrix("A^(-1)_xấp_xỉ.A", right, decimals)
        left_error = float(np.max(np.sum(np.abs(left - identity), axis=1)))
        right_error = float(np.max(np.sum(np.abs(right - identity), axis=1)))
        print(f"‖A.A^(-1)_xấp_xỉ-I‖_∞ = {_sig(left_error)}.")
        print(f"‖A^(-1)_xấp_xỉ.A-I‖_∞ = {_sig(right_error)}.")
    else:
        solution_label = (
            f"X* ≈ X^({last_k})"
            if status != "fixed_steps"
            else f"Giá trị xấp xỉ sau đúng {last_k} bước: X^({last_k})"
        )
        print_matrix(solution_label, X_final, decimals)


def print_direct_jacobi_result(
    B: np.ndarray,
    d: np.ndarray,
    result: Dict[str, object],
    decimals: int,
) -> None:
    B = np.asarray(B, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)
    history: List[Dict[str, object]] = list(result["history"])
    norm_kind = str(result["norm_kind"])
    p_symbol = {"inf": "∞", "one": "1", "two": "2"}[norm_kind]
    q = float(result["q"])

    print("\n1. JACOBI TRỰC TIẾP CHO x^(k+1)=B x^(k)+d")
    print("Không chuyển bài toán về Ax=b; mọi thành phần của x^(k+1) dùng x^(k).")
    print_matrix("B", B, decimals)
    print_matrix("d", d[:, None], decimals)
    print("Công thức tổng quát:")
    print("  x_i^(k+1) = d_i + Σ_(j=1..n) b_ij x_j^(k),  i=1..n.")
    print(f"Chuẩn kiểm tra: ‖.‖_{p_symbol}; q = ‖B‖_{p_symbol} = {_sig(q)}.")
    if q < 1.0:
        print("Vì q < 1 nên phép lặp được chứng nhận co trong chuẩn đã chọn.")
    else:
        print("q ≥ 1 nên chưa có chứng nhận co; chỉ báo cáo kết quả và residual.")

    print("\n2. CÁC BƯỚC LẶP")
    headers = ["k"] + [f"x_{i + 1}^(k)" for i in range(B.shape[0])] + [
        f"‖x^(k)-x^(k-1)‖_{p_symbol}",
        f"‖x^(k)-B x^(k)-d‖_{p_symbol}",
    ]
    rows: List[List[str]] = []
    for item in history:
        x = np.asarray(item["x"], dtype=float).reshape(-1)
        diff = item["diff_norm"]
        residual = float(item["residual_norm"])
        rows.append(
            [
                str(int(item["k"])),
                *[format_display_number(value, decimals) for value in x],
                "-" if diff is None else _sig(float(diff)),
                _sig(residual),
            ]
        )
    widths = [max(len(headers[j]), *(len(row[j]) for row in rows)) for j in range(len(headers))]
    print("  " + " | ".join(headers[j].rjust(widths[j]) for j in range(len(headers))))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print("  " + " | ".join(row[j].rjust(widths[j]) for j in range(len(row))))

    last = history[-1]
    last_k = int(last["k"])
    print("\n3. KẾT LUẬN")
    print_matrix(f"x^({last_k})", np.asarray(last["x"], dtype=float)[:, None], decimals)
    print(f"Residual điểm bất động = {_sig(float(last['residual_norm']))}.")
    status = str(result["status"])
    if status == "fixed_steps":
        print(f"KẾT LUẬN: đã thực hiện đúng {last_k} bước theo yêu cầu.")
    elif bool(result["certified"]):
        print(
            f"KẾT LUẬN: đạt ε = {_sig(float(result['epsilon']))} "
            "và được chứng nhận nhờ q < 1."
        )
    elif status == "converged_exact":
        print("KẾT LUẬN: x^(0) đã là điểm bất động trong số học máy.")
    elif status == "invalid_value":
        print("KẾT LUẬN: phép lặp sinh giá trị không hợp lệ, không chứng nhận kết quả.")
    else:
        print("KẾT LUẬN: chưa đủ điều kiện chứng nhận đạt ε trong giới hạn bước.")


# -----------------------------------------------------------------------------
# 6. GIAO DIỆN CHẠY
# -----------------------------------------------------------------------------


def main() -> None:
    print("GIẢI AX=B / JACOBI TRỰC TIẾP BẰNG PHƯƠNG PHÁP LẶP JACOBI")
    print("Các phần tử trong một hàng có thể cách nhau bởi khoảng trắng, dấu phẩy hoặc dấu chấm phẩy.")
    print("Có thể nhập số dạng 0.25, 0,25, 1/4 hoặc 2e-3.\n")

    task_mode = ask_choice(
        "1. Giải hệ AX=B\n"
        "2. Tìm ma trận nghịch đảo A^(-1)\n"
        "3. Jacobi trực tiếp cho x^(k+1)=B x^(k)+d\n"
        "Chọn",
        ("1", "2", "3"),
    )
    if task_mode == 3:
        n_direct = int(ask_number("Cấp n", positive=True, integer=True))
        B_direct = ask_matrix("B", n_direct, n_direct)
        d_direct = np.asarray(ask_row(f"Nhập vector d gồm {n_direct} phần tử: ", n_direct), dtype=float)
        x0_choice = ask_choice(
            "1. Dùng x^(0)=0\n2. Tự nhập x^(0)\nChọn",
            ("1", "2"),
            default="1",
        )
        x0_direct = (
            np.zeros(n_direct, dtype=float)
            if x0_choice == 1
            else np.asarray(ask_row(f"Nhập x^(0) gồm {n_direct} phần tử: ", n_direct), dtype=float)
        )
        stop_choice = ask_choice(
            "1. Thực hiện đúng k bước\n2. Lặp đến ε\nChọn",
            ("1", "2"),
            default="1",
        )
        if stop_choice == 1:
            fixed_steps = int(ask_number("Số bước k", nonnegative=True, integer=True))
            epsilon = 1e-12
            max_iter = max(1, fixed_steps)
            stop_mode = "fixed"
        else:
            epsilon = float(ask_number("Sai số ε", positive=True, default=1e-8))
            max_iter = int(ask_number("max_iter", positive=True, integer=True, default=10000))
            fixed_steps = 0
            stop_mode = "epsilon"
        norm_choice = ask_choice(
            "1. Chuẩn vô cùng\n2. Chuẩn 1\n3. Chuẩn 2\nChọn",
            ("1", "2", "3"),
            default="1",
        )
        norm_kind = {1: "inf", 2: "one", 3: "two"}[norm_choice]
        decimals = int(
            ask_number(
                "Số chữ số sau dấu phẩy dùng để trình bày",
                nonnegative=True,
                integer=True,
                default=7,
            )
        )
        try:
            result = jacobi_fixed_point(
                B_direct,
                d_direct,
                x0_direct,
                stop_mode=stop_mode,
                epsilon=epsilon,
                fixed_steps=fixed_steps,
                max_iter=max_iter,
                norm_kind=norm_kind,
            )
        except ValueError as exc:
            print(f"Không thể thực hiện: {exc}")
            return
        print_direct_jacobi_result(B_direct, d_direct, result, decimals)
        return

    n = int(ask_number("Cấp n của ma trận A", positive=True, integer=True))
    A = ask_matrix("A", n, n)

    if task_mode == 1:
        m = int(ask_number("Số cột của B", positive=True, integer=True, default=1))
        B = ask_matrix("B", n, m)
    else:
        m = n
        B = np.eye(n)

    x0_choice = ask_choice(
        "1. Dùng X^(0)=0\n2. Tự nhập X^(0)\nChọn",
        ("1", "2"),
        default="1",
    )
    X0 = np.zeros((n, m), dtype=float) if x0_choice == 1 else ask_matrix("X^(0)", n, m)

    print("\nGiả thiết chéo trội ghi trong đề:")
    dominance_choice = ask_choice(
        "1. Chéo trội cột (như đề mẫu trong ảnh)\n"
        "2. Chéo trội hàng\n"
        "3. Không ghi rõ, để chương trình tự nhận diện\n"
        "Chọn",
        ("1", "2", "3"),
        default="3",
    )
    dominance_preference = {
        1: "column",
        2: "row",
        3: "auto",
    }[dominance_choice]

    print("\nĐiều kiện dừng của đề:")
    stop_choice = ask_choice(
        "1. Sai số tuyệt đối hậu nghiệm\n"
        "2. Sai số tương đối hậu nghiệm\n"
        "3. Sai số tiên nghiệm\n"
        "4. Thực hiện đúng k bước\n"
        "5. d chữ số thập phân tin cậy\n"
        "Chọn",
        ("1", "2", "3", "4", "5"),
        default="1",
    )

    decimal_digits_request: Optional[int] = None
    fixed_steps = 0
    if stop_choice == 1:
        stop_mode = "posteriori_absolute"
        epsilon = float(ask_number("Sai số ε", positive=True, default=1e-6))
    elif stop_choice == 2:
        stop_mode = "posteriori_relative"
        epsilon = float(ask_number("Sai số tương đối ε", positive=True, default=1e-6))
    elif stop_choice == 3:
        stop_mode = "apriori"
        epsilon = float(ask_number("Sai số ε", positive=True, default=1e-6))
    elif stop_choice == 4:
        stop_mode = "fixed"
        fixed_steps = int(ask_number("Số bước k", nonnegative=True, integer=True))
        epsilon = 0.0
    else:
        stop_mode = "posteriori_absolute"
        decimal_digits_request = int(
            ask_number("Số chữ số thập phân tin cậy d", nonnegative=True, integer=True)
        )
        epsilon = 0.5 * 10.0 ** (-decimal_digits_request)

    decimals = int(
        ask_number(
            "Số chữ số sau dấu phẩy dùng để trình bày",
            nonnegative=True,
            integer=True,
            default=7,
        )
    )
    max_iter = int(ask_number("max_iter", positive=True, integer=True, default=10000))

    try:
        result = jacobi_solve(
            A,
            B,
            X0,
            stop_mode=stop_mode,
            epsilon=epsilon,
            fixed_steps=fixed_steps,
            max_iter=max_iter,
            dominance_preference=dominance_preference,
        )
    except ValueError as exc:
        print(f"Không thể thực hiện: {exc}")
        return

    print_solution(
        result,
        task_mode=task_mode,
        decimals=decimals,
        decimal_digits_request=decimal_digits_request,
    )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã dừng chương trình; không có dữ liệu đầu vào đầy đủ.")
    except Exception as error:
        print(f"\nKhông thể thực hiện: {error}")
