"""Giải hệ tuyến tính bằng phương pháp Gauss–Seidel (Seidel).

Nguyên tắc của bản này:
- Nghiệm chỉ được tạo bởi các bước lặp Gauss–Seidel.
- Không dùng solve/inv/eig/cholesky/rank/det của NumPy để lấy nghiệm hoặc
  để che giấu một phương pháp trực tiếp khác.
- Chỉ khẳng định sai số nghiệm khi có chặn co hợp lệ.
- Với ma trận chéo trội cột, đặt y_i=a_ii*x_i và dùng một chuẩn 1 có
  trọng số để xây dựng chặn hậu nghiệm đúng.
"""

import math
from exam_format import exam_print as print, to_subscript
from input_utils import MathInputError, parse_real as parse_input_real, split_number_row
import sys
from itertools import permutations

import numpy as np

try:
    import sympy as sp
except ImportError:
    sp = None


ZERO_TOL = 1e-14
HUGE_VALUE = 1e100


# ============================================================================
# NHẬP VÀ HIỂN THỊ
# ============================================================================


def parse_number(text):
    text = text.strip()
    if not text:
        raise ValueError("Thiếu giá trị.")
    try:
        if sp is not None:
            value = sp.Float(parse_input_real(text))
            if value.is_real is False or value.free_symbols:
                raise ValueError(f"'{text}' không phải số thực.")
            result = float(value.evalf())
        elif "/" in text:
            result = parse_input_real(text)
        else:
            result = float(text)
    except (TypeError, ValueError, ZeroDivisionError, SyntaxError) as exc:
        raise ValueError(f"Không đọc được số '{text}'.") from exc
    if not math.isfinite(result):
        raise ValueError("Không chấp nhận NaN hoặc vô cùng.")
    return result


def parse_numbers(line, expected):
    try:
        parts = split_number_row(line, expected)
    except MathInputError as exc:
        raise ValueError(str(exc)) from exc
    return np.array([parse_number(item) for item in parts], dtype=float)


def read_int(prompt, minimum=1, default=None):
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"  Lỗi: cần số nguyên >= {minimum}.")


def read_float(prompt, positive=False, default=None):
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            value = parse_number(raw)
            if not positive or value > 0:
                return value
        except ValueError as exc:
            print(f"  Lỗi: {exc}")
            continue
        print("  Lỗi: giá trị phải dương.")


def read_vector(n, prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return np.full(n, default, dtype=float)
        try:
            return parse_numbers(raw, n)
        except ValueError as exc:
            print(f"  Lỗi: {exc}")


def read_matrix(rows, columns, name):
    print(f"Nhập ma trận {name} kích thước {rows}x{columns}:")
    result = []
    for i in range(rows):
        while True:
            try:
                result.append(parse_numbers(input(f"  Hàng {i + 1}: "), columns))
                break
            except ValueError as exc:
                print(f"  Lỗi: {exc}")
    return np.vstack(result)


def _display_number(value, precision):
    value = float(value)
    threshold = 0.5 * 10.0 ** (-precision) if precision > 0 else 0.5
    if abs(value) < threshold:
        value = 0.0
    if value != 0 and (
        abs(value) >= 10 ** (precision + 2) or abs(value) < 10 ** (-max(precision, 1))
    ):
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def print_matrix(name, matrix, precision=6):
    array = np.asarray(matrix, dtype=float)
    rows = array.reshape(-1, 1) if array.ndim == 1 else array
    texts = [[_display_number(value, precision) for value in row] for row in rows]
    width = max(10, max(len(item) for row in texts for item in row) + 1)
    print(f"{name} =")
    for row in texts:
        print("  [" + " ".join(item.rjust(width) for item in row) + "]")


def format_table_number(value, precision):
    return _display_number(value, precision)


# ============================================================================
# PHÉP TOÁN PHỤ TRỢ – KHÔNG DÙNG BỘ GIẢI HỆ CÓ SẴN
# ============================================================================


def validate_system(a, b=None):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]:
        raise ValueError("A phải là ma trận vuông cấp n>0.")
    if not np.all(np.isfinite(a)):
        raise ValueError("A chứa NaN hoặc vô cùng.")
    if b is not None:
        b = np.asarray(b, dtype=float)
        if b.shape != (len(a),) or not np.all(np.isfinite(b)):
            raise ValueError("b sai kích thước hoặc chứa NaN/vô cùng.")


def forward_substitution(lower, rhs):
    """Giải hệ tam giác dưới L X=R bằng thế tiến."""
    lower = np.asarray(lower, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    n = lower.shape[0]
    if lower.shape != (n, n):
        raise ValueError("Ma trận thế tiến phải vuông.")
    vector_rhs = rhs.ndim == 1
    work = rhs.reshape(n, 1).copy() if vector_rhs else rhs.copy()
    if work.ndim != 2 or work.shape[0] != n:
        raise ValueError("Vế phải của phép thế tiến sai kích thước.")
    result = np.zeros_like(work, dtype=float)
    scale = max(1.0, float(np.max(np.abs(lower))))
    for i in range(n):
        pivot = lower[i, i]
        if abs(pivot) <= ZERO_TOL * scale:
            raise ArithmeticError("Phần tử chéo của hệ tam giác bằng hoặc quá gần 0.")
        result[i] = (work[i] - lower[i, :i] @ result[:i]) / pivot
    return result[:, 0] if vector_rhs else result


def iteration_matrix(a):
    """T=-(D+L_A)^(-1)U_A, được tính bằng thế tiến."""
    a = np.asarray(a, dtype=float)
    return forward_substitution(np.tril(a), -np.triu(a, 1))


def iteration_vector(a, b):
    """g=(D+L_A)^(-1)b, được tính bằng thế tiến."""
    return forward_substitution(np.tril(a), np.asarray(b, dtype=float))


def matrix_norm_inf(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return float(np.max(np.sum(np.abs(matrix), axis=1)))


def vector_norm_inf(vector):
    return float(np.max(np.abs(np.asarray(vector, dtype=float))))


def vector_norm_1(vector):
    return float(np.sum(np.abs(np.asarray(vector, dtype=float))))


def relative_residual(a, x, b):
    residual = np.asarray(b) - np.asarray(a) @ np.asarray(x)
    numerator = vector_norm_inf(residual)
    denominator = matrix_norm_inf(a) * vector_norm_inf(x) + vector_norm_inf(b)
    if denominator == 0:
        return 0.0 if numerator == 0 else math.inf
    return numerator / denominator


def row_dominance_margins(a):
    diagonal = np.abs(np.diag(a))
    return diagonal - (np.sum(np.abs(a), axis=1) - diagonal)


def column_dominance_margins(a):
    diagonal = np.abs(np.diag(a))
    return diagonal - (np.sum(np.abs(a), axis=0) - diagonal)


def dominance_kind(margins):
    margins = np.asarray(margins, dtype=float)
    if np.all(margins > ZERO_TOL):
        return "chéo trội nghiêm ngặt"
    if np.all(margins >= -ZERO_TOL):
        return "chéo trội không nghiêm ngặt"
    return "không chéo trội"


def ldlt_spd_certificate(a):
    """Kiểm tra SPD bằng phân tích LDL^T tự cài đặt, không dùng Cholesky có sẵn."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    scale = max(1.0, float(np.max(np.abs(a))))
    symmetric = np.allclose(a, a.T, rtol=1e-11, atol=1e-13 * scale)
    result = {
        "symmetric": symmetric,
        "spd": False,
        "L": None,
        "D": None,
        "leading_determinants": [],
    }
    if not symmetric:
        return result

    l_matrix = np.eye(n)
    d_vector = np.zeros(n)
    determinants = []
    product = 1.0
    for j in range(n):
        pivot = a[j, j] - math.fsum(
            l_matrix[j, k] * l_matrix[j, k] * d_vector[k] for k in range(j)
        )
        if not math.isfinite(pivot) or pivot <= ZERO_TOL * scale:
            return result
        d_vector[j] = pivot
        product *= pivot
        determinants.append(product)
        for i in range(j + 1, n):
            numerator = a[i, j] - math.fsum(
                l_matrix[i, k] * l_matrix[j, k] * d_vector[k] for k in range(j)
            )
            l_matrix[i, j] = numerator / pivot
    result.update(spd=True, L=l_matrix, D=d_vector, leading_determinants=determinants)
    return result


def sassenfeld_coefficients(a):
    """Hệ số Sassenfeld theo thứ tự cập nhật từ 1 đến n."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    beta = np.zeros(n)
    for i in range(n):
        denominator = abs(a[i, i])
        if denominator <= ZERO_TOL:
            return beta, math.inf
        lower = math.fsum(abs(a[i, j]) * beta[j] for j in range(i))
        upper = math.fsum(abs(a[i, j]) for j in range(i + 1, n))
        beta[i] = (lower + upper) / denominator
    return beta, float(np.max(beta))


def column_weighted_certificate(a):
    """Chứng nhận co cho nhánh y_i=a_ii*x_i.

    C=I-A D^(-1), C=L+U. Đặt
        ell_j = sum_{i>j}|c_ij|,
        u_j   = sum_{i<j}|c_ij|,
        w_j   = 1-ell_j,
        q     = max_j u_j/w_j.
    Nếu q<1 thì Gauss–Seidel là ánh xạ co theo chuẩn
        ||z||_w = sum_j w_j|z_j|.
    """
    a = np.asarray(a, dtype=float)
    diagonal = np.diag(a)
    if np.any(np.abs(diagonal) <= ZERO_TOL):
        return None
    c_matrix = np.eye(len(a)) - a @ np.diag(1.0 / diagonal)
    lower = np.tril(c_matrix, -1)
    upper = np.triu(c_matrix, 1)
    ell = np.sum(np.abs(lower), axis=0)
    upper_column = np.sum(np.abs(upper), axis=0)
    weights = 1.0 - ell
    if np.any(weights <= ZERO_TOL):
        return {
            "valid": False,
            "C": c_matrix,
            "L": lower,
            "U": upper,
            "weights": weights,
            "ell": ell,
            "u": upper_column,
            "q": math.inf,
            "scale_to_x_inf": math.inf,
        }
    ratios = upper_column / weights
    q_value = float(np.max(ratios))
    scale_to_x_inf = 1.0 / float(np.min(weights * np.abs(diagonal)))
    return {
        "valid": math.isfinite(q_value) and q_value < 1.0,
        "C": c_matrix,
        "L": lower,
        "U": upper,
        "weights": weights,
        "ell": ell,
        "u": upper_column,
        "q": q_value,
        "scale_to_x_inf": scale_to_x_inf,
    }


def weighted_one_norm(vector, weights):
    return float(np.sum(np.asarray(weights) * np.abs(np.asarray(vector))))


def pdf_seidel_parameters(a, b):
    """Các đại lượng Gauss-Seidel đúng ký hiệu PDF cho Ax=b.

    Dùng cách 2 trong tài liệu: T=diag(1/a_ii), C=I-T.A, d=T.b,
    rồi tách C=L+U. Với chéo trội hàng dùng chuẩn vô cùng; với
    chéo trội cột dùng chuẩn 1 và hệ số s. Hàm này chỉ lập chứng nhận
    sai số PDF, không dùng để giải hệ bằng phương pháp trực tiếp.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    diagonal = np.diag(a)
    if np.any(np.abs(diagonal) <= ZERO_TOL):
        return {"valid": False, "reason": "Có phần tử đường chéo bằng 0."}

    t_diag = np.diag(1.0 / diagonal)
    c_matrix = np.eye(n) - t_diag @ a
    d_vector = t_diag @ b
    lower = np.tril(c_matrix, -1)
    upper = np.triu(c_matrix, 1)

    row_lower = np.sum(np.abs(lower), axis=1)
    row_upper = np.sum(np.abs(upper), axis=1)
    col_lower = np.sum(np.abs(lower), axis=0)
    col_upper = np.sum(np.abs(upper), axis=0)

    row_valid = bool(np.all(1.0 - row_lower > ZERO_TOL))
    col_valid = bool(np.all(1.0 - col_lower > ZERO_TOL))
    row_q = math.inf
    col_q = math.inf
    col_s = math.inf
    if row_valid:
        row_q = float(np.max(row_upper / (1.0 - row_lower)))
    if col_valid:
        col_s = float(np.max(col_lower))
        col_q = float(np.max(col_upper / (1.0 - col_lower)))

    row_dom = bool(np.all(row_dominance_margins(a) > ZERO_TOL))
    col_dom = bool(np.all(column_dominance_margins(a) > ZERO_TOL))
    mode = None
    q_value = math.inf
    s_value = math.inf
    norm_name = None
    coefficient = math.inf
    if row_dom and row_valid and math.isfinite(row_q) and row_q < 1.0:
        mode = "row"
        q_value = row_q
        s_value = 0.0
        norm_name = "∞"
        coefficient = q_value / (1.0 - q_value)
    elif col_dom and col_valid and math.isfinite(col_q) and col_q < 1.0 and col_s < 1.0:
        mode = "column"
        q_value = col_q
        s_value = col_s
        norm_name = "1"
        coefficient = q_value / ((1.0 - q_value) * (1.0 - s_value))

    return {
        "valid": mode is not None,
        "mode": mode,
        "C_pdf": c_matrix,
        "d_pdf": d_vector,
        "L_pdf": lower,
        "U_pdf": upper,
        "row_q": row_q,
        "column_q": col_q,
        "s": s_value,
        "q": q_value,
        "coefficient": coefficient,
        "norm": norm_name,
        "row_lower": row_lower,
        "row_upper": row_upper,
        "col_lower": col_lower,
        "col_upper": col_upper,
        "row_dominant": row_dom,
        "column_dominant": col_dom,
        "reason": (
            "Dùng chéo trội hàng theo PDF." if mode == "row"
            else "Dùng chéo trội cột theo PDF." if mode == "column"
            else "Chưa lập được q,s theo công thức PDF."
        ),
    }


# ============================================================================
# CHỌN THỨ TỰ PHƯƠNG TRÌNH
# ============================================================================


def _candidate_analysis(a):
    diagonal = np.diag(a)
    if np.any(np.abs(diagonal) <= ZERO_TOL):
        return None
    try:
        t_matrix = iteration_matrix(a)
    except ArithmeticError:
        return None
    q_x = matrix_norm_inf(t_matrix)
    row_margins = row_dominance_margins(a)
    col_margins = column_dominance_margins(a)
    row_dom = bool(np.all(row_margins > ZERO_TOL))
    col_dom = bool(np.all(col_margins > ZERO_TOL))
    col_data = column_weighted_certificate(a)
    q_y = col_data["q"] if col_data is not None else math.inf
    certified_values = [q for q in (q_x, q_y) if math.isfinite(q) and q < 1]
    best_q = min(certified_values) if certified_values else min(q_x, q_y)
    return {
        "q_x": q_x,
        "q_y": q_y,
        "best_q": best_q,
        "certified": bool(certified_values),
        "row_dom": row_dom,
        "col_dom": col_dom,
        "dominance_sum": float(np.sum(row_margins)),
    }


def _candidate_rank(item):
    return (
        0 if item["certified"] else 1,
        item["best_q"],
        0 if item["row_dom"] else (1 if item["col_dom"] else 2),
        -item["dominance_sum"],
        item["changes"],
    )


def _greedy_nonzero_diagonal(a):
    """Ghép mỗi cột với một hàng có hệ số lớn, chỉ dùng khi n lớn."""
    n = len(a)
    choices = [list(np.argsort(-np.abs(a[:, column]))) for column in range(n)]
    column_to_row = np.full(n, -1, dtype=int)

    def augment(column, seen):
        for row in choices[column]:
            row = int(row)
            if abs(a[row, column]) <= ZERO_TOL or row in seen:
                continue
            seen.add(row)
            old = np.where(column_to_row == row)[0]
            if old.size == 0 or augment(int(old[0]), seen):
                column_to_row[column] = row
                return True
        return False

    for column in range(n):
        if not augment(column, set()):
            return None
    return column_to_row


def find_best_row_permutation(a, max_local_rounds=6):
    n = len(a)
    evaluated = []

    def add(permutation, source):
        permutation = np.asarray(permutation, dtype=int)
        data = _candidate_analysis(a[permutation])
        if data is None:
            return
        data.update(
            permutation=permutation.copy(),
            source=source,
            changes=int(np.count_nonzero(permutation != np.arange(n))),
        )
        evaluated.append(data)

    if n <= 8:
        for permutation in permutations(range(n)):
            add(permutation, "vét cạn")
        method = f"vét cạn {math.factorial(n)} hoán vị hàng"
    else:
        identity = np.arange(n)
        add(identity, "thứ tự ban đầu")
        seed = _greedy_nonzero_diagonal(a)
        if seed is not None:
            add(seed, "ghép đường chéo")
        active = [identity] + ([seed] if seed is not None else [])
        for round_index in range(max_local_rounds):
            next_active = []
            for base in active[:4]:
                local = []
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        candidate = np.array(base, dtype=int)
                        candidate[i], candidate[j] = candidate[j], candidate[i]
                        data = _candidate_analysis(a[candidate])
                        if data is not None:
                            data.update(
                                permutation=candidate.copy(),
                                source=f"đổi cặp vòng {round_index + 1}",
                                changes=int(
                                    np.count_nonzero(candidate != np.arange(n))
                                ),
                            )
                            evaluated.append(data)
                            local.append(data)
                if local:
                    next_active.append(min(local, key=_candidate_rank)["permutation"])
            if not next_active:
                break
            active = next_active
        method = "ghép đường chéo và tìm cục bộ bằng đổi cặp hàng"

    if not evaluated:
        return None, method, []
    evaluated.sort(key=_candidate_rank)
    return evaluated[0], method, evaluated[:5]


def prepare_system(a, b, auto_reorder=True):
    validate_system(a, b)
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    n = len(a)
    identity = np.arange(n)
    original_spd = ldlt_spd_certificate(a)["spd"]
    original = _candidate_analysis(a)

    info = {
        "permutation": identity,
        "reordered": False,
        "reason": "Giữ nguyên thứ tự phương trình.",
        "method": "không tìm",
        "top_candidates": [],
        "spd_original": original_spd,
    }

    if original_spd:
        info["reason"] = "A là SPD; không đổi riêng hàng để tránh phá tính đối xứng."
    elif auto_reorder:
        best, method, top = find_best_row_permutation(a)
        info["method"] = method
        info["top_candidates"] = top
        if best is None:
            raise ValueError("Không tìm được thứ tự hàng có đường chéo khác 0.")
        changed = not np.array_equal(best["permutation"], identity)
        original_certified = original is not None and original["certified"]
        original_q = original["best_q"] if original is not None else math.inf
        accept = False
        if original is None:
            accept = changed
            reason = "Đổi hàng để tạo đường chéo khác 0."
        elif best["certified"] and not original_certified:
            accept = changed
            reason = "Đổi hàng vì tìm được một chứng nhận co q<1."
        elif (
            best["certified"]
            and original_certified
            and best["best_q"] < original_q - 1e-12
        ):
            accept = changed
            reason = "Đổi hàng vì hệ số co được cải thiện."
        else:
            reason = (
                "Không đổi hàng vì không có phương án tốt hơn về chứng nhận hội tụ."
            )
        if accept:
            permutation = best["permutation"]
            a = a[permutation]
            b = b[permutation]
            info.update(permutation=permutation, reordered=True, reason=reason)
        else:
            info["reason"] = reason
    else:
        info["reason"] = "Người dùng tắt tự động đổi hàng."

    if np.any(np.abs(np.diag(a)) <= ZERO_TOL):
        raise ValueError("Đường chéo có phần tử bằng hoặc quá gần 0.")
    return a, b, info


# ============================================================================
# PHÂN TÍCH VÀ LẶP SEIDEL
# ============================================================================


def build_iteration_analysis(a, b):
    n = len(a)
    diagonal = np.diag(a)
    row_margins = row_dominance_margins(a)
    col_margins = column_dominance_margins(a)
    row_dom = bool(np.all(row_margins > ZERO_TOL))
    col_dom = bool(np.all(col_margins > ZERO_TOL))
    spd_data = ldlt_spd_certificate(a)

    d_matrix = np.diag(diagonal)
    lower_a = np.tril(a, -1)
    upper_a = np.triu(a, 1)
    t_direct = iteration_matrix(a)
    g_direct = iteration_vector(a, b)
    q_direct = matrix_norm_inf(t_direct)
    beta, q_sassenfeld = sassenfeld_coefficients(a)
    column_data = column_weighted_certificate(a)
    pdf_data = pdf_seidel_parameters(a, b)

    if pdf_data["valid"] and pdf_data["mode"] == "row":
        variable = "x"
        method = "Seidel theo PDF (chéo trội hàng)"
    elif pdf_data["valid"] and pdf_data["mode"] == "column":
        variable = "x"
        method = "Seidel theo PDF (chéo trội cột)"
    elif q_direct < 1:
        variable = "x"
        method = "Seidel trực tiếp với ||T||∞<1"
    elif column_data is not None and column_data["valid"]:
        variable = "y"
        method = "Seidel đổi biến với chuẩn 1 có trọng số"
    else:
        variable = "x"
        method = "Seidel tổng quát (chưa có chặn co)"

    if variable == "x":
        c_matrix = np.eye(n) - np.diag(1.0 / diagonal) @ a
        lower = np.tril(c_matrix, -1)
        upper = np.triu(c_matrix, 1)
        d_vector = b / diagonal
        theory = None
        if pdf_data["valid"]:
            theory = {
                "kind": "pdf_inf" if pdf_data["mode"] == "row" else "pdf_one",
                "q": pdf_data["q"],
                "s": pdf_data["s"],
                "coefficient": pdf_data["coefficient"],
                "description": pdf_data["reason"],
                "norm": pdf_data["norm"],
            }
        elif q_direct < 1:
            theory = {
                "kind": "x_inf",
                "q": q_direct,
                "s": 0.0,
                "coefficient": q_direct / (1 - q_direct),
                "description": "chuẩn vô cùng của ma trận lặp T",
            }
    else:
        c_matrix = column_data["C"]
        lower = column_data["L"]
        upper = column_data["U"]
        d_vector = b.copy()
        q_value = column_data["q"]
        theory = {
            "kind": "y_weighted",
            "q": q_value,
            "coefficient": q_value / (1 - q_value),
            "weights": column_data["weights"],
            "scale_to_x_inf": column_data["scale_to_x_inf"],
            "description": "chuẩn 1 có trọng số của biến y",
        }

    return {
        "method": method,
        "variable": variable,
        "D": d_matrix,
        "L_A": lower_a,
        "U_A": upper_a,
        "T": t_direct,
        "g": g_direct,
        "q_direct": q_direct,
        "beta": beta,
        "q_sassenfeld": q_sassenfeld,
        "C": c_matrix,
        "L": lower,
        "U": upper,
        "d": d_vector,
        "row_margins": row_margins,
        "col_margins": col_margins,
        "row_dom": row_dom,
        "col_dom": col_dom,
        "spd": spd_data["spd"],
        "spd_data": spd_data,
        "column_data": column_data,
        "pdf_data": pdf_data,
        "theory": theory,
    }


def estimate_apriori_iterations(q, first_difference, scale, epsilon):
    """Tìm k sao cho scale*q^k/(1-q)*first_difference <= epsilon."""
    if not (0 <= q < 1) or first_difference < 0 or scale <= 0:
        return None
    if first_difference == 0:
        return 0
    if q == 0:
        return 1
    target = epsilon * (1 - q) / (scale * first_difference)
    if target >= 1:
        return 0
    if target <= 0 or not math.isfinite(target):
        return None
    try:
        result = max(0, math.ceil(math.log(target) / math.log(q)))
        while scale * q**result / (1 - q) * first_difference > epsilon:
            result += 1
        return result
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


def _diagnose(xs, residuals, epsilon):
    if len(xs) >= 6:
        recent = [
            vector_norm_inf(xs[i] - xs[i - 1]) for i in range(len(xs) - 4, len(xs))
        ]
        machine = 100 * np.finfo(float).eps * max(1.0, vector_norm_inf(xs[-1]))
        if max(recent) <= machine and residuals[-1] > epsilon:
            return "Đình trệ do giới hạn số học máy."
    if len(residuals) >= 7 and all(
        residuals[i] > 1.2 * residuals[i - 1]
        for i in range(len(residuals) - 5, len(residuals))
    ):
        return "Phần dư tăng liên tiếp; phép lặp có dấu hiệu phân kỳ."
    if len(xs) >= 7:
        norms = [vector_norm_inf(value) for value in xs[-6:]]
        if norms[0] > 0 and all(
            norms[i] > 2 * norms[i - 1] for i in range(1, len(norms))
        ):
            return "Chuẩn vector lặp tăng rất nhanh; phép lặp có dấu hiệu phân kỳ."
    return None


def seidel_fixed_point(
    b_matrix,
    d,
    x0=None,
    epsilon=1e-6,
    max_iter=500,
    precision=6,
    show=True,
    fixed_iterations=None,
    stop_mode="both",
    residual_tolerance=None,
):
    """Lặp Seidel *trực tiếp* cho ``x=B x+d``.

    Thành phần đường chéo của ``B`` thuộc tổng dùng vòng cũ::

        x_i^(k+1) = d_i + sum_(j<i) b_ij x_j^(k+1)
                        + sum_(j>=i) b_ij x_j^(k).

    Đây không phải là việc đổi bài toán sang ``(I-B)x=d`` rồi áp dụng công
    thức Seidel của hệ ``Ax=b``; hai cách có các bước trung gian khác nhau.
    """
    valid_stop_modes = {"absolute", "relative", "residual", "both"}
    if stop_mode not in valid_stop_modes:
        raise ValueError(
            "stop_mode phải là 'absolute', 'relative', 'residual' hoặc 'both'."
        )
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon phải là số hữu hạn dương.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter phải là số nguyên dương.")
    if fixed_iterations is not None and (
        not isinstance(fixed_iterations, (int, np.integer))
        or fixed_iterations < 0
    ):
        raise ValueError("fixed_iterations phải là số nguyên không âm.")
    if residual_tolerance is None:
        residual_tolerance = epsilon
    if not math.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("residual_tolerance phải là số hữu hạn dương.")

    B = np.asarray(b_matrix, dtype=float)
    d = np.asarray(d, dtype=float)
    if B.ndim != 2 or B.shape[0] == 0 or B.shape[0] != B.shape[1]:
        raise ValueError("B phải là ma trận vuông cấp n>0.")
    n = B.shape[0]
    if d.shape != (n,):
        raise ValueError("d phải là vector có đúng n phần tử.")
    if not np.all(np.isfinite(B)) or not np.all(np.isfinite(d)):
        raise ValueError("B và d không được chứa NaN hoặc vô cùng.")
    x = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float).copy()
    if x.shape != (n,) or not np.all(np.isfinite(x)):
        raise ValueError("x0 sai kích thước hoặc chứa NaN/vô cùng.")

    # (I-L_B)x^(k+1)=(D_B+U_B)x^(k)+d.  Chỉ dùng thế tiến,
    # không gọi bộ giải hệ thư viện.
    lower = np.tril(B, -1)
    old_part = np.triu(B, 0)
    identity_minus_lower = np.eye(n) - lower
    iteration = forward_substitution(identity_minus_lower, old_part)
    constant = forward_substitution(identity_minus_lower, d)
    q_iteration = matrix_norm_inf(iteration)
    q_mapping = matrix_norm_inf(B)
    beta = np.zeros(n, dtype=float)
    for i in range(n):
        beta[i] = (
            float(np.abs(B[i, :i]) @ beta[:i])
            + float(np.sum(np.abs(B[i, i:])))
        )
    q_sassenfeld = float(np.max(beta))
    q = q_iteration
    theory = q < 1.0

    def fixed_residual(value):
        return vector_norm_inf(value - B @ value - d)

    def relative_fixed_residual(value, residual):
        scale = vector_norm_inf(value) + matrix_norm_inf(B) * vector_norm_inf(value) + vector_norm_inf(d)
        return residual / scale if scale else residual

    residual0 = fixed_residual(x)
    history = [{
        "k": 0,
        "x": x.copy(),
        "step": 0.0,
        "relative_step": 0.0,
        "residual": residual0,
        "relative_residual": relative_fixed_residual(x, residual0),
        "error_bound": math.nan,
    }]
    converged = residual0 == 0.0
    certified = converged
    fixed_completed = fixed_iterations == 0
    reason = (
        "x^(0) thỏa x=B x+d trong số học máy."
        if converged
        else ""
    )

    if show:
        print("\n" + "=" * 92)
        print("SEIDEL TRỰC TIẾP CHO x=B x+d")
        print("=" * 92)
        print_matrix("B", B, precision)
        print_matrix("d", d, precision)
        print_matrix("x^(0)", x, precision)
        print("\nCông thức tổng quát:")
        print("  x_i^(k+1)=d_i+Σ_(j=1..i-1)b_ij·x_j^(k+1)")
        print("                    +Σ_(j=i..n)b_ij·x_j^(k).")
        print("  Tổng thứ hai chứa b_ii·x_i^(k); không đổi sang (I-B)x=d.")
        print("\nCông thức cụ thể:")
        for i in range(n):
            new_terms = " ".join(
                f"+({B[i, j]:.{precision}g})x_{j + 1}^(k+1)" for j in range(i)
            )
            old_terms = " ".join(
                f"+({B[i, j]:.{precision}g})x_{j + 1}^(k)" for j in range(i, n)
            )
            print(f"  x_{i + 1}^(k+1)={d[i]:.{precision}g} {new_terms} {old_terms}")
        print("\nPhân tích hội tụ của đúng phép lặp tuần tự này:")
        print("  (I-L_B)x^(k+1)=(D_B+U_B)x^(k)+d")
        print_matrix("B_1=L_B", lower, precision)
        print_matrix("B_2=D_B+U_B", old_part, precision)
        print_matrix("Tₛ=(I-L_B)^(-1)(D_B+U_B)", iteration, precision)
        print(f"  ||B||∞={q_mapping:.{precision}g}")
        print(
            "  β_i=Σ_(j<i)|b_ij|β_j+Σ_(j≥i)|b_ij|: "
            + ", ".join(
                f"β{to_subscript(i + 1)}={value:.{precision}g}"
                for i, value in enumerate(beta)
            )
        )
        print(f"  β=max β_i={q_sassenfeld:.{precision}g}; ||Tₛ||∞={q_iteration:.{precision}g}.")
        if theory:
            print("  Vì ||Tₛ||∞<1, phép lặp là ánh xạ co và hội tụ với mọi x^(0).")
        else:
            print("  Chưa chứng minh được hội tụ bằng chuẩn ∞; điều này không có nghĩa là phân kỳ.")
        if fixed_iterations is not None:
            print(f"\nYêu cầu: thực hiện đúng k={fixed_iterations} bước, không dừng sớm.")
        else:
            print(f"\nYêu cầu: lặp đến epsilon={epsilon:.{precision}e}, chế độ '{stop_mode}'.")
        print("\nBẢNG LẶP")
        print("  Mỗi x_i mới được dùng ngay cho các thành phần đứng sau trong cùng vòng.")

    loop_limit = (
        fixed_iterations
        if fixed_iterations is not None
        else (0 if converged else max_iter)
    )
    k = 0
    for k in range(1, loop_limit + 1):
        old = x.copy()
        component_details = []
        for i in range(n):
            new_sum = float(B[i, :i] @ x[:i])
            old_sum = float(B[i, i:] @ old[i:])
            x[i] = d[i] + new_sum + old_sum
            component_details.append((i, new_sum, old_sum, float(x[i])))

        if not np.all(np.isfinite(x)) or vector_norm_inf(x) > HUGE_VALUE:
            reason = "Giá trị lặp không hữu hạn hoặc tăng quá lớn; phát hiện phân kỳ số."
            break
        step = vector_norm_inf(x - old)
        relative_step = step / max(1.0, vector_norm_inf(x))
        residual = fixed_residual(x)
        relative_residual = relative_fixed_residual(x, residual)
        error_bound = q / (1 - q) * step if theory else math.nan
        history.append({
            "k": k,
            "x": x.copy(),
            "step": step,
            "relative_step": relative_step,
            "residual": residual,
            "relative_residual": relative_residual,
            "error_bound": error_bound,
            "components": component_details,
        })

        absolute_ok = step <= epsilon
        relative_ok = relative_step <= epsilon
        residual_ok = residual <= residual_tolerance
        stop_ok = {
            "absolute": absolute_ok,
            "relative": relative_ok,
            "residual": residual_ok,
            "both": absolute_ok and residual_ok,
        }[stop_mode]

        if show:
            print_matrix(f"x^({k})", x, precision)
            for i, new_sum, old_sum, value in component_details:
                print(
                    f"  x_{i + 1}^({k})=d_{i + 1}"
                    f" + ({new_sum:.{precision}g})[mới]"
                    f" + ({old_sum:.{precision}g})[cũ]"
                    f" = {value:.{precision}g}"
                )
            print(
                f"  ||x^({k})-x^({k - 1})||∞={step:.{precision}e}; "
                f"||x^({k})-Bx^({k})-d||∞={residual:.{precision}e}"
            )

        if fixed_iterations is None and stop_ok:
            converged = True
            certified = bool(theory and error_bound <= epsilon and residual_ok)
            reason = (
                "Đạt điều kiện dừng đã chọn và residual điểm bất động."
                if residual_ok
                else "Đạt điều kiện dừng đã chọn."
            )
            break
        if fixed_iterations is None:
            diagnosis = _diagnose([row["x"] for row in history], [row["residual"] for row in history], epsilon)
            if diagnosis:
                reason = diagnosis
                break
    else:
        if fixed_iterations is not None:
            fixed_completed = True
            reason = f"Đã thực hiện đúng {fixed_iterations} bước."
        elif not converged:
            reason = "Hết max_iter nhưng chưa đạt điều kiện dừng."

    final_residual_vector = x - B @ x - d
    final_residual = vector_norm_inf(final_residual_vector)
    info = {
        "converged": converged,
        "certified": certified,
        "fixed_iterations_completed": fixed_completed,
        "iterations": k,
        "history": history,
        "iteration_matrix": iteration,
        "constant_vector": constant,
        "q": q,
        "q_mapping": q_mapping,
        "sassenfeld_beta": beta,
        "q_sassenfeld": q_sassenfeld,
        "residual": final_residual,
        "residual_vector": final_residual_vector,
        "reason": reason,
    }
    if show:
        print("\nKẾT LUẬN")
        print(reason)
        if fixed_completed:
            print("Kết quả sau đúng k bước không tự động đồng nghĩa dãy đã hội tụ.")
            if math.isfinite(history[-1]["error_bound"]):
                print(
                    f"Chặn sai số sau k bước: ||x^({k})-x*||∞ <= "
                    f"{history[-1]['error_bound']:.{precision}e}."
                )
            else:
                print("Chưa có chặn sai số lý thuyết cho kết quả sau k bước trong nhánh hiện tại.")
        print_matrix(f"x^({k})", x, precision)
        print_matrix(f"x^({k})-B x^({k})-d", final_residual_vector, precision)
        print(f"||x^({k})-B x^({k})-d||∞={final_residual:.{precision}e}")
    return x, info


# Tên gọi rõ nghĩa khác để dùng thuận tiện trong bài kiểm thử/bài thi.
gauss_seidel_fixed_point = seidel_fixed_point


def gauss_seidel(
    a,
    b,
    x0=None,
    epsilon=1e-6,
    max_iter=500,
    precision=6,
    auto_reorder=True,
    show=True,
    fixed_iterations=None,
    use_inverse_bound=False,
    residual_tolerance=None,
    show_y=False,
    numpy_check=False,
    stop_mode="posteriori",
):
    """Giải Ax=b bằng đúng phép lặp Gauss–Seidel."""
    del use_inverse_bound, numpy_check  # giữ tham số để tương thích bản cũ
    if stop_mode not in {"posteriori", "apriori"}:
        raise ValueError("stop_mode phải là 'posteriori' hoặc 'apriori'.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon phải là số hữu hạn dương.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter phải là số nguyên dương.")
    if fixed_iterations is not None and (
        not isinstance(fixed_iterations, (int, np.integer)) or fixed_iterations < 0
    ):
        raise ValueError("fixed_iterations phải là số nguyên không âm.")
    if residual_tolerance is None:
        residual_tolerance = epsilon
    if not math.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("residual_tolerance phải dương.")

    a_original = np.asarray(a, dtype=float)
    b_original = np.asarray(b, dtype=float)
    a, b, prep = prepare_system(a_original, b_original, auto_reorder)
    n = len(a)
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    if x.shape != (n,) or not np.all(np.isfinite(x)):
        raise ValueError("x0 sai kích thước hoặc chứa NaN/vô cùng.")

    analysis = build_iteration_analysis(a, b)
    diagonal = np.diag(a)
    y = diagonal * x if analysis["variable"] == "y" else x.copy()
    theory = analysis["theory"]
    if fixed_iterations is None and stop_mode == "apriori" and theory is None:
        raise ValueError(
            "Không lập được chặn tiên nghiệm vì chưa có hệ số co q<1 trong chuẩn đang dùng. "
            "Ma trận vẫn có thể hội tụ theo điều kiện khác (ví dụ SPD), nhưng công thức tiên nghiệm này không áp dụng."
        )

    if show:
        print("\n" + "=" * 92)
        print("GIẢI HỆ BẰNG PHƯƠNG PHÁP GAUSS–SEIDEL")
        print("=" * 92)
        print("\nPHẦN 1. DỮ LIỆU")
        print_matrix("A", a_original, precision)
        print_matrix("b", b_original, precision)
        print_matrix("x^(0)", x, precision)
        if fixed_iterations is None:
            print(f"Yêu cầu: sai số epsilon = {epsilon:.{precision}e}.")
        else:
            print(f"Yêu cầu: thực hiện đúng k={fixed_iterations} bước.")

        print("\nPHẦN 2. KIỂM TRA ĐIỀU KIỆN")
        if prep["reordered"]:
            order = ", ".join(str(int(index) + 1) for index in prep["permutation"])
            print(f"Đổi thứ tự phương trình thành ({order}).")
            print_matrix("A'", a, precision)
            print_matrix("b'", b, precision)
        else:
            print("Không đổi thứ tự phương trình.")
        print("Lý do:", prep["reason"])

        print("\nChéo trội hàng:")
        for i in range(n):
            diagonal_abs = abs(a[i, i])
            off_sum = float(np.sum(np.abs(a[i]))) - diagonal_abs
            relation = (
                ">"
                if diagonal_abs > off_sum
                else ("=" if abs(diagonal_abs - off_sum) <= ZERO_TOL else "<")
            )
            print(
                f"  i={i + 1}: |a_ii|={diagonal_abs:.{precision}g} {relation} "
                f"Σ_(j≠i)|a_ij|={off_sum:.{precision}g}"
            )
        print("Kết luận:", dominance_kind(analysis["row_margins"]))

        print("\nChéo trội cột:")
        for j in range(n):
            diagonal_abs = abs(a[j, j])
            off_sum = float(np.sum(np.abs(a[:, j]))) - diagonal_abs
            relation = (
                ">"
                if diagonal_abs > off_sum
                else ("=" if abs(diagonal_abs - off_sum) <= ZERO_TOL else "<")
            )
            print(
                f"  j={j + 1}: |a_jj|={diagonal_abs:.{precision}g} {relation} "
                f"Σ_(i≠j)|a_ij|={off_sum:.{precision}g}"
            )
        print("Kết luận:", dominance_kind(analysis["col_margins"]))

        print("\nKẾT LUẬN ĐIỀU KIỆN HỘI TỤ:")
        if analysis["row_dom"]:
            print("  A chéo trội hàng nghiêm ngặt ⇒ Gauss–Seidel hội tụ với mọi x^(0).")
            print("  Đã đủ điều kiện nên không cần xét SPD, Sassenfeld hay ||T||∞ để kết luận hội tụ.")
        elif analysis["col_dom"]:
            print("  A chéo trội cột nghiêm ngặt ⇒ Gauss–Seidel hội tụ với mọi x^(0).")
            print("  Đã đủ điều kiện nên không cần xét các tiêu chuẩn hội tụ khác.")
        else:
            print("  A không chéo trội hàng và cũng không chéo trội cột.")
            print("  Chéo trội chỉ là điều kiện đủ, không phải điều kiện cần; vì vậy xét lần lượt:")
            if analysis["spd"]:
                print("\n  1) Tiêu chuẩn SPD:")
                print("     A=A^T và phân tích A=L·D·L^T có mọi d_i>0.")
                print_matrix("L", analysis["spd_data"]["L"], precision)
                print_matrix("diag(D)", analysis["spd_data"]["D"], precision)
                print("     ⇒ A đối xứng xác định dương, nên định lý Gauss–Seidel bảo đảm hội tụ.")
                print("     Đã có chứng nhận SPD nên không cần dùng các hệ số sau để kết luận.")
            else:
                print("\n  1) A không có chứng nhận SPD bằng phân tích LDL^T.")
                print("\n  2) Chuẩn ma trận lặp:")
                print("     Tₛ=-(D+L_A)^(-1)U_A và q_T=||Tₛ||∞=max_i Σ_j|t_ij|.")
                print_matrix("Tₛ=-(D+L_A)^(-1)U_A", analysis["T"], precision)
                print(f"     q_T={analysis['q_direct']:.{precision}g}.")
                if analysis["q_direct"] < 1:
                    print("     Vì q_T<1, ánh xạ x↦Tx+g là ánh xạ co ⇒ dãy Seidel hội tụ.")
                else:
                    print("     q_T≥1 nên tiêu chuẩn này không kết luận được hội tụ hay phân kỳ.")

                print("\n  3) Tiêu chuẩn Sassenfeld:")
                print("     β_i=[Σ_(j<i)|a_ij|β_j + Σ_(j>i)|a_ij|]/|a_ii|,")
                print("     β=max_i β_i.")
                print(
                    "     " + ", ".join(
                        f"β{to_subscript(i + 1)}={value:.{precision}g}"
                        for i, value in enumerate(analysis["beta"])
                    )
                )
                print(f"     β={analysis['q_sassenfeld']:.{precision}g}.")
                if analysis["q_sassenfeld"] < 1:
                    print("     Vì β<1, tiêu chuẩn Sassenfeld bảo đảm Gauss–Seidel hội tụ.")
                else:
                    print("     β≥1 nên Sassenfeld cũng không đưa ra kết luận.")

                if analysis["column_data"] is not None and analysis["column_data"]["valid"]:
                    data = analysis["column_data"]
                    print("\n  4) Chuẩn 1 có trọng số sau đổi biến y_i=a_ii·x_i:")
                    print("     ℓ_j là tổng hệ số phía dưới, u_j là tổng hệ số phía trên của cột j;")
                    print("     w_j=1-ℓ_j và q_w=max_j(u_j/w_j).")
                    print(f"     q_w={data['q']:.{precision}g}<1 ⇒ ánh xạ co trong chuẩn ||·||_w, nên hội tụ.")
                elif analysis["q_direct"] >= 1 and analysis["q_sassenfeld"] >= 1:
                    print("\n  Không tiêu chuẩn nào ở trên chứng nhận hội tụ.")
                    print("  Phép lặp vẫn tính được vì a_ii≠0, nhưng chỉ là thử số và không được khẳng định hội tụ.")

        print("\nPHẦN 3. CÔNG THỨC LẶP")
        print("Phân tách A=D+L_A+U_A:")
        print("  D: phần đường chéo của A; L_A: phần tam giác dưới; U_A: phần tam giác trên.")
        print_matrix("D", analysis["D"], precision)
        print_matrix("L_A", analysis["L_A"], precision)
        print_matrix("U_A", analysis["U_A"], precision)
        print_matrix("T=-(D+L_A)^(-1)U_A", analysis["T"], precision)
        print_matrix("g=(D+L_A)^(-1)b", analysis["g"], precision)
        print("Dạng ma trận: x^(k+1)=T·x^(k)+g.")
        print("  T là ma trận lặp, g là vector hằng; cả hai chỉ phụ thuộc A và b.")

        if analysis["variable"] == "y":
            data = analysis["column_data"]
            print("\nDo dùng nhánh chéo trội cột, đặt y_i=a_ii*x_i.")
            print("Hệ tương đương: A D^(-1)y=b và y=Cy+b,")
            print("trong đó C=I-A D^(-1).")
            print_matrix("C", data["C"], precision)
            print("ell_j:", ", ".join(f"{v:.{precision}g}" for v in data["ell"]))
            print("u_j:", ", ".join(f"{v:.{precision}g}" for v in data["u"]))
            print(
                "w_j=1-ell_j:", ", ".join(f"{v:.{precision}g}" for v in data["weights"])
            )
            print(f"q=max_j u_j/w_j={data['q']:.{precision}g}<1.")
            print("Chuẩn dùng: ||z||_w=Σ_j w_j|z_j|.")
            if stop_mode == "apriori":
                print("Chặn tiên nghiệm đã chọn:")
                print("  E_k ≤ s·q^k/(1-q)·||y^(1)-y^(0)||_w.")
            else:
                print("Chặn hậu nghiệm đã chọn:")
                print(
                    "  ||x^(k)-x*||_∞ ≤ [1/min_j(w_j|a_jj|)]·q/(1-q)·||y^(k)-y^(k-1)||_w."
                )
        elif theory:
            pdf_data = analysis.get("pdf_data", {})
            if theory["kind"] in {"pdf_inf", "pdf_one"}:
                print("\nChế độ trình bày theo PDF:")
                print("  T = diag(1/a₁₁,...,1/aₙₙ), C = I - T·A, d = T·b.")
                print_matrix("C=I-T·A", pdf_data["C_pdf"], precision)
                print_matrix("d=T·b", pdf_data["d_pdf"], precision)
                print_matrix("L", pdf_data["L_pdf"], precision)
                print_matrix("U", pdf_data["U_pdf"], precision)
                if theory["kind"] == "pdf_inf":
                    print("  A chéo trội hàng: chọn chuẩn ||·||∞, s=0.")
                    print("  q=max_i Σ_(j>i)|c_ij|/(1-Σ_(j<i)|c_ij|).")
                    print(f"  q={theory['q']:.{precision}g}, s=0.")
                else:
                    print("  A chéo trội cột: chọn chuẩn ||·||₁.")
                    print("  q=max_j Σ_(i<j)|c_ij|/(1-Σ_(i>j)|c_ij|),")
                    print("  s=max_j Σ_(i>j)|c_ij|.")
                    print(f"  q={theory['q']:.{precision}g}, s={theory['s']:.{precision}g}.")
                if stop_mode == "apriori":
                    print("Chặn tiên nghiệm theo PDF:")
                    if theory["kind"] == "pdf_inf":
                        print("  E_k ≤ q^k/(1-q)·||x^(1)-x^(0)||∞.")
                    else:
                        print("  E_k ≤ q^k/((1-q)(1-s))·||x^(1)-x^(0)||₁.")
                else:
                    print("Chặn hậu nghiệm theo PDF:")
                    if theory["kind"] == "pdf_inf":
                        print("  Δ = q/(1-q)·||x^(k)-x^(k-1)||∞, dừng khi Δ≤ε.")
                    else:
                        print("  Δ = q/((1-q)(1-s))·||x^(k)-x^(k-1)||₁, dừng khi Δ≤ε.")
                print("  Phần dư r=b-Ax chỉ in để kiểm tra, không dùng làm điều kiện dừng PDF.")
            elif analysis["row_dom"]:
                print("\nA đã chéo trội hàng nên hội tụ đã được bảo đảm.")
                print(
                    f"Để lập chặn sai số, dùng q=||T||∞={analysis['q_direct']:.{precision}g}<1; "
                    "q ở đây dùng cho công thức sai số, không phải xét thêm điều kiện hội tụ."
                )
                if stop_mode == "apriori":
                    print("Chặn tiên nghiệm đã chọn:")
                    print("  ||x^(k)-x*||_∞ ≤ q^k/(1-q)·||x^(1)-x^(0)||_∞.")
                else:
                    print("Chặn hậu nghiệm đã chọn:")
                    print("  ||x^(k)-x*||_∞ ≤ q/(1-q)·||x^(k)-x^(k-1)||_∞.")
            else:
                print("\nVì ||T||∞<1, phép lặp là ánh xạ co.")
                if stop_mode == "apriori":
                    print("Chặn tiên nghiệm đã chọn:")
                    print("  ||x^(k)-x*||_∞ ≤ q^k/(1-q)·||x^(1)-x^(0)||_∞.")
                else:
                    print("Chặn hậu nghiệm đã chọn:")
                    print("  ||x^(k)-x*||_∞ ≤ q/(1-q)·||x^(k)-x^(k-1)||_∞.")
        elif analysis["spd"]:
            print(
                "\nSPD bảo đảm dãy Seidel hội tụ, nhưng bản này không giả tạo một chặn"
                " sai số ∞ khi chưa có hằng số thích hợp."
            )
        else:
            print(
                "\nChưa có điều kiện đủ hoặc chặn co để bảo đảm hội tụ với mọi x^(0)."
            )

        print("\nCông thức Seidel tổng quát theo từng ẩn:")
        print("  x_i^(k+1) = [b_i - Σ_(j=1..i-1)a_ij·x_j^(k+1)")
        print("                         - Σ_(j=i+1..n)a_ij·x_j^(k)] / a_ii.")
        print("  Các x_j^(k+1), j<i vừa tính trong vòng hiện tại được dùng ngay;")
        print("  các x_j^(k), j>i chưa cập nhật nên dùng giá trị của vòng trước.")
        print("\nThay số vào từng công thức:")
        for i in range(n):
            lower_terms = " ".join(
                f"-({a[i, j]:.{precision}g})x_{j + 1}^(k+1)" for j in range(i)
            )
            upper_terms = " ".join(
                f"-({a[i, j]:.{precision}g})x_{j + 1}^(k)" for j in range(i + 1, n)
            )
            terms = " ".join(item for item in (lower_terms, upper_terms) if item)
            print(
                f"  x_{i + 1}^(k+1)=({b[i]:.{precision}g} {terms})/({a[i, i]:.{precision}g})"
            )

        print("\nPHẦN 4. BẢNG LẶP")
        print("Ký hiệu trong bảng:")
        print("  ||Δx||∞ = ||x^(k)-x^(k-1)||∞: độ thay đổi giữa hai vòng liên tiếp.")
        print("  r=b-Ax^(k), ||r||∞ là chuẩn phần dư (không đồng nhất với sai số nghiệm).")
        print("  eta=||r||∞/(||A||∞·||x^(k)||∞+||b||∞): phần dư tương đối.")
        print("  E_k: chặn trên lý thuyết của ||x^(k)-x*||∞ theo chế độ đã chọn.")
        width = max(12, precision + 8)
        error_header = "E_k tiên nghiệm" if stop_mode == "apriori" else "E_k hậu nghiệm"
        status_width = 12
        header = f"| {'k':>4} |" + "".join(
            f" {('x_' + str(i + 1)):>{width}} |" for i in range(n)
        )
        header += (
            f" {'||Δx||∞':>{width}} | {'||r||∞':>{width}}"
            f" | {'eta':>{width}} | {error_header:>{width}} | {'trạng thái':>{status_width}} |"
        )
        border = "+" + "-" * (len(header) - 2) + "+"
        print(border)
        print(header)
        print(border)
        residual0 = vector_norm_inf(b - a @ x)
        eta0 = relative_residual(a, x, b)
        print(
            f"| {0:4d} |"
            + "".join(
                f" {format_table_number(value, precision):>{width}} |" for value in x
            )
            + f" {'-':>{width}} | {residual0:{width}.{precision}e}"
            f" | {eta0:{width}.{precision}e} | {'-':>{width}} | {'khởi tạo':>{status_width}} |"
        )
        if show_y and analysis["variable"] == "y":
            print_matrix("y^(0)", y, precision)

    xs = [x.copy()]
    residuals = [vector_norm_inf(b - a @ x)]
    converged = False
    certified = False
    numerical_stop = False
    fixed_completed = fixed_iterations == 0
    reason = ""
    error_bound = math.nan
    apriori_iterations = None
    first_difference = None
    first_scale = 1.0
    step_inf = 0.0
    k = 0

    if residuals[0] == 0:
        converged = True
        certified = True
        reason = "x^(0) thỏa Ax=b trong số học máy."

    loop_limit = fixed_iterations if fixed_iterations is not None else max_iter
    if fixed_iterations == 0:
        reason = "Đã thực hiện đúng 0 bước theo yêu cầu."

    for k in range(1, loop_limit + 1):
        old_x = x.copy()
        old_y = y.copy()

        if analysis["variable"] == "y":
            for i in range(n):
                y[i] = (
                    analysis["d"][i]
                    + analysis["L"][i, :i] @ y[:i]
                    + analysis["U"][i, i + 1 :] @ old_y[i + 1 :]
                )
            x = y / diagonal
        else:
            for i in range(n):
                lower_value = a[i, :i] @ x[:i]
                upper_value = a[i, i + 1 :] @ old_x[i + 1 :]
                x[i] = (b[i] - lower_value - upper_value) / a[i, i]
            y = x.copy()

        if not np.all(np.isfinite(x)):
            reason = "Xuất hiện NaN hoặc vô cùng."
            break
        if vector_norm_inf(x) > HUGE_VALUE:
            reason = "Giá trị lặp tăng quá lớn."
            break

        step_inf = vector_norm_inf(x - old_x)
        residual = vector_norm_inf(b - a @ x)
        eta = relative_residual(a, x, b)

        if theory:
            if theory["kind"] in {"x_inf", "pdf_inf"}:
                difference = step_inf
                scale = 1.0
                denominator_scale = 1.0
            elif theory["kind"] == "pdf_one":
                difference = vector_norm_1(x - old_x)
                scale = 1.0 / (1.0 - theory["s"])
                denominator_scale = scale
            else:
                difference = weighted_one_norm(y - old_y, theory["weights"])
                scale = theory["scale_to_x_inf"]
                denominator_scale = scale
            error_bound = theory["coefficient"] * difference
            if first_difference is None:
                first_difference = difference
                first_scale = denominator_scale
                apriori_iterations = estimate_apriori_iterations(
                    theory["q"], first_difference, first_scale, epsilon
                )
            if stop_mode == "apriori":
                error_bound = (
                    first_scale
                    * theory["q"] ** k
                    / (1 - theory["q"])
                    * first_difference
                )
        else:
            error_bound = math.nan

        state = "tiếp tục"
        posteriori_reached = stop_mode == "posteriori" and error_bound <= epsilon
        apriori_reached = (
            stop_mode == "apriori"
            and apriori_iterations is not None
            and k >= apriori_iterations
        )
        if fixed_iterations is None and theory and (posteriori_reached or apriori_reached):
            converged = True
            certified = True
            reason = "Chặn hậu nghiệm không vượt quá epsilon."
            state = "đạt chặn"
            if apriori_reached:
                reason = "Chặn tiên nghiệm không vượt quá epsilon."
        elif fixed_iterations is None and not theory:
            scaled_step = epsilon * max(1.0, vector_norm_inf(x))
            if step_inf <= scaled_step and eta <= residual_tolerance:
                converged = True
                numerical_stop = True
                reason = (
                    "Đạt tiêu chuẩn dừng số (bước lặp tương đối và phần dư tương đối); "
                    "không coi đây là chặn sai số nghiệm."
                )
                state = "đạt số"

        if show:
            width = max(12, precision + 8)
            e_text = (
                f"{error_bound:.{precision}e}" if math.isfinite(error_bound) else "N/A"
            )
            print(
                f"| {k:4d} |"
                + "".join(
                    f" {format_table_number(value, precision):>{width}} |" for value in x
                )
                + f" {step_inf:{width}.{precision}e} | {residual:{width}.{precision}e}"
                f" | {eta:{width}.{precision}e} | {e_text:>{width}} | {state:>{status_width}} |"
            )
            if show_y and analysis["variable"] == "y":
                print_matrix(f"y^({k})", y, precision)
            if k == 1 and theory and stop_mode == "apriori":
                print("  Chặn tiên nghiệm:")
                print("    E_k <= s·q^k/(1-q)·||z^(1)-z^(0)||,")
                print(
                    f"    q={theory['q']:.{precision}g}, s={first_scale:.{precision}g}."
                )
                if apriori_iterations is not None:
                    print(
                        f"    Suy ra k >= {apriori_iterations} để chặn không vượt epsilon."
                    )

        xs.append(x.copy())
        residuals.append(residual)
        if converged and fixed_iterations is None:
            break
        diagnosis = _diagnose(xs, residuals, epsilon)
        if fixed_iterations is None and diagnosis:
            reason = diagnosis
            break
    else:
        if fixed_iterations is not None:
            fixed_completed = True
            reason = f"Đã thực hiện đúng {fixed_iterations} bước."
        elif not converged:
            reason = "Hết max_iter nhưng chưa đạt điều kiện dừng."

    final_residual_vector = b_original - a_original @ x
    final_residual = vector_norm_inf(final_residual_vector)
    final_eta = relative_residual(a_original, x, b_original)

    info = {
        **prep,
        "converged": converged,
        "certified": certified,
        "numerical_stop": numerical_stop,
        "fixed_iterations_completed": fixed_completed,
        "iterations": k,
        "error_bound": error_bound,
        "residual": final_residual,
        "relative_residual": final_eta,
        "reason": reason,
        "apriori_iterations": apriori_iterations,
        "iteration_method": analysis["method"],
        "theory": theory,
        "q": theory["q"] if theory else math.nan,
    }

    if show:
        print(border)
        print("\nPHẦN 5. ĐIỀU KIỆN DỪNG VÀ KẾT QUẢ")
        if fixed_completed:
            print(reason)
            print(
                "Đây là giá trị sau đúng số bước yêu cầu, không tự động đồng nghĩa đã hội tụ."
            )
            if math.isfinite(error_bound):
                print(
                    f"Vì có hệ số co q<1, chặn sai số sau k bước là "
                    f"E_k={error_bound:.{precision}e}."
                )
            else:
                print(
                    "Chưa có chặn sai số nghiệm theo chuẩn ∞ cho kết quả sau k bước "
                    "trong nhánh hiện tại."
                )
        elif certified:
            print(reason)
            print(
                f"E_k={error_bound:.{precision}e} <= epsilon={epsilon:.{precision}e}."
            )
        elif numerical_stop:
            print(reason)
        else:
            print("Chưa xác nhận đạt yêu cầu:", reason)
        print_matrix(f"x^({k})", x, precision)
        print_matrix("A x^(k)", a_original @ x, precision)
        print_matrix("b-A x^(k)", final_residual_vector, precision)
        print(f"||b-Ax^(k)||_∞={final_residual:.{precision}e}")
        print(f"eta={final_eta:.{precision}e}")
        if math.isfinite(error_bound):
            print(f"Chặn sai số nghiệm theo chuẩn ∞: E_k={error_bound:.{precision}e}")
        else:
            print("Không có chặn sai số nghiệm theo chuẩn ∞ trong nhánh hiện tại.")

    return x, info


# ============================================================================
# NHIỀU VẾ PHẢI VÀ NGHỊCH ĐẢO
# ============================================================================


def solve_multiple_rhs(
    a,
    big_b,
    x0=None,
    epsilon=1e-6,
    max_iter=500,
    precision=6,
    auto_reorder=True,
    show=True,
    fixed_iterations=None,
    use_inverse_bound=False,
    residual_tolerance=None,
    stop_mode="posteriori",
):
    del use_inverse_bound
    a = np.asarray(a, dtype=float)
    big_b = np.asarray(big_b, dtype=float)
    validate_system(a)
    if big_b.ndim != 2 or big_b.shape[0] != len(a) or not np.all(np.isfinite(big_b)):
        raise ValueError("B sai kích thước hoặc chứa NaN/vô cùng.")
    initial = np.zeros(len(a)) if x0 is None else np.asarray(x0, dtype=float)
    if initial.shape != (len(a),):
        raise ValueError("x0 sai kích thước.")

    columns = []
    infos = []
    for j in range(big_b.shape[1]):
        if show:
            print(f"\n{'#' * 28} VẾ PHẢI {j + 1}/{big_b.shape[1]} {'#' * 28}")
        solution, info = gauss_seidel(
            a,
            big_b[:, j],
            initial,
            epsilon,
            max_iter,
            precision,
            auto_reorder,
            show,
            fixed_iterations,
            residual_tolerance=residual_tolerance,
            stop_mode=stop_mode,
        )
        columns.append(solution)
        infos.append(info)
    result = np.column_stack(columns)
    if show:
        print_matrix("Ma trận nghiệm X", result, precision)
        print(f"||B-AX||_∞={matrix_norm_inf(big_b - a @ result):.{precision}e}")
    return result, infos


def solve_inverse(
    a,
    x0=None,
    epsilon=1e-6,
    max_iter=500,
    precision=6,
    auto_reorder=True,
    show=True,
    fixed_iterations=None,
    use_inverse_bound=False,
    residual_tolerance=None,
    stop_mode="posteriori",
):
    del use_inverse_bound
    a = np.asarray(a, dtype=float)
    validate_system(a)
    n = len(a)
    initial = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)
    inverse, infos = solve_multiple_rhs(
        a,
        np.eye(n),
        initial,
        epsilon,
        max_iter,
        precision,
        auto_reorder,
        show,
        fixed_iterations,
        residual_tolerance=residual_tolerance,
        stop_mode=stop_mode,
    )
    left_error = matrix_norm_inf(a @ inverse - np.eye(n))
    right_error = matrix_norm_inf(inverse @ a - np.eye(n))
    all_converged = all(info["converged"] for info in infos)
    all_certified = all(info["certified"] for info in infos)
    if show:
        print("\nKẾT QUẢ TÌM NGHỊCH ĐẢO")
        print_matrix("A^(-1) gần đúng", inverse, precision)
        print(f"||AA^(-1)-I||_∞={left_error:.{precision}e}")
        print(f"||A^(-1)A-I||_∞={right_error:.{precision}e}")
        if all_certified:
            print("Mọi cột đều đạt chặn sai số lý thuyết.")
        elif all_converged:
            print(
                "Mọi cột đạt tiêu chuẩn dừng số, nhưng không phải cột nào cũng có chặn lý thuyết."
            )
        else:
            print("CẢNH BÁO: ít nhất một cột chưa đạt điều kiện dừng.")
    return inverse, {
        "columns": infos,
        "left_error": left_error,
        "right_error": right_error,
        "converged": all_converged,
        "certified": all_certified,
    }


# ============================================================================
# GIAO DIỆN
# ============================================================================


def print_exam_overview(task_choice):
    print("\n" + "=" * 92)
    print("THUẬT TOÁN GAUSS–SEIDEL")
    print("=" * 92)
    print("Input:")
    if task_choice == "1":
        print("  • Ma trận vuông A, vector b, vector đầu x^(0).")
    elif task_choice == "2":
        print("  • Ma trận B, vector d và x^(0) của dạng đề x=B x+d.")
        print("  • Không yêu cầu nhập A hoặc b và không đổi bài toán sang (I-B)x=d.")
    elif task_choice == "3":
        print("  • Ma trận khả nghịch A; giải lần lượt A x_j=e_j để ghép A^(-1).")
    else:
        print("  • Ma trận vuông A, ma trận nhiều vế phải B và vector đầu x^(0).")
    print("  • Sai số epsilon hoặc số bước lặp k theo yêu cầu đề bài.")
    print("Output:")
    print("  • Nghiệm gần đúng, bảng lặp, chặn sai số và phần dư kiểm tra.")
    if task_choice == "2":
        print("Các bước Seidel trực tiếp cho dạng điểm bất động:")
        print("  B1. Giữ nguyên dữ liệu B,d và tách B=L_B+D_B+U_B.")
        print("  B2. Tính tuần tự:")
        print("      x_i^(k+1)=d_i+Σ_(j<i)b_ij x_j^(k+1)+Σ_(j≥i)b_ij x_j^(k).")
        print("  B3. Kiểm tra ||x^(k)-Bx^(k)-d||∞ trên đúng bài toán đã cho.")
        return
    print("Các bước cho một vế phải Ax=b:")
    print("  B1. Phân tách A=D+L+U và kiểm tra điều kiện hội tụ.")
    print("  B2. Với i=1,...,n, tính tuần tự:")
    print("      x_i^(k+1)=[b_i-Σ_(j<i)a_ij x_j^(k+1)-Σ_(j>i)a_ij x_j^(k)]/a_ii.")
    print("  B3. Dùng ngay các thành phần mới x_j^(k+1), j<i.")
    print("  B4. Đánh giá sai số, phần dư r^(k)=b-Ax^(k), rồi kiểm tra dừng.")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    print("GAUSS–SEIDEL – BÀI GIẢI CHI TIẾT")
    print("1. Giải hệ Ax=b")
    print("2. Seidel trực tiếp cho x=Bx+d")
    print("3. Tìm A^(-1) bằng Seidel (giải AX=I)")
    print("4. Giải nhiều vế phải AX=B")
    print("0. Thoát")
    choice = input("Chọn [1]: ").strip() or "1"
    if choice == "0":
        return
    if choice not in {"1", "2", "3", "4"}:
        print("Lựa chọn không hợp lệ.")
        return

    print_exam_overview(choice)

    try:
        if choice == "2":
            n = read_int("Cấp n = ", minimum=1)
            b_matrix = read_matrix(n, n, "B")
            d = read_vector(n, f"Nhập d ({n} phần tử): ")
            x0 = read_vector(n, f"Nhập x^(0) ({n} phần tử, Enter=0): ", default=0.0)
            print("Chế độ chạy:")
            print("1. Thực hiện đúng k bước")
            print("2. Lặp đến sai số epsilon")
            direct_mode = input("Chọn [1]: ").strip() or "1"
            if direct_mode not in {"1", "2"}:
                raise ValueError("Chế độ chạy không hợp lệ.")
            if direct_mode == "1":
                fixed_iterations = read_int("Số bước k = ", minimum=0)
                epsilon = 1e-6
                max_iter = max(1, fixed_iterations)
                stop_mode = "both"
                residual_tolerance = epsilon
            else:
                epsilon = read_float("epsilon [1e-6] = ", positive=True, default=1e-6)
                max_iter = read_int("max_iter [500] = ", minimum=1, default=500)
                fixed_iterations = None
                print("Điều kiện dừng:")
                print("1. Sai khác tuyệt đối")
                print("2. Sai khác tương đối")
                print("3. Residual điểm bất động")
                print("4. Đồng thời sai khác tuyệt đối và residual")
                direct_stop = input("Chọn [4]: ").strip() or "4"
                stop_mode = {
                    "1": "absolute", "2": "relative", "3": "residual", "4": "both"
                }.get(direct_stop)
                if stop_mode is None:
                    raise ValueError("Điều kiện dừng không hợp lệ.")
                residual_tolerance = epsilon
            precision = read_int("Số chữ số sau dấu phẩy [7] = ", minimum=0, default=7)
            seidel_fixed_point(
                b_matrix,
                d,
                x0=x0,
                epsilon=epsilon,
                max_iter=max_iter,
                precision=precision,
                fixed_iterations=fixed_iterations,
                stop_mode=stop_mode,
                residual_tolerance=residual_tolerance,
            )
            return

        n = read_int("Cấp ma trận n = ", minimum=1)
        a = read_matrix(n, n, "A")
        auto_reorder = input(
            "Tự động đổi hàng nếu có lợi? [C/k]: "
        ).strip().lower() not in {"k", "khong", "không", "n", "no"}
        precision = read_int("Số chữ số sau dấu phẩy [7] = ", minimum=0, default=7)

        print("Chế độ dừng:")
        print("1. Chặn sai số hậu nghiệm")
        print("2. Chặn sai số tiên nghiệm")
        print("3. Số chữ số thập phân chính xác")
        print("4. Thực hiện đúng k bước")
        stop_choice = input("Chọn [1]: ").strip() or "1"
        if stop_choice not in {"1", "2", "3", "4"}:
            raise ValueError("Chế độ dừng không hợp lệ.")
        stop_mode = "posteriori"
        if stop_choice in {"1", "2"}:
            epsilon = read_float("epsilon [1e-6] = ", positive=True, default=1e-6)
            max_iter = read_int("max_iter [500] = ", minimum=1, default=500)
            fixed_iterations = None
            residual_tolerance = epsilon
            stop_mode = "posteriori" if stop_choice == "1" else "apriori"
        elif stop_choice == "3":
            exact_digits = read_int("Số chữ số thập phân chính xác d = ", minimum=0)
            epsilon = 0.5 * 10.0 ** (-exact_digits)
            print(f"Chọn epsilon = 0.5·10^(-{exact_digits}) = {epsilon:.6e}.")
            max_iter = read_int("max_iter [500] = ", minimum=1, default=500)
            fixed_iterations = None
            residual_tolerance = epsilon
        else:
            fixed_iterations = read_int("Số bước k = ", minimum=0)
            epsilon = read_float(
                "epsilon để tham khảo [1e-6] = ", positive=True, default=1e-6
            )
            max_iter = max(1, fixed_iterations)
            residual_tolerance = epsilon

        if choice == "1":
            b = read_vector(n, f"Nhập b ({n} phần tử): ")
            x0 = read_vector(n, f"Nhập x^(0) ({n} phần tử, Enter=0): ", default=0.0)
            gauss_seidel(
                a,
                b,
                x0,
                epsilon,
                max_iter,
                precision,
                auto_reorder,
                fixed_iterations=fixed_iterations,
                residual_tolerance=residual_tolerance,
                stop_mode=stop_mode,
            )
        elif choice == "4":
            m = read_int("Số vế phải m = ", minimum=1)
            big_b = read_matrix(n, m, "B")
            x0 = read_vector(n, f"x^(0) dùng chung ({n} số, Enter=0): ", default=0.0)
            solve_multiple_rhs(
                a,
                big_b,
                x0,
                epsilon,
                max_iter,
                precision,
                auto_reorder,
                fixed_iterations=fixed_iterations,
                residual_tolerance=residual_tolerance,
                stop_mode=stop_mode,
            )
        else:  # choice == "3"
            x0 = read_vector(n, f"x^(0) dùng chung ({n} số, Enter=0): ", default=0.0)
            solve_inverse(
                a,
                x0,
                epsilon,
                max_iter,
                precision,
                auto_reorder,
                fixed_iterations=fixed_iterations,
                residual_tolerance=residual_tolerance,
                stop_mode=stop_mode,
            )
    except (ValueError, ArithmeticError) as exc:
        print(f"\nKhông thể thực hiện: {exc}")
    except (EOFError, KeyboardInterrupt):
        print("\nĐã dừng chương trình.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã dừng chương trình; không có dữ liệu đầu vào đầy đủ.")
