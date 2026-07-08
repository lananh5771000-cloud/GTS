"""Tìm gần đúng ma trận nghịch đảo bằng lặp Newton–Schulz.

Thuật toán chính duy nhất:
    X_(k+1) = X_k (2I - A X_k) = (2I - X_k A) X_k.

Chương trình dùng chuẩn phổ (chuẩn 2) để chứng nhận hội tụ. Chuẩn 2 được
ước lượng kèm cận trên an toàn bằng phép quay Jacobi trên M^T M; không dùng
numpy.linalg.inv/solve/eig/svd.
"""

import math
import sys
from fractions import Fraction


reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure_stdout):
    reconfigure_stdout(encoding="utf-8", errors="replace")


MACHINE_EPS = sys.float_info.epsilon
MAX_MEANINGFUL_DECIMALS = 15


# ============================================================
# NHẬP DỮ LIỆU
# ============================================================


def input_positive_integer(prompt, default=None):
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "" and default is not None:
                return default
            value = int(raw)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Lỗi: Vui lòng nhập một số nguyên dương.")


def input_nonnegative_integer(prompt, default=None):
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "" and default is not None:
                return default
            value = int(raw)
            if value < 0:
                raise ValueError
            return value
        except ValueError:
            print("Lỗi: Vui lòng nhập một số nguyên không âm.")


def input_positive_number(prompt, default=None):
    while True:
        token = input(prompt).strip()
        if token == "" and default is not None:
            return default
        try:
            value = float(Fraction(token))
            if not math.isfinite(value) or value <= 0:
                raise ValueError
            return value
        except (ValueError, ZeroDivisionError, OverflowError):
            print(
                "Lỗi: Vui lòng nhập một số dương hữu hạn, "
                "ví dụ 1e-8, 0.0001 hoặc 1/1000."
            )


def input_matrix_row(prompt, expected_count):
    while True:
        tokens = input(prompt).split()
        if len(tokens) != expected_count:
            print(
                f"Lỗi: Dòng phải có đúng {expected_count} phần tử. Vui lòng nhập lại."
            )
            continue
        try:
            return [Fraction(token) for token in tokens]
        except (ValueError, ZeroDivisionError):
            print(
                "Lỗi: Chỉ nhập số nguyên, số thập phân hoặc phân số hợp lệ "
                "(ví dụ 2, -3, 0.25, 1/3)."
            )


def input_matrix_exact(name, rows, columns):
    print(f"\nNhập ma trận {name} ({rows}x{columns}):")
    return [
        input_matrix_row(
            f"Nhập dòng {i + 1} (cách nhau bởi khoảng trắng): ",
            columns,
        )
        for i in range(rows)
    ]


def choose_initial_approximation():
    print("\nChọn xấp xỉ đầu X_0:")
    print("1. Tự động: X_0 = A^T/(||A||_1 ||A||_∞)")
    print("2. Tự nhập ma trận X_0")
    print("0. Thoát")
    while True:
        choice = input("Chọn [Enter = 1]: ").strip() or "1"
        if choice in {"0", "1", "2"}:
            return choice
        print("Lỗi: Vui lòng chỉ chọn 0, 1 hoặc 2.")


# ============================================================
# HIỂN THỊ
# ============================================================


def exact_number(value):
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def exact_matrix_lines(matrix):
    if not matrix:
        return ["[]"]
    values = [[exact_number(value) for value in row] for row in matrix]
    widths = [
        max(len(values[i][j]) for i in range(len(values)))
        for j in range(len(values[0]))
    ]
    return [
        "[" + "  ".join(values[i][j].rjust(widths[j]) for j in range(len(widths))) + "]"
        for i in range(len(values))
    ]


def decimal_matrix_lines(matrix, decimals):
    threshold = 0.5 * 10.0 ** (-decimals) if decimals > 0 else 0.5
    lines = []
    for row in matrix:
        values = []
        for value in row:
            shown = 0.0 if abs(value) < threshold else float(value)
            values.append(f"{shown:14.{decimals}f}")
        lines.append("[" + "  ".join(values) + "]")
    return lines


def print_exact_matrix(matrix, prefix=""):
    lines = exact_matrix_lines(matrix)
    middle = len(lines) // 2
    padding = " " * len(prefix)
    for i, line in enumerate(lines):
        print((prefix if i == middle else padding) + line)


def print_decimal_matrix(matrix, decimals, prefix=""):
    lines = decimal_matrix_lines(matrix, decimals)
    middle = len(lines) // 2
    padding = " " * len(prefix)
    for i, line in enumerate(lines):
        print((prefix if i == middle else padding) + line)


def print_two_decimal_matrices(
    left, right, decimals, left_name="A = ", right_name="B = "
):
    left_lines = decimal_matrix_lines(left, decimals)
    right_lines = decimal_matrix_lines(right, decimals)
    row_count = max(len(left_lines), len(right_lines))
    middle = row_count // 2
    left_width = max(len(line) for line in left_lines)
    for i in range(row_count):
        left_line = left_lines[i] if i < len(left_lines) else ""
        right_line = right_lines[i] if i < len(right_lines) else ""
        left_prefix = left_name if i == middle else " " * len(left_name)
        right_prefix = right_name if i == middle else " " * len(right_name)
        print(
            (left_prefix + left_line).ljust(len(left_name) + left_width + 7)
            + right_prefix
            + right_line
        )


def print_three_decimal_matrices(first, second, third, decimals, names):
    """In ba ma trận cạnh nhau để một bước lặp gọn như bài làm trên giấy."""
    blocks = [decimal_matrix_lines(matrix, decimals) for matrix in (first, second, third)]
    widths = [max(len(line) for line in block) for block in blocks]
    row_count = max(len(block) for block in blocks)
    middle = row_count // 2
    for i in range(row_count):
        parts = []
        for block, width, name in zip(blocks, widths, names):
            line = block[i] if i < len(block) else ""
            prefix = name if i == middle else " " * len(name)
            parts.append((prefix + line).ljust(len(name) + width))
        print("     ".join(parts).rstrip())


def print_iteration_summary(records, precision, error_header):
    if not records:
        return
    headers = ["k", "q_k", error_header, "Trạng thái"]
    rows = []
    table_digits = min(6, max(2, precision))

    def compact(value):
        return f"{float(value):.{table_digits}e}"

    for item in records:
        rows.append(
            [
                str(item["k"]),
                compact(item["q"]),
                compact(item["selected_error"]),
                item["status"],
            ]
        )
    widths = [max(len(headers[j]), *(len(row[j]) for row in rows)) for j in range(len(headers))]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print("\nBảng tóm tắt quá trình lặp:")
    print(border)
    print("| " + " | ".join(headers[j].center(widths[j]) for j in range(len(headers))) + " |")
    print(border)
    for row in rows:
        print("| " + " | ".join(row[j].rjust(widths[j]) for j in range(len(headers))) + " |")
    print(border)


def scientific(value):
    return f"{float(value):.12e}"


# ============================================================
# PHÉP TOÁN MA TRẬN
# ============================================================


def exact_to_float(matrix):
    result = []
    for row in matrix:
        numeric_row = []
        for value in row:
            converted = float(value)
            if not math.isfinite(converted):
                raise OverflowError("Có phần tử quá lớn để biểu diễn bằng số thực máy.")
            numeric_row.append(converted)
        result.append(numeric_row)
    return result


def copy_matrix(matrix):
    return [row[:] for row in matrix]


def identity_matrix(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose_matrix(matrix):
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]


def subtract_matrices(left, right):
    return [
        [left[i][j] - right[i][j] for j in range(len(left[0]))]
        for i in range(len(left))
    ]


def scalar_multiply(matrix, scalar):
    return [[scalar * value for value in row] for row in matrix]


def multiply_matrices(left, right):
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("Kích thước hai ma trận không phù hợp để nhân.")
    rows = len(left)
    middle = len(right)
    columns = len(right[0])
    result = []
    for i in range(rows):
        row = []
        for j in range(columns):
            value = math.fsum(left[i][k] * right[k][j] for k in range(middle))
            if not math.isfinite(value):
                raise FloatingPointError("Phép nhân ma trận tạo ra NaN hoặc vô cực.")
            row.append(value)
        result.append(row)
    return result


def all_finite(matrix):
    return all(math.isfinite(value) for row in matrix for value in row)


def matrix_norm_one(matrix):
    rows = len(matrix)
    columns = len(matrix[0])
    return max(
        math.fsum(abs(matrix[i][j]) for i in range(rows)) for j in range(columns)
    )


def matrix_norm_infinity(matrix):
    return max(math.fsum(abs(value) for value in row) for row in matrix)


def matrix_norm_frobenius(matrix):
    return math.sqrt(math.fsum(value * value for row in matrix for value in row))


def round_matrix(matrix, decimals):
    rounded = []
    for row in matrix:
        rounded_row = []
        for value in row:
            item = round(float(value), decimals)
            rounded_row.append(0.0 if item == 0.0 else item)
        rounded.append(rounded_row)
    return rounded


# ============================================================
# KIỂM TRA KHẢ NGHỊCH CHÍNH XÁC
# ============================================================


def exact_rank(matrix):
    """Khử Gauss trên Fraction; không phân loại sai do ngưỡng số thực."""
    work = [row[:] for row in matrix]
    rows = len(work)
    columns = len(work[0])
    pivot_row = 0
    for column in range(columns):
        chosen = next((r for r in range(pivot_row, rows) if work[r][column] != 0), None)
        if chosen is None:
            continue
        if chosen != pivot_row:
            work[pivot_row], work[chosen] = work[chosen], work[pivot_row]
        pivot = work[pivot_row][column]
        for row in range(pivot_row + 1, rows):
            if work[row][column] == 0:
                continue
            factor = work[row][column] / pivot
            for j in range(column, columns):
                work[row][j] -= factor * work[pivot_row][j]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


# ============================================================
# CHUẨN 2: PHÉP QUAY JACOBI + CẬN GERSHGORIN
# ============================================================


def _symmetric_eigenvalue_bounds(
    matrix, relative_tolerance=2e-15, maximum_rotations=None
):
    """Cận dưới/trên của lambda_max cho ma trận đối xứng.

    Sau mỗi phép quay trực giao, max đường chéo là cận dưới Rayleigh;
    max_i(a_ii + sum_{j!=i}|a_ij|) là cận trên Gershgorin.
    """
    n = len(matrix)
    if n == 1:
        value = float(matrix[0][0])
        return value, value, 0, True

    work = copy_matrix(matrix)
    scale = max(1.0, matrix_norm_frobenius(work))
    if maximum_rotations is None:
        maximum_rotations = max(200, 200 * n * n)

    converged = False
    rotations = 0

    for rotations in range(maximum_rotations + 1):
        lower = max(work[i][i] for i in range(n))
        upper = max(
            work[i][i] + math.fsum(abs(work[i][j]) for j in range(n) if j != i)
            for i in range(n)
        )
        if upper - lower <= relative_tolerance * max(1.0, abs(upper), scale):
            converged = True
            break

        p, q = 0, 1
        largest = abs(work[p][q])
        for i in range(n):
            for j in range(i + 1, n):
                candidate = abs(work[i][j])
                if candidate > largest:
                    largest = candidate
                    p, q = i, j

        if largest <= relative_tolerance * scale:
            converged = True
            break

        app = work[p][p]
        aqq = work[q][q]
        apq = work[p][q]
        if apq == 0.0:
            continue

        tau = (aqq - app) / (2.0 * apq)
        if tau == 0.0:
            t = 1.0
        else:
            t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        for k in range(n):
            if k in (p, q):
                continue
            akp = work[k][p]
            akq = work[k][q]
            new_kp = c * akp - s * akq
            new_kq = s * akp + c * akq
            work[k][p] = work[p][k] = new_kp
            work[k][q] = work[q][k] = new_kq

        work[p][p] = app - t * apq
        work[q][q] = aqq + t * apq
        work[p][q] = work[q][p] = 0.0

    lower = max(work[i][i] for i in range(n))
    upper = max(
        work[i][i] + math.fsum(abs(work[i][j]) for j in range(n) if j != i)
        for i in range(n)
    )
    return lower, upper, rotations, converged


def matrix_norm_two_bounds(matrix):
    """Trả estimate, cận trên, số phép quay, trạng thái hội tụ.

    Cận trên được nới rất nhẹ để hạn chế đánh giá thiếu vì làm tròn khi lập M^T M.
    """
    gram = multiply_matrices(transpose_matrix(matrix), matrix)
    lower_eig, upper_eig, rotations, converged = _symmetric_eigenvalue_bounds(gram)

    frob = matrix_norm_frobenius(matrix)
    rounding_guard = (
        128.0 * MACHINE_EPS * max(1, len(matrix), len(matrix[0])) * (frob * frob)
    )
    lower_eig = max(0.0, lower_eig)
    upper_eig = max(lower_eig, upper_eig + rounding_guard)

    lower_norm = math.sqrt(lower_eig)
    upper_norm = math.sqrt(upper_eig)
    estimate = 0.5 * (lower_norm + upper_norm)
    return {
        "estimate": estimate,
        "lower": lower_norm,
        "upper": upper_norm,
        "rotations": rotations,
        "converged": converged,
    }


# ============================================================
# KHỞI TẠO XẤP XỈ ĐẦU
# ============================================================


def automatic_initial_approximation(A):
    """X0 = alpha A^T, alpha = 1/(||A||1 ||A||inf).

    Nếu A vuông khả nghịch thì trong số học chính xác:
        ||I - A X0||2 < 1 và ||I - X0 A||2 < 1.
    """
    norm_one = matrix_norm_one(A)
    norm_infinity = matrix_norm_infinity(A)
    denominator = norm_one * norm_infinity
    if denominator == 0.0 or not math.isfinite(denominator):
        return None, norm_one, norm_infinity, None
    alpha = 1.0 / denominator
    X0 = scalar_multiply(transpose_matrix(A), alpha)
    return X0, norm_one, norm_infinity, alpha


# ============================================================
# LÝ THUYẾT VÀ ĐÁNH GIÁ
# ============================================================


def print_method(stop_mode):
    print("\n" + "=" * 92)
    print("BÀI GIẢI: TÌM MA TRẬN NGHỊCH ĐẢO BẰNG PHƯƠNG PHÁP NEWTON–SCHULZ")
    print("=" * 92)
    print("\nInput:")
    print("  • Ma trận vuông khả nghịch A, xấp xỉ đầu X_0.")
    print("  • Sai số epsilon hoặc số bước lặp k theo yêu cầu đề bài.")
    print("Output:")
    print("  • Ma trận X_k xấp xỉ A^(-1), chặn sai số và phần dư hai phía.")
    print("\nTa cần tìm X sao cho A·X=I, khi đó X=A^(-1).")
    print("\n1. Công thức lặp")
    print("   X_(k+1) = X_k·(2I-A·X_k).")
    print("   Trong mỗi bước, ta lần lượt tính A·X_k, M_k=2I-A·X_k rồi X_(k+1)=X_k·M_k.")
    print("\n2. Điều kiện hội tụ")
    print("   Đặt R_k=I-A·X_k và S_k=I-X_k·A.")
    print("   Nếu ||R_0||_2<1 hoặc ||S_0||_2<1 thì phép lặp hội tụ tới A^(-1).")
    print("   Vì R_(k+1)=R_k² và S_(k+1)=S_k² nên phần dư giảm theo bình phương;")
    print("   do đó Newton–Schulz có tốc độ hội tụ bậc hai.")
    print("\n3. Công thức sai số sử dụng trong bài")
    print("   Với T_k là phần dư được chọn, q=||T_0||_2 và q_k=||T_k||_2:")
    if stop_mode == "apriori":
        print("   Chặn tiên nghiệm:")
        print("   E_k=||X_0||_2·q^(2^k)/(1-q).")
        print("   Ta giải bất đẳng thức E_k≤epsilon để tìm trước số bước k cần lặp.")
    elif stop_mode == "posteriori_relative":
        print("   Chặn hậu nghiệm tuyệt đối: E_k=||X_k||_2·q_k/(1-q_k).")
        print("   Chặn sai số tương đối:")
        print("   r_k=E_k/(||X_k||_2-E_k), với ||X_k||_2>E_k.")
        print("   Dừng khi r_k≤epsilon_tương_đối.")
    elif stop_mode == "fixed":
        print("   Bài yêu cầu thực hiện đúng số bước nên không dùng công thức sai số để dừng.")
    else:
        print("   Chặn hậu nghiệm:")
        print("   E_k=||X_k||_2·q_k/(1-q_k).")
        print("   Dừng khi E_k cộng sai số làm tròn không vượt quá epsilon.")


def residual_matrix(A, X, side):
    n = len(A)
    identity = identity_matrix(n)
    if side == "right":
        return subtract_matrices(identity, multiply_matrices(A, X))
    return subtract_matrices(identity, multiply_matrices(X, A))


def choose_certificate(A, X0):
    right = residual_matrix(A, X0, "right")
    left = residual_matrix(A, X0, "left")
    right_norm = matrix_norm_two_bounds(right)
    left_norm = matrix_norm_two_bounds(left)

    candidates = []
    if right_norm["upper"] < 1.0:
        candidates.append((right_norm["upper"], "right", right, right_norm))
    if left_norm["upper"] < 1.0:
        candidates.append((left_norm["upper"], "left", left, left_norm))

    selected = min(candidates, key=lambda item: item[0]) if candidates else None
    return {
        "right_residual": right,
        "left_residual": left,
        "right_norm": right_norm,
        "left_norm": left_norm,
        "selected": selected,
    }


def required_display_decimals(n, epsilon, requested):
    """Bảo đảm cận làm tròn ||X-round(X)||2 <= n*0.5*10^-d < eps."""
    ratio = n / epsilon
    if not math.isfinite(ratio):
        return MAX_MEANINGFUL_DECIMALS + 1
    needed = max(0, math.ceil(math.log10(ratio)) + 1)
    return max(requested, needed)


def predicted_iterations(q, norm_x0_upper, target):
    if target <= 0:
        return None
    factor = norm_x0_upper / (1.0 - q)
    if factor * q <= target:
        return 0
    if q == 0.0:
        return 0
    ratio = target / factor
    if ratio <= 0.0:
        return None
    needed_power = math.log(ratio) / math.log(q)
    if needed_power <= 1.0:
        return 0
    return max(0, math.ceil(math.log2(needed_power)))


# ============================================================
# PHƯƠNG PHÁP LẶP NEWTON–SCHULZ
# ============================================================


def newton_inverse(
    A,
    X0,
    epsilon,
    maximum_iterations,
    requested_decimals,
    fixed_iterations=None,
    stop_mode="posteriori_absolute",
):
    n = len(A)
    identity = identity_matrix(n)
    print_method(stop_mode)

    decimals = (
        requested_decimals
        if fixed_iterations is not None
        else required_display_decimals(n, epsilon, requested_decimals)
    )
    if decimals > MAX_MEANINGFUL_DECIMALS:
        if fixed_iterations is not None:
            print(
                f"\nKẾT LUẬN: Số thực kép chỉ trình bày đáng tin cậy khoảng "
                f"{MAX_MEANINGFUL_DECIMALS} chữ số sau dấu phẩy."
            )
        else:
            print("\nKẾT LUẬN: Sai số yêu cầu quá nhỏ so với độ chính xác số thực kép.")
            print(
                f"Để số đã in không làm mất sai số cần ít nhất {decimals} chữ số, "
                f"nhưng chương trình chỉ chứng nhận an toàn đến khoảng {MAX_MEANINGFUL_DECIMALS} chữ số."
            )
        return None

    if decimals > requested_decimals:
        print(
            f"\nTự tăng số chữ số hiển thị từ {requested_decimals} lên {decimals} "
            "để số làm tròn cuối vẫn đáp ứng eps."
        )

    rounding_bound = 0.5 * n * 10.0 ** (-decimals)
    print("\n4. Kiểm tra xấp xỉ ban đầu X_0")
    print_decimal_matrix(X0, decimals, prefix="X_0 = ")

    certificate = choose_certificate(A, X0)
    print("\nTính hai ma trận phần dư ban đầu:")
    print_decimal_matrix(
        certificate["right_residual"], decimals, prefix="R_0 = I-A X_0 = "
    )
    print_decimal_matrix(
        certificate["left_residual"], decimals, prefix="S_0 = I-X_0 A = "
    )

    q_right = certificate["right_norm"]
    q_left = certificate["left_norm"]
    print("\nTính chuẩn 2 của hai phần dư:")
    print(
        f"||R_0||_2 ≈ {scientific(q_right['estimate'])}, "
        f"cận trên = {scientific(q_right['upper'])}"
    )
    print(
        f"||S_0||_2 ≈ {scientific(q_left['estimate'])}, "
        f"cận trên = {scientific(q_left['upper'])}"
    )

    if certificate["selected"] is None:
        print("\nKẾT LUẬN: Không chứng minh được điều kiện co bằng chuẩn 2:")
        print("             ||I-A X_0||_2 < 1 hoặc ||I-X_0 A||_2 < 1.")
        print("Chương trình dừng, không cố lặp rồi nhận một kết quả chưa được bảo đảm.")
        return None

    q, side, selected_residual, selected_norm = certificate["selected"]
    residual_name = "R_k = I-A X_k" if side == "right" else "S_k = I-X_k A"
    print(f"\nVì q={scientific(q)}<1, ta chọn {residual_name} để theo dõi sai số.")
    print("Vậy điều kiện hội tụ được thỏa mãn và X_k hội tụ bậc hai tới A^(-1).")

    x0_norm_data = matrix_norm_two_bounds(X0)
    norm_x0_upper = x0_norm_data["upper"]
    denominator = 1.0 - q
    prior_power = q
    prior_bound = norm_x0_upper * prior_power / denominator

    predicted = None if stop_mode == "fixed" else predicted_iterations(
        q, norm_x0_upper, max(0.0, epsilon - rounding_bound)
    )
    print(
        "\n5. Tính trước số bước lặp"
        if stop_mode == "apriori"
        else "\n5. Các đại lượng dùng để kiểm tra"
    )
    print(f"Cận trên ||X_0||_2 = {scientific(norm_x0_upper)}")
    print(f"Cận sai số do làm tròn ma trận cuối = {scientific(rounding_bound)}")
    if stop_mode == "apriori":
        print("Dùng bất đẳng thức:")
        print("  ||X_0||_2·q^(2^k)/(1-q) + E_làm_tròn ≤ epsilon.")
        if predicted is None:
            print("Không tính được số bước tiên nghiệm từ dữ liệu hiện tại.")
            return None
        print(f"Thay số và giải bất đẳng thức thu được k≥{predicted}.")
        print(f"Vì vậy ta sẽ thực hiện đúng {predicted} bước Newton–Schulz.")
        if predicted > maximum_iterations:
            print(f"KẾT LUẬN: Cần {predicted} bước nhưng giới hạn chỉ là {maximum_iterations}.")
            return None

    current = copy_matrix(X0)
    final_iteration = 0
    final_internal_bound = prior_bound
    final_total_bound = prior_bound + rounding_bound
    x0_norm_lower = x0_norm_data["lower"]
    initial_relative = (
        final_total_bound / (x0_norm_lower - final_total_bound)
        if x0_norm_lower > final_total_bound
        else math.inf
    )
    if stop_mode == "posteriori_relative":
        converged = initial_relative <= epsilon
        final_selected_error = initial_relative
    elif stop_mode in {"posteriori_absolute", "apriori"}:
        converged = final_total_bound <= epsilon
        final_selected_error = final_total_bound
    else:
        converged = False
        final_selected_error = final_total_bound
    previous_selected_norm = q
    stagnation_count = 0
    iteration_records = []

    print("\n" + "=" * 92)
    print("6. THỰC HIỆN CÁC BƯỚC LẶP")
    print("=" * 92)

    if converged:
        if stop_mode == "posteriori_relative":
            print("\nNgay tại k=0, chặn sai số tương đối đã không vượt quá epsilon.")
        else:
            print("\nNgay tại k=0, chặn sai số kể cả làm tròn đã không vượt quá epsilon.")

    loop_limit = predicted if stop_mode == "apriori" else maximum_iterations
    for iteration in range(1, loop_limit + 1):
        if converged:
            break

        try:
            AX = multiply_matrices(A, current)
            correction = subtract_matrices(scalar_multiply(identity, 2.0), AX)
            next_matrix = multiply_matrices(current, correction)
        except FloatingPointError as exc:
            print(f"\nKẾT LUẬN: {exc}")
            return None

        if not all_finite(next_matrix):
            print("\nKẾT LUẬN: Phép lặp sinh NaN hoặc vô cực; không xác nhận kết quả.")
            return None

        difference = subtract_matrices(next_matrix, current)
        difference_norm = matrix_norm_two_bounds(difference)
        selected_current = residual_matrix(A, next_matrix, side)
        selected_current_norm = matrix_norm_two_bounds(selected_current)
        q_current = selected_current_norm["upper"]

        # q^(2^k) được cập nhật bằng bình phương, tránh tạo số mũ khổng lồ.
        prior_power *= prior_power
        prior_bound = norm_x0_upper * prior_power / denominator

        xk_norm_data = matrix_norm_two_bounds(next_matrix)
        xk_norm_upper = xk_norm_data["upper"]
        posterior_bound = math.inf
        if q_current < 1.0:
            posterior_bound = xk_norm_upper * q_current / (1.0 - q_current)

        if stop_mode == "apriori":
            internal_bound = prior_bound
            total_bound = prior_bound + rounding_bound
            selected_error = total_bound
            error_header = "E tiên nghiệm"
        else:
            internal_bound = posterior_bound
            total_bound = posterior_bound + rounding_bound
            if stop_mode == "posteriori_relative":
                selected_error = (
                    total_bound / (xk_norm_data["lower"] - total_bound)
                    if xk_norm_data["lower"] > total_bound
                    else math.inf
                )
                error_header = "r hậu nghiệm"
            elif stop_mode == "fixed":
                selected_error = total_bound
                error_header = "E tham khảo"
            else:
                selected_error = total_bound
                error_header = "E hậu nghiệm"
        reached_target = stop_mode != "fixed" and selected_error <= epsilon
        iteration_records.append(
            {
                "k": iteration,
                "q": q_current,
                "selected_error": selected_error,
                "status": "đạt yêu cầu" if reached_target else "tiếp tục",
            }
        )

        print("\n" + "-" * 92)
        print(f"BƯỚC {iteration}: TÍNH X_{iteration}")
        print(f"Theo công thức X_{iteration}=X_{iteration - 1}·(2I-A·X_{iteration - 1}).")
        print_three_decimal_matrices(
            AX,
            correction,
            next_matrix,
            decimals,
            (f"A·X_{iteration - 1} = ", f"M_{iteration - 1} = ", f"X_{iteration} = "),
        )
        residual_formula = f"I-A·X_{iteration}" if side == "right" else f"I-X_{iteration}·A"
        print(f"\nPhần dư T_{iteration}={residual_formula}:")
        print_decimal_matrix(selected_current, decimals, prefix=f"T_{iteration} = ")
        print(
            f"||X_{iteration}-X_{iteration - 1}||_2≤{scientific(difference_norm['upper'])}; "
            f"q_{iteration}≤{scientific(q_current)}; "
            f"{error_header}={scientific(selected_error)}."
        )

        if q_current >= previous_selected_norm * (1.0 - 32.0 * MACHINE_EPS):
            stagnation_count += 1
        else:
            stagnation_count = 0
        previous_selected_norm = q_current

        current = next_matrix
        final_iteration = iteration
        final_internal_bound = internal_bound
        final_total_bound = total_bound
        final_selected_error = selected_error

        if stop_mode != "fixed" and selected_error <= epsilon:
            converged = True
            print(f"\nCận cuối <= eps = {scientific(epsilon)}. Dừng lặp.")
            break

        if (
            stop_mode != "fixed"
            and stagnation_count >= 4
            and q_current > 100.0 * MACHINE_EPS
        ):
            print(
                "\nCẢNH BÁO: phần dư không còn giảm rõ do giới hạn số học máy. "
                "Dừng để tránh tuyên bố độ chính xác giả."
            )
            break

    if stop_mode == "fixed" and iteration_records:
        iteration_records[-1]["status"] = "đủ số bước"
    summary_header = (
        "E tiên nghiệm"
        if stop_mode == "apriori"
        else (
            "r hậu nghiệm"
            if stop_mode == "posteriori_relative"
            else ("E tham khảo" if stop_mode == "fixed" else "E hậu nghiệm")
        )
    )
    print_iteration_summary(iteration_records, decimals, summary_header)

    print("\n" + "=" * 92)
    print("7. KẾT LUẬN")
    print("=" * 92)

    if fixed_iterations is not None:
        print(f"\nĐã thực hiện đúng {fixed_iterations} bước lặp theo yêu cầu.")
        print("Kết quả dưới đây là giá trị xấp xỉ sau số bước đã chọn.")
    elif not converged:
        print(f"\nChưa đạt eps sau {final_iteration} bước đã thực hiện.")
        print(
            "Ma trận hiện tại chỉ được in để tham khảo, không xác nhận đạt sai số yêu cầu."
        )

    rounded = round_matrix(current, decimals)
    print(f"\nSố lần lặp: {final_iteration}")
    if fixed_iterations is None:
        print(f"Sai số yêu cầu eps = {scientific(epsilon)}")
    if stop_mode == "posteriori_relative":
        print(f"Chặn sai số tương đối r = {scientific(final_selected_error)}")
    else:
        print(f"Chặn sai số trước làm tròn = {scientific(final_internal_bound)}")
        print(f"Chặn sai số kể cả làm tròn = {scientific(final_total_bound)}")
    print("\nVậy ma trận nghịch đảo gần đúng là:")
    print_decimal_matrix(rounded, decimals, prefix="A^(-1) ≈ ")

    left_product = multiply_matrices(A, rounded)
    right_product = multiply_matrices(rounded, A)
    left_residual = subtract_matrices(identity, left_product)
    right_residual = subtract_matrices(identity, right_product)
    left_norm = matrix_norm_two_bounds(left_residual)
    right_norm = matrix_norm_two_bounds(right_residual)

    print("\n" + "=" * 92)
    print("8. KIỂM TRA LẠI KẾT QUẢ")
    print("=" * 92)
    print("\nA X và I:")
    print_two_decimal_matrices(
        left_product, identity, decimals, left_name="A X = ", right_name="I = "
    )
    print("\nX A và I:")
    print_two_decimal_matrices(
        right_product, identity, decimals, left_name="X A = ", right_name="I = "
    )
    print("\nChuẩn phần dư của ma trận đã in:")
    print(f"||I-A X||_2 <= {scientific(left_norm['upper'])}")
    print(f"||I-X A||_2 <= {scientific(right_norm['upper'])}")

    if fixed_iterations is not None:
        print("\nKẾT LUẬN: Đã thu được ma trận nghịch đảo gần đúng sau đúng số bước yêu cầu;")
        print("không dùng số bước cố định để khẳng định một ngưỡng sai số cho trước.")
    elif converged:
        if stop_mode == "posteriori_relative":
            print(
                "\nKẾT LUẬN: Ma trận đã in xấp xỉ A^(-1) với chặn sai số tương đối "
                "không vượt quá epsilon."
            )
        else:
            print(
                "\nKẾT LUẬN: Theo chặn sai số Newton–Schulz, ma trận đã in "
                "xấp xỉ A^(-1) với sai số chuẩn 2 không vượt quá epsilon."
            )
    else:
        print("\nKẾT LUẬN: Chưa đủ cơ sở xác nhận độ chính xác đã yêu cầu.")

    return {
        "matrix": rounded,
        "raw_matrix": current,
        "converged": converged,
        "iterations": final_iteration,
        "error_bound": final_selected_error,
        "certificate_side": side,
    }


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================


def main():
    print("=" * 108)
    print("PHƯƠNG PHÁP LẶP NEWTON–SCHULZ TÌM GẦN ĐÚNG MA TRẬN NGHỊCH ĐẢO")
    print("=" * 108)

    m = input_positive_integer("Nhập số dòng của ma trận A (m): ")
    n = input_positive_integer("Nhập số cột của ma trận A (n): ")
    A_exact = input_matrix_exact("A", m, n)

    if m != n:
        print("\nKẾT LUẬN: A không vuông nên không có nghịch đảo thông thường.")
        return

    print("\nMa trận đầu vào chính xác:")
    print_exact_matrix(A_exact, prefix="A = ")

    rank_A = exact_rank(A_exact)
    print(f"\nHạng chính xác của A: {rank_A}")
    if rank_A < n:
        print(f"KẾT LUẬN: rank(A)={rank_A}<{n}, nên A suy biến và không có nghịch đảo.")
        return

    try:
        A = exact_to_float(A_exact)
    except OverflowError as exc:
        print(f"\nKẾT LUẬN: {exc}")
        return

    print("\nChọn điều kiện dừng:")
    print("1. Sai số tuyệt đối hậu nghiệm")
    print("2. Sai số tương đối hậu nghiệm")
    print("3. Sai số tiên nghiệm (tính trước số bước k)")
    print("4. Đạt d chữ số thập phân chính xác")
    print("5. Thực hiện đúng số lần lặp đã cho")
    print("0. Thoát")
    while True:
        stop_choice = input("Chọn [Enter = 1]: ").strip() or "1"
        if stop_choice == "0":
            return
        if stop_choice in {"1", "2", "3", "4", "5"}:
            break
        print("Lỗi: Vui lòng chỉ chọn từ 1 đến 5.")

    fixed_iterations = None
    if stop_choice in {"1", "2", "3"}:
        if stop_choice == "1":
            stop_mode = "posteriori_absolute"
            prompt = "Nhập sai số tuyệt đối hậu nghiệm epsilon: "
        elif stop_choice == "2":
            stop_mode = "posteriori_relative"
            prompt = "Nhập sai số tương đối hậu nghiệm epsilon: "
        else:
            stop_mode = "apriori"
            prompt = "Nhập sai số tuyệt đối tiên nghiệm epsilon: "
        epsilon = input_positive_number(prompt.replace(": ", " [Enter = 1e-7]: "), 1e-7)
        maximum_iterations = input_positive_integer(
            "Nhập số lần lặp tối đa [Enter = 100]: ", default=100
        )
        requested_decimals = input_nonnegative_integer(
            "Nhập số chữ số sau dấu phẩy muốn hiển thị [Enter = 7]: ",
            default=7,
        )
    elif stop_choice == "4":
        stop_mode = "posteriori_absolute"
        accurate_decimals = input_nonnegative_integer(
            "Nhập số chữ số thập phân chính xác d: "
        )
        epsilon = 0.5 * 10.0 ** (-accurate_decimals)
        print(
            f"Để bảo đảm {accurate_decimals} chữ số thập phân, chọn "
            f"epsilon=0.5·10^(-{accurate_decimals})={scientific(epsilon)}."
        )
        maximum_iterations = input_positive_integer(
            "Nhập số lần lặp tối đa [Enter = 100]: ", default=100
        )
        requested_decimals = accurate_decimals
    else:
        stop_mode = "fixed"
        fixed_iterations = input_nonnegative_integer("Nhập số lần lặp cần thực hiện: ")
        maximum_iterations = fixed_iterations
        requested_decimals = input_nonnegative_integer(
            "Nhập số chữ số sau dấu phẩy muốn hiển thị [Enter = 7]: ",
            default=7,
        )
        # Không dùng epsilon để dừng trong chế độ này; giá trị chỉ phục vụ các phép đánh giá nội bộ.
        epsilon = 1.0

    choice = choose_initial_approximation()
    if choice == "0":
        return
    if choice == "1":
        X0, norm_one, norm_infinity, alpha = automatic_initial_approximation(A)
        if X0 is None:
            print("\nKẾT LUẬN: Không thể tạo xấp xỉ đầu tự động trong số học máy.")
            return
        print("\nKhởi tạo tự động:")
        print("X_0 = alpha A^T,  alpha=1/(||A||_1 ||A||_∞)")
        print(f"||A||_1 = {scientific(norm_one)}")
        print(f"||A||_∞ = {scientific(norm_infinity)}")
        print(f"alpha = {scientific(alpha)}")
        print(
            "Vì A khả nghịch nên 0 < alpha*sigma_i(A)^2 <= 1; "
            "do đó ||I-A X_0||_2<1 trong số học chính xác."
        )
    else:
        X0_exact = input_matrix_exact("X_0", n, n)
        try:
            X0 = exact_to_float(X0_exact)
        except OverflowError as exc:
            print(f"\nKẾT LUẬN: {exc}")
            return
        print("\nXấp xỉ đầu do người dùng nhập:")
        print_exact_matrix(X0_exact, prefix="X_0 = ")

    newton_inverse(
        A,
        X0,
        epsilon,
        maximum_iterations,
        requested_decimals,
        fixed_iterations=fixed_iterations,
        stop_mode=stop_mode,
    )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
