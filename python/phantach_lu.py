import math
from exam_format import exam_print as print
import sys
from fractions import Fraction
import sympy as sp
from input_utils import MathInputError, parse_exact, split_number_row


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# NHẬP VÀ HIỂN THỊ
# ============================================================


def input_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt).strip())
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


def input_matrix_row(prompt, expected_count):
    """
    Nhập đúng expected_count phần tử.
    Cho phép số nguyên, số thập phân và phân số:
    2, -3, 0.25, 1/3, -5/7.
    """
    while True:
        try:
            tokens = split_number_row(input(prompt), expected_count)
            return [parse_exact(token) for token in tokens]
        except (MathInputError, ValueError, ZeroDivisionError):
            print(
                "Lỗi: Chỉ nhập số nguyên, số thập phân hoặc phân số hợp lệ "
                "(ví dụ 2, -3, 0.25, 1/3)."
            )


def input_matrix(name, rows, columns):
    print(f"\nNhập ma trận {name} ({rows}x{columns}):")
    return [
        input_matrix_row(f"Nhập dòng {i + 1} (cách nhau bởi khoảng trắng): ", columns)
        for i in range(rows)
    ]


def exact_number(value):
    """Hiển thị số chính xác dưới dạng số nguyên hoặc phân số."""
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    return sp.sstr(sp.simplify(value))


def decimal_number(value, decimals):
    """Chỉ chuyển Fraction sang float khi hiển thị ma trận."""
    if value == 0:
        value = Fraction(0)
    return f"{float(value):10.{decimals}f}"


def matrix_to_lines(matrix, decimals):
    return [
        "[" + "  ".join(decimal_number(value, decimals) for value in row) + "]"
        for row in matrix
    ]


def print_matrix(matrix, decimals, prefix=""):
    if not matrix:
        print(prefix + "[]")
        return

    lines = matrix_to_lines(matrix, decimals)
    middle = len(lines) // 2
    padding = " " * len(prefix)

    for i, line in enumerate(lines):
        print((prefix if i == middle else padding) + line)


def print_two_matrices(A, B, decimals, name_a="A = ", name_b="B = "):
    lines_a = matrix_to_lines(A, decimals)
    lines_b = matrix_to_lines(B, decimals)

    row_count = max(len(lines_a), len(lines_b))
    width_a = max(len(line) for line in lines_a)
    middle = row_count // 2

    for i in range(row_count):
        left = lines_a[i] if i < len(lines_a) else ""
        right = lines_b[i] if i < len(lines_b) else ""

        prefix_a = name_a if i == middle else " " * len(name_a)
        prefix_b = name_b if i == middle else " " * len(name_b)

        print((prefix_a + left).ljust(len(name_a) + width_a + 6) + prefix_b + right)


def print_lu_pair(L, U, decimals, step=None):
    """In L và U cạnh nhau thành một khối trình bày gọn."""
    if step is None:
        title = "Các ma trận L và U:"
        name_l = "L = "
        name_u = "U = "
    else:
        title = f"Sau bước {step}:"
        name_l = f"L^({step}) = "
        name_u = f"U^({step}) = "

    print(f"\n{title}")
    print_two_matrices(L, U, decimals, name_a=name_l, name_b=name_u)


# ============================================================
# PHÉP TOÁN MA TRẬN
# ============================================================


def copy_matrix(matrix):
    return [row[:] for row in matrix]


def zero_matrix(rows, columns):
    return [[Fraction(0) for _ in range(columns)] for _ in range(rows)]


def identity_matrix(n):
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def transpose_matrix(matrix):
    if not matrix:
        return []
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]


def multiply_matrices(A, B):
    if not A or not B or len(A[0]) != len(B):
        raise ValueError("Kích thước hai ma trận không phù hợp để nhân.")

    rows = len(A)
    middle = len(B)
    columns = len(B[0])

    return [
        [
            sum((A[i][k] * B[k][j] for k in range(middle)), Fraction(0))
            for j in range(columns)
        ]
        for i in range(rows)
    ]


def matrices_equal(A, B):
    return (
        len(A) == len(B)
        and (not A or len(A[0]) == len(B[0]))
        and all(A[i][j] == B[i][j] for i in range(len(A)) for j in range(len(A[0])))
    )


def augment(A, B):
    return [A[i][:] + B[i][:] for i in range(len(A))]


def manual_rank(matrix):
    """Tính hạng bằng Gauss trên một bản sao, không dùng hàm có sẵn."""
    if not matrix:
        return 0

    work = copy_matrix(matrix)
    rows = len(work)
    columns = len(work[0])
    pivot_row = 0

    for column in range(columns):
        candidate = None

        for row in range(pivot_row, rows):
            if work[row][column] != 0:
                candidate = row
                break

        if candidate is None:
            continue

        if candidate != pivot_row:
            work[pivot_row], work[candidate] = (work[candidate], work[pivot_row])

        for row in range(pivot_row + 1, rows):
            if work[row][column] != 0:
                factor = work[row][column] / work[pivot_row][column]
                for j in range(column, columns):
                    work[row][j] -= factor * work[pivot_row][j]

        pivot_row += 1

        if pivot_row == rows:
            break

    return pivot_row


def format_sum_terms(terms):
    """
    Chuyển danh sách (l_ik, u_kj) thành chuỗi thay số:
    (2)(3) + (1/2)(4)
    """
    if not terms:
        return "0"

    return " + ".join(
        f"({exact_number(left)})({exact_number(right)})" for left, right in terms
    )


def normal_equation_system(A, B):
    """Lap he theo dung luong PDF: T=A^T, A_bar=T*A, B_bar=T*B."""
    T = transpose_matrix(A)
    return multiply_matrices(T, A), multiply_matrices(T, B)


# ============================================================
# PHÂN TÁCH LU DOOLITTLE: A = L*U, l_ii = 1
# ============================================================


def lu_doolittle(A, decimals, show_steps=True):
    """
    Phân tách LU theo phương pháp Doolittle:

        A = L*U

    L là ma trận tam giác dưới, l_ii = 1.
    U là ma trận tam giác trên.

    Không đổi hàng và không tạo ma trận P.
    """
    n = len(A)
    L = zero_matrix(n, n)
    U = zero_matrix(n, n)

    for i in range(n):
        L[i][i] = Fraction(1)

    if show_steps:
        print("\n" + "=" * 92)
        print("PHÂN TÁCH LU THEO PHƯƠNG PHÁP DOOLITTLE")
        print("=" * 92)
        print("\nTa phân tách ma trận A dưới dạng:")
        print("                              A = L*U")
        print("\nTrong đó:")
        print("  - L là ma trận tam giác dưới và l_ii = 1.")
        print("  - U là ma trận tam giác trên.")

        print("\nCông thức tính các phần tử:")
        print("  u_ij = a_ij - Σ(k=1..i-1) l_ik*u_kj,   với i <= j")
        print("  l_ij = [a_ij - Σ(k=1..j-1) l_ik*u_kj] / u_jj,   với i > j")
        print("  l_ii = 1")

        print("\nThay lần lượt các giá trị theo công thức trên vào L và U:")
        print_lu_pair(L, U, decimals, step=0)
        print("\nCông thức tổng quát dùng để tính từng phần tử (chỉ số từ 1):")
        print("  u_ij = a_ij - Σ(k=1..i-1) l_ik·u_kj,             với i ≤ j")
        print("  l_ji = [a_ji - Σ(k=1..i-1) l_jk·u_ki] / u_ii,   với j > i")
        print("  l_ii = 1")

    for i in range(n):
        # Tính hàng i của U:
        # u_ij = a_ij - sum(k=0..i-1) l_ik*u_kj, j >= i
        for j in range(i, n):
            terms = [(L[i][k], U[k][j]) for k in range(i)]
            total = sum((left * right for left, right in terms), Fraction(0))
            U[i][j] = A[i][j] - total
            if show_steps:
                print(f"\nTính phần tử u_{i + 1},{j + 1}:")
                print(
                    f"  u_{i + 1},{j + 1} = a_{i + 1},{j + 1} "
                    f"- Σ(k=1..{i}) l_{i + 1},k·u_k,{j + 1}"
                )
                print(
                    f"          = {exact_number(A[i][j])} "
                    f"- ({format_sum_terms(terms)})"
                )
                print(f"          = {exact_number(U[i][j])}")

        # LU Doolittle không đổi hàng chỉ tiếp tục được khi u_ii != 0.
        if U[i][i] == 0:
            if show_steps:
                print("\n" + "-" * 92)
                print(f"Dừng tại bước {i + 1} vì u_{i + 1},{i + 1} = 0.")
                print(
                    "Ma trận không thỏa điều kiện phân tách LU Doolittle "
                    "không đổi hàng."
                )
                print_lu_pair(L, U, decimals, step=i + 1)
            return None, None, i

        # Tính cột i của L:
        # l_ji = [a_ji - sum(k=0..i-1) l_jk*u_ki] / u_ii, j > i
        for j in range(i + 1, n):
            terms = [(L[j][k], U[k][i]) for k in range(i)]
            total = sum((left * right for left, right in terms), Fraction(0))
            L[j][i] = (A[j][i] - total) / U[i][i]
            if show_steps:
                print(f"\nTính phần tử l_{j + 1},{i + 1}:")
                print(
                    f"  l_{j + 1},{i + 1} = [a_{j + 1},{i + 1} "
                    f"- Σ(k=1..{i}) l_{j + 1},k·u_k,{i + 1}] / u_{i + 1},{i + 1}"
                )
                print(
                    f"          = [{exact_number(A[j][i])} "
                    f"- ({format_sum_terms(terms)})] / {exact_number(U[i][i])}"
                )
                print(f"          = {exact_number(L[j][i])}")

        if show_steps:
            print("\n" + "-" * 92)
            print(f"Bước {i + 1}: đã tính hàng {i + 1} của U và cột {i + 1} của L.")
            print_lu_pair(L, U, decimals, step=i + 1)

    if show_steps:
        print("\nHoàn thành việc thay các phần tử vào hai ma trận L và U.")

    return L, U, None


# ============================================================
# PHAN TACH LU CROUT THEO PDF: A = L*U, u_ii = 1
# ============================================================


def lu_crout(A, decimals=7, show_steps=True):
    """
    Phan tach LU theo kieu Crout dung trong PDF:

        A = L*U,  U co duong cheo bang 1.

    Cong thuc voi chi so 0-based trong code:
        l_it = a_it - sum(k=0..t-1) l_ik*u_kt,  i=t..n-1
        u_tj = (a_tj - sum(k=0..t-1) l_tk*u_kj) / l_tt,  j=t+1..n-1
    """
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A phai la ma tran vuong khac rong.")

    L = zero_matrix(n, n)
    U = identity_matrix(n)

    if show_steps:
        print("\n" + "=" * 92)
        print("PHAN TACH LU THEO PHUONG PHAP CROUT (THEO PDF)")
        print("=" * 92)
        print("Khoi tao L = O, U = I.")
        print("Voi t = 1..n:")
        print("  l_it = a_it - sum(k=1..t-1) l_ik u_kt, i=t..n")
        print("  neu l_tt = 0 thi khong co phan tach LU Crout khong doi hang")
        print("  u_tj = (a_tj - sum(k=1..t-1) l_tk u_kj)/l_tt, j=t+1..n")
        print_lu_pair(L, U, decimals, step=0)

    for t in range(n):
        for i in range(t, n):
            terms = [(L[i][k], U[k][t]) for k in range(t)]
            total = sum((left * right for left, right in terms), Fraction(0))
            L[i][t] = A[i][t] - total
            if show_steps:
                print(f"\nTinh l_{i + 1},{t + 1}:")
                print(
                    f"  l_{i + 1},{t + 1} = a_{i + 1},{t + 1} "
                    f"- sum(k=1..{t}) l_{i + 1},k*u_k,{t + 1}"
                )
                print(f"          = {exact_number(A[i][t])} - ({format_sum_terms(terms)})")
                print(f"          = {exact_number(L[i][t])}")

        if L[t][t] == 0:
            if show_steps:
                print("\n" + "-" * 92)
                print(f"Dung tai buoc {t + 1} vi l_{t + 1},{t + 1} = 0.")
                print("Khong co phan tach LU Crout khong doi hang.")
                print_lu_pair(L, U, decimals, step=t + 1)
            return None, None, t

        for j in range(t + 1, n):
            terms = [(L[t][k], U[k][j]) for k in range(t)]
            total = sum((left * right for left, right in terms), Fraction(0))
            U[t][j] = (A[t][j] - total) / L[t][t]
            if show_steps:
                print(f"\nTinh u_{t + 1},{j + 1}:")
                print(
                    f"  u_{t + 1},{j + 1} = [a_{t + 1},{j + 1} "
                    f"- sum(k=1..{t}) l_{t + 1},k*u_k,{j + 1}] / l_{t + 1},{t + 1}"
                )
                print(
                    f"          = [{exact_number(A[t][j])} - ({format_sum_terms(terms)})] "
                    f"/ {exact_number(L[t][t])}"
                )
                print(f"          = {exact_number(U[t][j])}")

        if show_steps:
            print("\n" + "-" * 92)
            print(f"Buoc {t + 1}: da tinh cot {t + 1} cua L va hang {t + 1} cua U.")
            print_lu_pair(L, U, decimals, step=t + 1)

    return L, U, None


def plu_decomposition(A, pivot_tolerance=None, show_steps=False):
    """Phân tích PA=LU bằng pivot từng phần, không dùng hàm giải có sẵn."""
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A phải là ma trận vuông khác rỗng.")
    try:
        scale = max(abs(float(value)) for row in A for value in row)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("A chỉ được chứa số hữu hạn.") from exc
    if not all(math.isfinite(float(value)) for row in A for value in row):
        raise ValueError("A chỉ được chứa số hữu hạn.")
    if pivot_tolerance is None:
        pivot_tolerance = n * sys.float_info.epsilon * max(1.0, scale)
    if not math.isfinite(pivot_tolerance) or pivot_tolerance <= 0:
        raise ValueError("pivot_tolerance phải dương và hữu hạn.")

    U = copy_matrix(A)
    L = zero_matrix(n, n)
    P = identity_matrix(n)
    for i in range(n):
        L[i][i] = Fraction(1)
    swaps = []
    near_singular = False

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(float(U[i][k])))
        pivot_abs = abs(float(U[pivot_row][k]))
        if pivot_abs == 0.0:
            raise ArithmeticError(f"Ma trận suy biến: cột {k + 1} không còn pivot khác 0.")
        if pivot_abs <= pivot_tolerance:
            near_singular = True
        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            P[k], P[pivot_row] = P[pivot_row], P[k]
            for j in range(k):
                L[k][j], L[pivot_row][j] = L[pivot_row][j], L[k][j]
            swaps.append((k, pivot_row))
            if show_steps:
                print(f"Đổi hàng R{k + 1} <-> R{pivot_row + 1}.")
        pivot = U[k][k]
        for i in range(k + 1, n):
            factor = U[i][k] / pivot
            L[i][k] = factor
            U[i][k] = Fraction(0)
            for j in range(k + 1, n):
                U[i][j] -= factor * U[k][j]
    return P, L, U, swaps, near_singular


def apply_permutation(P, B):
    if not B or len(P) != len(B):
        raise ValueError("Kích thước P và B không phù hợp.")
    return multiply_matrices(P, B)


def plu_solve(A, B, pivot_tolerance=None, show_steps=False):
    """Giải AX=B qua PA=LU, thế tiến và thế lùi thủ công."""
    if not B or any(len(row) != len(B[0]) for row in B) or len(B) != len(A):
        raise ValueError("B phải có cùng số hàng với A.")
    P, L, U, swaps, near_singular = plu_decomposition(
        A, pivot_tolerance=pivot_tolerance, show_steps=show_steps
    )
    PB = apply_permutation(P, B)
    Y = forward_substitution(L, PB, show_steps=show_steps)
    X = back_substitution(U, Y, show_steps=show_steps)
    return X, P, L, U, swaps, near_singular


def solve_crout_pdf(A, B, decimals=7, show_steps=False, use_normal_equations=True):
    """Giai he theo ban PDF: lap normal equation roi phan tach LU Crout."""
    if use_normal_equations:
        A_work, B_work = normal_equation_system(A, B)
    else:
        A_work, B_work = copy_matrix(A), copy_matrix(B)
    L, U, failed_index = lu_crout(A_work, decimals, show_steps=show_steps)
    if failed_index is not None:
        raise ArithmeticError("Khong co phan tach LU Crout khong doi hang.")
    Y = forward_substitution(L, B_work, show_steps=show_steps)
    X = back_substitution(U, Y, show_steps=show_steps)
    return X, L, U, A_work, B_work


# ============================================================
# THẾ XUÔI: L*Y = B
# ============================================================


def forward_substitution(L, B, show_steps=True):
    n = len(L)
    rhs_columns = len(B[0])
    Y = zero_matrix(n, rhs_columns)

    if show_steps:
        print("\n" + "=" * 92)
        print("GIẢI HỆ L*Y = B BẰNG QUÁ TRÌNH THẾ XUÔI")

    for column in range(rhs_columns):
        if show_steps:
            print(f"\nVế phải thứ {column + 1}:")

        for i in range(n):
            terms = [(L[i][k], Y[k][column]) for k in range(i)]
            total = sum((left * right for left, right in terms), Fraction(0))

            Y[i][column] = (B[i][column] - total) / L[i][i]

            if show_steps:
                print(
                    f"y_{i + 1},{column + 1} "
                    f"= [b_{i + 1},{column + 1} "
                    f"- Σ(l_{i + 1},k*y_k,{column + 1})] "
                    f"/ l_{i + 1},{i + 1}"
                )
                print(
                    f"              = [{exact_number(B[i][column])} "
                    f"- ({format_sum_terms(terms)})] "
                    f"/ {exact_number(L[i][i])}"
                )
                print(f"              = {exact_number(Y[i][column])}")

    return Y


# ============================================================
# THẾ LÙI: U*X = Y
# ============================================================


def back_substitution(U, Y, show_steps=True):
    n = len(U)
    rhs_columns = len(Y[0])
    X = zero_matrix(n, rhs_columns)

    if show_steps:
        print("\n" + "=" * 92)
        print("GIẢI HỆ U*X = Y BẰNG QUÁ TRÌNH THẾ LÙI")

    for column in range(rhs_columns):
        if show_steps:
            print(f"\nVế phải thứ {column + 1}:")

        for i in range(n - 1, -1, -1):
            if U[i][i] == 0:
                raise ZeroDivisionError("U có phần tử đường chéo bằng 0.")

            terms = [(U[i][k], X[k][column]) for k in range(i + 1, n)]
            total = sum((left * right for left, right in terms), Fraction(0))

            X[i][column] = (Y[i][column] - total) / U[i][i]

            if show_steps:
                print(
                    f"x_{i + 1},{column + 1} "
                    f"= [y_{i + 1},{column + 1} "
                    f"- Σ(u_{i + 1},k*x_k,{column + 1})] "
                    f"/ u_{i + 1},{i + 1}"
                )
                print(
                    f"              = [{exact_number(Y[i][column])} "
                    f"- ({format_sum_terms(terms)})] "
                    f"/ {exact_number(U[i][i])}"
                )
                print(f"              = {exact_number(X[i][column])}")

    return X


# ============================================================
# XỬ LÝ KẾT QUẢ
# ============================================================


def determinant_from_u(U):
    determinant = Fraction(1)
    for i in range(len(U)):
        determinant *= U[i][i]
    return determinant


def determinant_from_lu(L, U):
    determinant = Fraction(1)
    for i in range(len(L)):
        determinant *= L[i][i] * U[i][i]
    return determinant


def print_lu_result(A, L, U, decimals):
    print("\n" + "=" * 92)
    print("KẾT QUẢ PHÂN TÁCH LU\n")

    print_lu_pair(L, U, decimals)

    product = multiply_matrices(L, U)

    print("\nKiểm tra L*U:")
    print_two_matrices(product, A, decimals, name_a="L*U = ", name_b="A = ")

    if matrices_equal(product, A):
        print("\nKết luận kiểm tra: L*U = A (đúng).")
    else:
        print("\nCảnh báo: L*U khác A.")

    determinant = determinant_from_lu(L, U)
    print("\ndet(A) = det(L)*det(U) = tich duong cheo L nhan tich duong cheo U")
    print(f"det(A) = {exact_number(determinant)}")

    return determinant


def explain_failed_decomposition(A, B=None, inverse_mode=False):
    """
    Pivot Doolittle bằng 0 không nhất thiết đồng nghĩa A suy biến.
    Vì vậy cần nói rõ là phương pháp không đổi hàng không áp dụng được.
    """
    rank_A = manual_rank(A)
    n = len(A)

    print(f"\nHạng của A: {rank_A}")

    if inverse_mode:
        if rank_A < n:
            print("\nKẾT LUẬN: Ma trận A suy biến nên không có nghịch đảo.")
        else:
            print(
                "\nKẾT LUẬN: A vẫn không suy biến, nhưng không phân tách được "
                "theo LU Doolittle không đổi hàng vì xuất hiện pivot u_ii = 0."
            )
            print(
                "Trường hợp này phải đổi hàng hoặc dùng phân tách PLU, "
                "nhưng không thuộc đúng công thức LU trong tài liệu."
            )
        return

    if B is not None:
        rank_augmented = manual_rank(augment(A, B))
        print(f"Hạng của ma trận bổ sung [A|B]: {rank_augmented}")

        if rank_A < rank_augmented:
            print("\nKẾT LUẬN: Hệ phương trình VÔ NGHIỆM.")
        elif rank_A < n:
            print(f"\nKẾT LUẬN: Hệ có VÔ SỐ NGHIỆM ({n - rank_A} biến tự do).")
        else:
            print(
                "\nKẾT LUẬN: Hệ có nghiệm duy nhất, nhưng không thể tiếp tục "
                "bằng LU Doolittle không đổi hàng do pivot u_ii = 0."
            )


def decompose_and_solve(A, B, decimals, inverse_mode=False):
    A_original = copy_matrix(A)
    B_original = copy_matrix(B)
    n = len(A)

    print("\n" + "=" * 92)

    if inverse_mode:
        print("TÌM MA TRẬN NGHỊCH ĐẢO BẰNG PHÂN TÁCH LU")
        print("Ta giải hệ A*X = I.")
    else:
        print("GIẢI HỆ PHƯƠNG TRÌNH A*X = B BẰNG PHÂN TÁCH LU")

    print("\nDữ liệu ban đầu:")
    print_matrix(A_original, decimals, prefix="A = ")
    print_matrix(B_original, decimals, prefix="I = " if inverse_mode else "B = ")

    L, U, failed_index = lu_crout(A_original, decimals, show_steps=True)

    if failed_index is not None:
        print("\nLU Crout không đổi hàng không áp dụng được; chuyển sang PLU với pivot từng phần.")
        try:
            X, P, L, U, swaps, near_singular = plu_solve(
                A_original, B_original, show_steps=True
            )
        except ArithmeticError:
            explain_failed_decomposition(A_original, B_original, inverse_mode=inverse_mode)
            return None
        print("\nPhân tích PLU thu được P*A = L*U:")
        print_matrix(P, decimals, prefix="P = ")
        print_matrix(L, decimals, prefix="L = ")
        print_matrix(U, decimals, prefix="U = ")
        PA = multiply_matrices(P, A_original)
        LU = multiply_matrices(L, U)
        print_two_matrices(PA, LU, decimals, name_a="P*A = ", name_b="L*U = ")
        if near_singular:
            print("CẢNH BÁO: pivot rất nhỏ; ma trận gần suy biến và nghiệm nhạy với làm tròn.")
        print_matrix(X, decimals, prefix="A^(-1) = " if inverse_mode else "X = ")
        residual = multiply_matrices(A_original, X)
        print_two_matrices(
            residual,
            B_original,
            decimals,
            name_a="A*X = ",
            name_b="B = " if not inverse_mode else "I = ",
        )
        return X

    determinant = print_lu_result(A_original, L, U, decimals)

    rank_A = manual_rank(A_original)
    rank_augmented = manual_rank(augment(A_original, B_original))

    print(f"\nHạng của A: {rank_A}")
    print(f"Hạng của ma trận bổ sung [A|B]: {rank_augmented}")

    if determinant == 0:
        if inverse_mode:
            print("\nKẾT LUẬN: Ma trận A không khả nghịch.")
        elif rank_A < rank_augmented:
            print("\nKẾT LUẬN: Hệ phương trình VÔ NGHIỆM.")
        else:
            print(f"\nKẾT LUẬN: Hệ có VÔ SỐ NGHIỆM ({n - rank_A} biến tự do).")
        return None

    print("\nDo A = L*U nên:")
    print("A*X = B")
    print("L*U*X = B")
    print("Đặt Y = U*X, ta giải lần lượt:")
    print("L*Y = B")
    print("U*X = Y")

    Y = forward_substitution(L, B_original, show_steps=True)

    print("\nMa trận nghiệm trung gian Y:")
    print_matrix(Y, decimals, prefix="Y = ")

    X = back_substitution(U, Y, show_steps=True)

    print("\n" + "=" * 92)

    if inverse_mode:
        print("KẾT LUẬN: Ma trận A khả nghịch và A^(-1) là:\n")
        print_matrix(X, decimals, prefix="A^(-1) = ")

        product = multiply_matrices(A_original, X)

        print("\nKiểm tra A*A^(-1):")
        print_matrix(product, decimals, prefix="A*A^(-1) = ")

        if matrices_equal(product, identity_matrix(n)):
            print("\nKết luận kiểm tra: A*A^(-1) = I (đúng).")
        else:
            print("\nCảnh báo: A*A^(-1) khác I.")
    else:
        print("KẾT LUẬN: Hệ có NGHIỆM DUY NHẤT X:\n")
        print_matrix(X, decimals, prefix="X = ")

        product = multiply_matrices(A_original, X)

        print("\nKiểm tra A*X và B:")
        print_two_matrices(
            product, B_original, decimals, name_a="A*X = ", name_b="B = "
        )

        if matrices_equal(product, B_original):
            print("\nKết luận kiểm tra: A*X = B (đúng).")
        else:
            print("\nCảnh báo: A*X khác B.")

    return X


# ============================================================
# MENU CHƯƠNG TRÌNH
# ============================================================


def input_square_matrix():
    m = input_positive_integer("Nhập số dòng của ma trận A (m): ")
    n = input_positive_integer("Nhập số cột của ma trận A (n): ")
    A = input_matrix("A", m, n)

    if m != n:
        print("\nKẾT LUẬN: Phân tách LU yêu cầu A là ma trận vuông.")
        return None

    return A


def decompose_only():
    print("\n--- Phân tách A = L*U theo phương pháp Crout (theo PDF) ---")
    A = input_square_matrix()

    if A is None:
        return

    decimals = input_nonnegative_integer(
        "\nSố chữ số sau dấu phẩy [Enter = 7]: ", default=7
    )

    print("\nMa trận ban đầu:")
    print_matrix(A, decimals, prefix="A = ")

    L, U, failed_index = lu_crout(A, decimals, show_steps=True)

    if failed_index is not None:
        explain_failed_decomposition(A)
        return

    print_lu_result(A, L, U, decimals)

    rank_A = manual_rank(A)
    print(f"\nHạng của A: {rank_A}")

    if rank_A == len(A):
        print("\nKẾT LUẬN: A là ma trận không suy biến.")
    else:
        print("\nKẾT LUẬN: A là ma trận suy biến.")


def solve_system():
    print("\n--- Giải hệ A*X = B bằng phân tách LU ---")
    A = input_square_matrix()

    if A is None:
        return

    n = len(A)
    k = input_positive_integer("\nNhập số cột của ma trận B (k): ")
    B = input_matrix("B", n, k)

    decimals = input_nonnegative_integer(
        "\nSố chữ số sau dấu phẩy [Enter = 7]: ", default=7
    )

    decompose_and_solve(A, B, decimals, inverse_mode=False)


def find_inverse():
    print("\n--- Tìm A^(-1) bằng phân tách LU ---")
    A = input_square_matrix()

    if A is None:
        return

    decimals = input_nonnegative_integer(
        "\nSố chữ số sau dấu phẩy [Enter = 7]: ", default=7
    )

    decompose_and_solve(A, identity_matrix(len(A)), decimals, inverse_mode=True)


def main():
    print("=" * 92)
    print("PHÂN TÁCH LU DOOLITTLE - A = L*U")
    print("=" * 92)
    print("1. Phân tách ma trận A thành A = L*U theo Crout (PDF)")
    print("2. Giải hệ phương trình A*X = B bằng phân tách LU")
    print("3. Tìm ma trận nghịch đảo A^(-1) bằng phân tách LU")
    print("0. Thoát")
    print("Input: A (và B nếu giải hệ). Output: P,L,U, nghiệm/nghịch đảo và kiểm tra.")

    while True:
        choice = input("Chọn [Enter = 1]: ").strip() or "1"

        if choice == "0":
            return

        if choice == "1":
            decompose_only()
            return
        if choice == "2":
            solve_system()
            return
        if choice == "3":
            find_inverse()
            return

        print("Lỗi: Vui lòng chỉ chọn 1, 2 hoặc 3.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except Exception as error:
        print(f"\nKhông thể thực hiện: {error}")
