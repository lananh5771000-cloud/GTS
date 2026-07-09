import sys
from exam_format import exam_print as print
import sympy as sp
from input_utils import MathInputError, parse_exact, split_number_row


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# NHẬP VÀ HIỂN THỊ
# ============================================================


def input_positive_integer(prompt):
    """Nhập số nguyên dương và bắt nhập lại nếu không hợp lệ."""
    while True:
        try:
            value = int(input(prompt).strip())
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Lỗi: Vui lòng nhập một số nguyên dương.")


def input_nonnegative_integer(prompt, default=None):
    """Nhập số nguyên không âm và bắt nhập lại nếu không hợp lệ."""
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
        except (MathInputError, ValueError, TypeError, ZeroDivisionError):
            print(
                "Lỗi: Chỉ nhập số nguyên, số thập phân hoặc phân số hợp lệ "
                "(ví dụ 2, -3, 0.25, 1/3)."
            )


def input_matrix(name, rows, columns):
    """Nhập ma trận kích thước rows x columns."""
    print(f"\nNhập ma trận {name} ({rows}x{columns}):")
    return [
        input_matrix_row(f"Nhập dòng {i + 1} (cách nhau bởi khoảng trắng): ", columns)
        for i in range(rows)
    ]


def exact_number(value):
    """Hiển thị kết quả chính xác, kể cả căn thức."""
    return sp.sstr(sp.simplify(value))


def decimal_number(value, decimals):
    """Chuyển sang số thập phân chỉ tại thời điểm hiển thị ma trận."""
    numeric_value = float(sp.N(value, max(20, decimals + 8)))

    if (
        abs(numeric_value) < 0.5 * 10 ** (-decimals)
        if decimals > 0
        else abs(numeric_value) < 0.5
    ):
        numeric_value = 0.0

    return f"{numeric_value:10.{decimals}f}"


def matrix_to_lines(matrix, decimals):
    """Chuyển ma trận thành các dòng định dạng."""
    return [
        "[" + "  ".join(decimal_number(value, decimals) for value in row) + "]"
        for row in matrix
    ]


def print_matrix(matrix, decimals, prefix=""):
    """In một ma trận với tên nằm ở dòng giữa."""
    if not matrix:
        print(prefix + "[]")
        return

    lines = matrix_to_lines(matrix, decimals)
    middle = len(lines) // 2
    padding = " " * len(prefix)

    for i, line in enumerate(lines):
        print((prefix if i == middle else padding) + line)


def print_two_matrices(A, B, decimals, name_a="A = ", name_b="B = "):
    """In hai ma trận cạnh nhau."""
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


def print_cholesky_pair(U, decimals, step=None):
    """In U^T và U cạnh nhau thành một khối."""
    transpose_u = transpose_matrix(U)

    if step is None:
        title = "Hai ma trận U^T và U:"
        name_left = "U^T = "
        name_right = "U = "
    else:
        title = f"Sau bước {step}:"
        name_left = f"(U^({step}))^T = "
        name_right = f"U^({step}) = "

    print(f"\n{title}")
    print_two_matrices(transpose_u, U, decimals, name_a=name_left, name_b=name_right)


# ============================================================
# PHÉP TOÁN MA TRẬN
# ============================================================


def copy_matrix(matrix):
    return [row[:] for row in matrix]


def zero_matrix(rows, columns):
    return [[sp.Integer(0) for _ in range(columns)] for _ in range(rows)]


def identity_matrix(n):
    return [[sp.Integer(int(i == j)) for j in range(n)] for i in range(n)]


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
            sp.simplify(sum((A[i][k] * B[k][j] for k in range(middle)), sp.Integer(0)))
            for j in range(columns)
        ]
        for i in range(rows)
    ]


def matrices_equal(A, B):
    return (
        len(A) == len(B)
        and (not A or len(A[0]) == len(B[0]))
        and all(
            sp.simplify(A[i][j] - B[i][j]) == 0
            for i in range(len(A))
            for j in range(len(A[0]))
        )
    )


def is_symmetric(A):
    n = len(A)
    return all(sp.simplify(A[i][j] - A[j][i]) == 0 for i in range(n) for j in range(n))


def is_strictly_positive(value):
    """Kiểm tra một biểu thức thực có dương hay không."""
    value = sp.simplify(value)

    if value.is_positive is True:
        return True

    if value.is_positive is False or value == 0:
        return False

    return float(sp.N(value, 40)) > 0


def manual_rank(matrix):
    """Tính hạng bằng khử Gauss tự cài đặt."""
    if not matrix:
        return 0

    work = copy_matrix(matrix)
    rows = len(work)
    columns = len(work[0])
    pivot_row = 0

    for column in range(columns):
        candidate = None

        for row in range(pivot_row, rows):
            if sp.simplify(work[row][column]) != 0:
                candidate = row
                break

        if candidate is None:
            continue

        if candidate != pivot_row:
            work[pivot_row], work[candidate] = (work[candidate], work[pivot_row])

        for row in range(pivot_row + 1, rows):
            if sp.simplify(work[row][column]) != 0:
                factor = sp.simplify(work[row][column] / work[pivot_row][column])

                for j in range(column, columns):
                    work[row][j] = sp.simplify(
                        work[row][j] - factor * work[pivot_row][j]
                    )

        pivot_row += 1

        if pivot_row == rows:
            break

    return pivot_row


# ============================================================
# PHÂN TÁCH CHOLESKY: A = U^T * U
# ============================================================


def cholesky_decomposition(A, decimals, show_steps=True):
    """
    Phân tách Cholesky dạng:

        A = U^T * U

    U là ma trận tam giác trên và u_ii > 0.
    """
    n = len(A)

    if not is_symmetric(A):
        if show_steps:
            print(
                "\nKẾT LUẬN: A không đối xứng nên không thể phân tách Cholesky trực tiếp."
            )
        return None, "not_symmetric"

    U = zero_matrix(n, n)

    if show_steps:
        print("\n" + "=" * 96)
        print("PHÂN TÁCH CHOLESKY")
        print("=" * 96)

        print("\nTa phân tách ma trận A dưới dạng:")
        print("                              A = U^T * U")
        print("\nTrong đó U là ma trận tam giác trên và u_ii > 0.")

        print("\nCông thức tính các phần tử:")
        print("  u_ii = sqrt(a_ii - Σ(k=1..i-1) u_ki^2)")
        print("  u_ij = [a_ij - Σ(k=1..i-1) u_ki*u_kj] / u_ii,   với i < j")
        print("  u_ij = 0,   với i > j")

        print("\nThay lần lượt các giá trị theo công thức trên vào ma trận U:")
        print_matrix(U, decimals, prefix="U^(0) = ")

    for i in range(n):
        diagonal_sum = sp.simplify(sum((U[k][i] ** 2 for k in range(i)), sp.Integer(0)))

        diagonal_value = sp.simplify(A[i][i] - diagonal_sum)

        if not is_strictly_positive(diagonal_value):
            if show_steps:
                print("\n" + "-" * 96)
                print(
                    f"Dừng tại bước {i + 1} vì "
                    f"a_{i + 1},{i + 1} - Σu_k,{i + 1}^2 "
                    f"= {exact_number(diagonal_value)} <= 0."
                )
                print("Suy ra A không phải ma trận đối xứng xác định dương.")
                print_matrix(U, decimals, prefix=f"U^({i}) = ")

            return None, "not_positive_definite"

        U[i][i] = sp.sqrt(diagonal_value)

        for j in range(i + 1, n):
            cross_sum = sp.simplify(
                sum((U[k][i] * U[k][j] for k in range(i)), sp.Integer(0))
            )

            U[i][j] = sp.simplify((A[i][j] - cross_sum) / U[i][i])

        if show_steps:
            print("\n" + "-" * 96)
            print(
                f"Bước {i + 1}: d_{i + 1} = "
                f"{exact_number(diagonal_value)} > 0, "
                f"u_{i + 1},{i + 1} = "
                f"{exact_number(U[i][i])}."
            )
            print(f"Đã tính xong hàng {i + 1} của ma trận U.")
            print_matrix(U, decimals, prefix=f"U^({i + 1}) = ")

    if show_steps:
        print("\nHoàn thành việc thay các phần tử vào ma trận U.")

    return U, None


# ============================================================
# GIẢI U^T * Y = B BẰNG THẾ XUÔI
# ============================================================


def solve_transpose_u_y(U, B, show_steps=True):
    """
    Giải U^T * Y = B.
    Vì U^T là ma trận tam giác dưới nên dùng thế xuôi.
    """
    n = len(U)
    rhs_columns = len(B[0])
    Y = zero_matrix(n, rhs_columns)

    if show_steps:
        print("\n" + "=" * 96)
        print("GIẢI HỆ U^T * Y = B BẰNG PHÉP THẾ XUÔI")
        print("\nCông thức:")
        print("  y_iq = [b_iq - Σ(k=1..i-1) u_ki*y_kq] / u_ii")

    for column in range(rhs_columns):
        if show_steps:
            print(f"\nVế phải thứ {column + 1}:")

        for i in range(n):
            known_sum = sp.simplify(
                sum((U[k][i] * Y[k][column] for k in range(i)), sp.Integer(0))
            )

            Y[i][column] = sp.simplify((B[i][column] - known_sum) / U[i][i])

            if show_steps:
                print(
                    f"  y_{i + 1},{column + 1} "
                    f"= ({exact_number(B[i][column])} "
                    f"- {exact_number(known_sum)}) "
                    f"/ {exact_number(U[i][i])} "
                    f"= {exact_number(Y[i][column])}"
                )

    return Y


# ============================================================
# GIẢI U * X = Y BẰNG THẾ LÙI
# ============================================================


def solve_u_x(U, Y, show_steps=True):
    """Giải U*X = Y bằng thế lùi."""
    n = len(U)
    rhs_columns = len(Y[0])
    X = zero_matrix(n, rhs_columns)

    if show_steps:
        print("\n" + "=" * 96)
        print("GIẢI HỆ U * X = Y BẰNG PHÉP THẾ LÙI")
        print("\nCông thức:")
        print("  x_iq = [y_iq - Σ(k=i+1..n) u_ik*x_kq] / u_ii")

    for column in range(rhs_columns):
        if show_steps:
            print(f"\nVế phải thứ {column + 1}:")

        for i in range(n - 1, -1, -1):
            known_sum = sp.simplify(
                sum((U[i][k] * X[k][column] for k in range(i + 1, n)), sp.Integer(0))
            )

            X[i][column] = sp.simplify((Y[i][column] - known_sum) / U[i][i])

            if show_steps:
                print(
                    f"  x_{i + 1},{column + 1} "
                    f"= ({exact_number(Y[i][column])} "
                    f"- {exact_number(known_sum)}) "
                    f"/ {exact_number(U[i][i])} "
                    f"= {exact_number(X[i][column])}"
                )

    return X


# ============================================================
# KẾT QUẢ VÀ KIỂM TRA
# ============================================================


def determinant_from_u(U):
    determinant = sp.Integer(1)

    for i in range(len(U)):
        determinant = sp.simplify(determinant * U[i][i] ** 2)

    return determinant


def print_cholesky_result(A, U, decimals):
    print("\n" + "=" * 96)
    print("KẾT QUẢ PHÂN TÁCH CHOLESKY\n")

    print_cholesky_pair(U, decimals)

    product = multiply_matrices(transpose_matrix(U), U)

    print("\nKiểm tra U^T * U và A:")
    print_two_matrices(product, A, decimals, name_a="U^T*U = ", name_b="A = ")

    if matrices_equal(product, A):
        print("\nKết luận kiểm tra: U^T * U = A (đúng).")
    else:
        print("\nCảnh báo: U^T * U khác A.")

    determinant = determinant_from_u(U)

    print("\ndet(A) = det(U^T)*det(U)")
    print("       = (u_11*u_22*...*u_nn)^2")
    print(f"det(A) = {exact_number(determinant)}")

    return determinant


def process(A, B, decimals, inverse_mode=False):
    A_original = copy_matrix(A)
    B_original = copy_matrix(B)
    n = len(A)

    print("\n" + "=" * 96)

    if inverse_mode:
        print("TÌM MA TRẬN NGHỊCH ĐẢO BẰNG PHÂN TÁCH CHOLESKY")
        print("Ta giải hệ A*X = I.")
    else:
        print("GIẢI HỆ PHƯƠNG TRÌNH A*X = B BẰNG PHÂN TÁCH CHOLESKY")

    print("\nDữ liệu ban đầu:")
    print_matrix(A_original, decimals, prefix="A = ")
    print_matrix(B_original, decimals, prefix="I = " if inverse_mode else "B = ")

    U, error = cholesky_decomposition(A_original, decimals, show_steps=True)

    if error is not None:
        rank_A = manual_rank(A_original)
        print(f"\nHạng của A: {rank_A}")

        if inverse_mode:
            print(
                "\nKẾT LUẬN: Không thể tìm A^(-1) bằng Cholesky "
                "vì A không đối xứng xác định dương."
            )
        else:
            print(
                "\nKẾT LUẬN: Không thể giải hệ bằng Cholesky trực tiếp "
                "vì A không đối xứng xác định dương."
            )

        return None

    print_cholesky_result(A_original, U, decimals)

    print(f"\nHạng của A: {manual_rank(A_original)}")
    print("A đối xứng xác định dương nên hệ có nghiệm duy nhất với mọi vế phải B.")

    print("\nDo A = U^T * U nên:")
    print("A*X = B")
    print("U^T*U*X = B")
    print("Đặt Y = U*X, ta giải lần lượt:")
    print("U^T*Y = B")
    print("U*X = Y")

    Y = solve_transpose_u_y(U, B_original, show_steps=True)

    print("\nMa trận nghiệm trung gian Y:")
    print_matrix(Y, decimals, prefix="Y = ")

    X = solve_u_x(U, Y, show_steps=True)

    print("\n" + "=" * 96)

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
        print("\nKẾT LUẬN: Phân tách Cholesky yêu cầu A là ma trận vuông.")
        return None

    return A


def decompose_only():
    print("\n--- Phân tách A = U^T*U bằng phương pháp Cholesky ---")

    A = input_square_matrix()

    if A is None:
        return

    decimals = input_nonnegative_integer(
        "\nSố chữ số sau dấu phẩy [Enter = 7]: ", default=7
    )

    print("\nMa trận ban đầu:")
    print_matrix(A, decimals, prefix="A = ")

    U, error = cholesky_decomposition(A, decimals, show_steps=True)

    if error is not None:
        print("\nKẾT LUẬN: A không thỏa điều kiện đối xứng xác định dương.")
        return

    print_cholesky_result(A, U, decimals)

    print(f"\nHạng của A: {manual_rank(A)}")
    print(
        "\nKẾT LUẬN: A là ma trận đối xứng xác định dương "
        "và phân tách Cholesky thành công."
    )


def solve_system():
    print("\n--- Giải hệ A*X = B bằng phân tách Cholesky ---")

    A = input_square_matrix()

    if A is None:
        return

    n = len(A)
    k = input_positive_integer("\nNhập số cột của ma trận B (k): ")
    B = input_matrix("B", n, k)

    decimals = input_nonnegative_integer(
        "\nSố chữ số sau dấu phẩy [Enter = 7]: ", default=7
    )

    process(A, B, decimals, inverse_mode=False)


def find_inverse():
    print("\n--- Tìm A^(-1) bằng phân tách Cholesky ---")

    A = input_square_matrix()

    if A is None:
        return

    decimals = input_nonnegative_integer(
        "\nSố chữ số sau dấu phẩy [Enter = 7]: ", default=7
    )

    process(A, identity_matrix(len(A)), decimals, inverse_mode=True)


def main():
    print("=" * 96)
    print("PHƯƠNG PHÁP CHOLESKY - A = U^T * U")
    print("=" * 96)
    print("1. Phân tách ma trận A thành A = U^T*U")
    print("2. Giải hệ phương trình A*X = B bằng Cholesky")
    print("3. Tìm ma trận nghịch đảo A^(-1) bằng Cholesky")
    print("0. Thoát")
    print("Input: A đối xứng xác định dương. Output: U, nghiệm/nghịch đảo và kiểm tra.")

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
