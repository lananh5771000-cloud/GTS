import sys
from fractions import Fraction


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# NHẬP DỮ LIỆU
# ============================================================


def input_positive_integer(prompt, default=None):
    """Nhập số nguyên dương và bắt nhập lại nếu dữ liệu không hợp lệ."""
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
    """Nhập số nguyên không âm."""
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
    Chấp nhận số nguyên, số thập phân và phân số:
    2, -3, 0.25, 1/3, -5/7.
    """
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


def input_matrix(name, rows, columns):
    """Nhập ma trận rows x columns."""
    print(f"\nNhập ma trận {name} ({rows}x{columns}):")
    return [
        input_matrix_row(f"Nhập dòng {i + 1} (cách nhau bởi khoảng trắng): ", columns)
        for i in range(rows)
    ]


# ============================================================
# HIỂN THỊ
# ============================================================


def exact_number(value):
    """Hiển thị Fraction dưới dạng số nguyên hoặc phân số."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def decimal_number(value, decimals):
    """Định dạng số thập phân và loại bỏ -0.000."""
    number = float(value)
    threshold = 0.5 * 10 ** (-decimals) if decimals > 0 else 0.5

    if abs(number) < threshold:
        number = 0.0

    return f"{number:12.{decimals}f}"


def matrix_exact_lines(matrix):
    """Chuyển ma trận thành các dòng căn cột theo dạng chính xác."""
    if not matrix:
        return ["[]"]

    values = [[exact_number(value) for value in row] for row in matrix]

    column_count = len(values[0])
    widths = [
        max(len(values[i][j]) for i in range(len(values))) for j in range(column_count)
    ]

    return [
        "["
        + "  ".join(values[i][j].rjust(widths[j]) for j in range(column_count))
        + "]"
        for i in range(len(values))
    ]


def matrix_decimal_lines(matrix, decimals):
    return [
        "[" + "  ".join(decimal_number(value, decimals) for value in row) + "]"
        for row in matrix
    ]


def print_matrix_exact(matrix, prefix=""):
    """In ma trận chính xác, tên ma trận nằm tại dòng giữa."""
    lines = matrix_exact_lines(matrix)
    middle = len(lines) // 2
    padding = " " * len(prefix)

    for i, line in enumerate(lines):
        print((prefix if i == middle else padding) + line)


def print_matrix_decimal(matrix, decimals, prefix=""):
    """In ma trận dưới dạng thập phân."""
    lines = matrix_decimal_lines(matrix, decimals)
    middle = len(lines) // 2
    padding = " " * len(prefix)

    for i, line in enumerate(lines):
        print((prefix if i == middle else padding) + line)


def print_two_exact_matrices(left, right, left_name="A = ", right_name="B = "):
    """In hai ma trận chính xác nằm cạnh nhau."""
    left_lines = matrix_exact_lines(left)
    right_lines = matrix_exact_lines(right)

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


# ============================================================
# PHÉP TOÁN MA TRẬN
# ============================================================


def copy_matrix(matrix):
    return [row[:] for row in matrix]


def zero_matrix(rows, columns):
    return [[Fraction(0) for _ in range(columns)] for _ in range(rows)]


def identity_matrix(n):
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def leading_principal_submatrix(matrix, order):
    """Lấy ma trận con chính đầu cấp order."""
    return [matrix[i][:order] for i in range(order)]


def determinant_exact(matrix):
    """Tính định thức chính xác bằng khử Gauss trên Fraction."""
    if not matrix:
        return Fraction(1)

    rows = len(matrix)
    columns = len(matrix[0])

    if rows != columns:
        raise ValueError("Định thức chỉ xác định cho ma trận vuông.")

    work = copy_matrix(matrix)
    determinant = Fraction(1)
    sign = 1

    for column in range(rows):
        pivot = next(
            (row for row in range(column, rows) if work[row][column] != 0), None
        )

        if pivot is None:
            return Fraction(0)

        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign *= -1

        pivot_value = work[column][column]
        determinant *= pivot_value

        for row in range(column + 1, rows):
            if work[row][column] == 0:
                continue

            factor = work[row][column] / pivot_value
            work[row][column] = Fraction(0)

            for j in range(column + 1, rows):
                work[row][j] -= factor * work[column][j]

    return determinant if sign > 0 else -determinant


def leading_principal_determinants(matrix):
    """Trả về det của mọi ma trận con chính đầu cấp 1,...,n."""
    return [
        determinant_exact(leading_principal_submatrix(matrix, order))
        for order in range(1, len(matrix) + 1)
    ]


def permute_matrix(matrix, row_order, column_order):
    """Tạo B với B[i,j] = A[row_order[i], column_order[j]]."""
    return [
        [matrix[row_order[i]][column_order[j]] for j in range(len(column_order))]
        for i in range(len(row_order))
    ]


def row_permutation_matrix(row_order):
    """P sao cho (P*A)[i,:] = A[row_order[i],:]."""
    n = len(row_order)
    P = zero_matrix(n, n)

    for new_row, old_row in enumerate(row_order):
        P[new_row][old_row] = Fraction(1)

    return P


def column_permutation_matrix(column_order):
    """Q sao cho (A*Q)[:,j] = A[:,column_order[j]]."""
    n = len(column_order)
    Q = zero_matrix(n, n)

    for new_column, old_column in enumerate(column_order):
        Q[old_column][new_column] = Fraction(1)

    return Q


def complete_pivot_permutations(matrix):
    """
    Tìm hoán vị hàng và cột bằng khử Gauss với pivot toàn phần.

    Nếu A khả nghịch thì kết quả B=P*A*Q có mọi pivot Gauss khác 0,
    tương đương mọi ma trận con chính đầu của B đều khả nghịch.
    Phép khử chỉ dùng để chọn thứ tự viền, không dùng để tính A^(-1).
    """
    n = len(matrix)
    work = copy_matrix(matrix)
    row_order = list(range(n))
    column_order = list(range(n))
    pivot_log = []

    for step in range(n):
        pivot_position = None

        # Ưu tiên giữ nguyên phần tử chéo nếu có thể.
        if work[step][step] != 0:
            pivot_position = (step, step)
        else:
            # Ưu tiên chỉ đổi hàng.
            for row in range(step + 1, n):
                if work[row][step] != 0:
                    pivot_position = (row, step)
                    break

        if pivot_position is None:
            # Tiếp theo ưu tiên chỉ đổi cột.
            for column in range(step + 1, n):
                if work[step][column] != 0:
                    pivot_position = (step, column)
                    break

        if pivot_position is None:
            # Cuối cùng chọn một phần tử khác 0 bất kỳ trong khối còn lại.
            candidates = [
                (abs(work[row][column]), row, column)
                for row in range(step, n)
                for column in range(step, n)
                if work[row][column] != 0
            ]

            if not candidates:
                return None

            _, row, column = max(candidates)
            pivot_position = (row, column)

        pivot_row, pivot_column = pivot_position
        old_row = row_order[pivot_row]
        old_column = column_order[pivot_column]

        if pivot_row != step:
            work[step], work[pivot_row] = work[pivot_row], work[step]
            row_order[step], row_order[pivot_row] = (
                row_order[pivot_row],
                row_order[step],
            )

        if pivot_column != step:
            for row in range(n):
                work[row][step], work[row][pivot_column] = (
                    work[row][pivot_column],
                    work[row][step],
                )

            column_order[step], column_order[pivot_column] = (
                column_order[pivot_column],
                column_order[step],
            )

        pivot = work[step][step]

        if pivot == 0:
            return None

        pivot_log.append(
            {
                "step": step + 1,
                "selected_original_row": old_row,
                "selected_original_column": old_column,
                "pivot": pivot,
            }
        )

        for row in range(step + 1, n):
            if work[row][step] == 0:
                continue

            factor = work[row][step] / pivot
            work[row][step] = Fraction(0)

            for column in range(step + 1, n):
                work[row][column] -= factor * work[step][column]

    B = permute_matrix(matrix, row_order, column_order)
    leading_determinants = leading_principal_determinants(B)

    if any(value == 0 for value in leading_determinants):
        raise ArithmeticError(
            "Hoán vị pivot toàn phần không tạo được dãy khối đầu khả nghịch."
        )

    return {
        "row_order": row_order,
        "column_order": column_order,
        "B": B,
        "P": row_permutation_matrix(row_order),
        "Q": column_permutation_matrix(column_order),
        "pivot_log": pivot_log,
        "leading_determinants": leading_determinants,
    }


def restore_inverse_from_permutation(inverse_permuted, row_order, column_order):
    """
    Nếu B=P*A*Q thì A^(-1)=Q*B^(-1)*P.

    Theo chỉ số:
        A^(-1)[column_order[i], row_order[j]] = B^(-1)[i,j].
    """
    n = len(inverse_permuted)
    inverse = zero_matrix(n, n)

    for i in range(n):
        for j in range(n):
            inverse[column_order[i]][row_order[j]] = inverse_permuted[i][j]

    return inverse


def column_matrix(values):
    return [[value] for value in values]


def row_matrix(values):
    return [values[:]]


def multiply_matrices(left, right):
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("Kích thước hai ma trận không phù hợp để nhân.")

    rows = len(left)
    middle = len(right)
    columns = len(right[0])

    return [
        [
            sum((left[i][k] * right[k][j] for k in range(middle)), Fraction(0))
            for j in range(columns)
        ]
        for i in range(rows)
    ]


def add_matrices(left, right):
    return [
        [left[i][j] + right[i][j] for j in range(len(left[0]))]
        for i in range(len(left))
    ]


def scalar_multiply(matrix, scalar):
    return [[scalar * value for value in row] for row in matrix]


def matrices_equal(left, right):
    return (
        len(left) == len(right)
        and (not left or len(left[0]) == len(right[0]))
        and all(
            left[i][j] == right[i][j]
            for i in range(len(left))
            for j in range(len(left[0]))
        )
    )


def assemble_block_inverse(top_left, top_right, bottom_left, bottom_right):
    """
    Ghép:
        [top_left       top_right]
        [bottom_left    bottom_right]
    """
    old_order = len(top_left)
    new_order = old_order + 1
    result = zero_matrix(new_order, new_order)

    for i in range(old_order):
        for j in range(old_order):
            result[i][j] = top_left[i][j]

    for i in range(old_order):
        result[i][old_order] = top_right[i][0]

    for j in range(old_order):
        result[old_order][j] = bottom_left[0][j]

    result[old_order][old_order] = bottom_right

    return result


def manual_rank(matrix):
    """Tính hạng bằng phép khử Gauss chính xác."""
    if not matrix:
        return 0

    work = copy_matrix(matrix)
    rows = len(work)
    columns = len(work[0])
    pivot_row = 0

    for column in range(columns):
        candidates = [row for row in range(pivot_row, rows) if work[row][column] != 0]

        if not candidates:
            continue

        chosen = max(candidates, key=lambda row: abs(work[row][column]))

        if chosen != pivot_row:
            work[pivot_row], work[chosen] = (work[chosen], work[pivot_row])

        for row in range(pivot_row + 1, rows):
            if work[row][column] != 0:
                factor = work[row][column] / work[pivot_row][column]

                for j in range(column, columns):
                    work[row][j] -= factor * work[pivot_row][j]

        pivot_row += 1

        if pivot_row == rows:
            break

    return pivot_row


# ============================================================
# IN CÔNG THỨC PHƯƠNG PHÁP
# ============================================================


def print_method_formula():
    print("\n" + "=" * 100)
    print("PHƯƠNG PHÁP VIỀN QUANH")
    print("=" * 100)

    print("\nInput:")
    print("  • Ma trận vuông A cấp n.")
    print("Output:")
    print("  • Ma trận nghịch đảo A^(-1), nếu A khả nghịch.")
    print("  • Các khối viền, hệ số theta và phép kiểm tra hai phía.")
    print("\nB1. Chia ma trận cấp n thành bốn khối:")
    print()
    print("             [A_(n-1)    A_12]")
    print("  A_n      = [                 ]")
    print("             [  A_21      A_22]")
    print()
    print("B2. Giả sử đã biết A_(n-1)^(-1). Đặt:")
    print()
    print("  X      = A_(n-1)^(-1) * A_12")
    print("  Y      = A_21 * A_(n-1)^(-1)")
    print("  theta  = A_22 - Y*A_12")
    print()
    print("B3. Nếu theta != 0 thì:")
    print()
    print("  b_nn        = 1/theta")
    print("  beta_12     = -X/theta")
    print("  beta_21     = -Y/theta")
    print("  B_(n-1)     = A_(n-1)^(-1) + X*Y/theta")
    print()
    print("                    [B_(n-1)    beta_12]")
    print("  A_n^(-1)        = [                     ]")
    print("                    [ beta_21      b_nn   ]")
    print()
    print("B4. Ghép bốn khối để thu A_n^(-1), rồi tăng cấp và lặp lại.")
    print("\nĐiều kiện thực hiện ở mỗi cấp:")
    print("  - A_(n-1)^(-1) phải tồn tại.")
    print("  - theta = A_22 - A_21*A_(n-1)^(-1)*A_12 != 0.")
    print()
    print("Nếu A khả nghịch nhưng một ma trận con chính đầu bị suy biến,")
    print("ta được phép đổi độc lập thứ tự hàng và thứ tự cột:")
    print()
    print("                    B = P*A*Q")
    print()
    print("sao cho mọi ma trận con chính đầu của B đều khả nghịch.")
    print("Sau khi dùng viền quanh để tìm B^(-1), khôi phục:")
    print()
    print("                    A^(-1) = Q*B^(-1)*P")
    print()
    print("Các phép hoán vị chỉ chọn thứ tự viền; nghịch đảo vẫn được tính")
    print("hoàn toàn bằng công thức viền quanh.")


# ============================================================
# VIỀN QUANH ĐỆ QUY ĐÚNG THEO TÀI LIỆU
# ============================================================
# ============================================================
# VIỀN QUANH ĐỆ QUY ĐÚNG THEO TÀI LIỆU
# ============================================================


def inverse_order_one(A, symbol="A"):
    """Cơ sở cho ma trận cấp 1."""
    value = A[0][0]

    print("\n" + "-" * 100)
    print("TRƯỜNG HỢP CƠ SỞ: MA TRẬN CẤP 1")
    print_matrix_exact(A, prefix=f"{symbol}_1 = ")

    if value == 0:
        print(f"\na_11 = 0 nên {symbol}_1 không khả nghịch.")
        return None, None, {"type": "singular_leading_block", "order": 1}

    inverse = [[Fraction(1, 1) / value]]

    print(f"\n{symbol}_1^(-1) = [1/a_11] = [1/{exact_number(value)}]")
    print_matrix_exact(inverse, prefix=f"{symbol}_1^(-1) = ")

    return inverse, value, None


def inverse_order_two(A, symbol="A"):
    """Cơ sở cấp 2 bằng công thức nghịch đảo trực tiếp."""
    a11 = A[0][0]
    a12 = A[0][1]
    a21 = A[1][0]
    a22 = A[1][1]

    determinant = a11 * a22 - a12 * a21

    print("\n" + "-" * 100)
    print("TRƯỜNG HỢP CƠ SỞ: MA TRẬN CẤP 2")
    print_matrix_exact(A, prefix=f"{symbol}_2 = ")

    print("\nTính định thức:")
    print(f"det({symbol}_2) = a_11*a_22 - a_12*a_21")
    print(
        f"det({symbol}_2) = ({exact_number(a11)})"
        f"({exact_number(a22)})"
        f" - ({exact_number(a12)})"
        f"({exact_number(a21)})"
        f" = {exact_number(determinant)}"
    )

    if determinant == 0:
        print(f"\ndet({symbol}_2) = 0 nên {symbol}_2 không khả nghịch.")
        return None, None, {"type": "singular_leading_block", "order": 2}

    inverse = [
        [a22 / determinant, -a12 / determinant],
        [-a21 / determinant, a11 / determinant],
    ]

    print("\nTheo công thức:")
    print()
    print("                  1       [ a_22   -a_12]")
    print(f"  {symbol}_2^(-1) = -----------  [               ]")
    print(f"              det({symbol}_2)    [-a_21    a_11]")
    print()
    print("Suy ra:")
    print_matrix_exact(inverse, prefix=f"{symbol}_2^(-1) = ")

    check = multiply_matrices(A, inverse)
    print(f"\nKiểm tra {symbol}_2*{symbol}_2^(-1):")
    print_matrix_exact(check, prefix="Kết quả = ")

    return inverse, determinant, None


def bordering_inverse_recursive(A, symbol="A"):
    """
    Tìm nghịch đảo bằng viền quanh trên dãy ma trận con chính đầu.

    Ma trận đầu vào phải có mọi ma trận con chính đầu khả nghịch.
    """
    order = len(A)

    if order == 1:
        return inverse_order_one(A, symbol)

    if order == 2:
        return inverse_order_two(A, symbol)

    previous = leading_principal_submatrix(A, order - 1)

    inverse_previous, determinant_previous, error = bordering_inverse_recursive(
        previous, symbol
    )

    if error is not None:
        return None, None, error

    A12 = column_matrix([A[i][order - 1] for i in range(order - 1)])
    A21 = row_matrix(A[order - 1][: order - 1])
    A22 = A[order - 1][order - 1]

    X = multiply_matrices(inverse_previous, A12)
    Y = multiply_matrices(A21, inverse_previous)
    Y_A12 = multiply_matrices(Y, A12)[0][0]
    theta = A22 - Y_A12

    print("\n" + "-" * 100)
    print(f"VIỀN QUANH TỪ MA TRẬN CẤP {order - 1} LÊN MA TRẬN CẤP {order}")

    print(f"\nPhân hoạch {symbol}_{order}:")
    print()
    print(f"  {symbol}_({order - 1}) là ma trận con chính đầu cấp {order - 1}.")
    print_matrix_exact(previous, prefix=f"{symbol}_({order - 1}) = ")

    print("\nCác khối viền:")
    print_matrix_exact(A12, prefix="A_12 = ")
    print_matrix_exact(A21, prefix="A_21 = ")
    print(f"A_22 = {exact_number(A22)}")

    print("\nMa trận nghịch đảo đã biết:")
    print_matrix_exact(inverse_previous, prefix=f"{symbol}_({order - 1})^(-1) = ")

    print("\n1. Tính X:")
    print(f"X = {symbol}_({order - 1})^(-1) * A_12")
    print_matrix_exact(X, prefix="X = ")

    print("\n2. Tính Y:")
    print(f"Y = A_21 * {symbol}_({order - 1})^(-1)")
    print_matrix_exact(Y, prefix="Y = ")

    print("\n3. Tính theta:")
    print("theta = A_22 - Y*A_12")
    print(
        f"theta = {exact_number(A22)} - {exact_number(Y_A12)} = {exact_number(theta)}"
    )

    if theta == 0:
        print(f"\ntheta = 0 nên ma trận con chính đầu cấp {order} không khả nghịch.")
        return None, None, {"type": "zero_theta", "order": order}

    b_nn = Fraction(1, 1) / theta
    beta_12 = scalar_multiply(X, -b_nn)
    beta_21 = scalar_multiply(Y, -b_nn)
    correction = scalar_multiply(multiply_matrices(X, Y), b_nn)
    B_previous = add_matrices(inverse_previous, correction)

    inverse = assemble_block_inverse(B_previous, beta_12, beta_21, b_nn)

    determinant = determinant_previous * theta

    print("\n4. Tính các khối của ma trận nghịch đảo:")
    print(f"b_{order},{order} = 1/theta = {exact_number(b_nn)}")

    print("\nbeta_12 = -X/theta:")
    print_matrix_exact(beta_12, prefix="beta_12 = ")

    print("\nbeta_21 = -Y/theta:")
    print_matrix_exact(beta_21, prefix="beta_21 = ")

    print("\nB_(n-1) = A_(n-1)^(-1) + X*Y/theta:")
    print_matrix_exact(B_previous, prefix="B_(n-1) = ")

    print(f"\n5. Ghép các khối, thu được {symbol}_{order}^(-1):")
    print_matrix_exact(inverse, prefix=f"{symbol}_{order}^(-1) = ")

    print("\nQuan hệ định thức:")
    print(f"det({symbol}_{order}) = det({symbol}_{order - 1})*theta")
    print(
        f"det({symbol}_{order}) "
        f"= ({exact_number(determinant_previous)})"
        f"({exact_number(theta)}) "
        f"= {exact_number(determinant)}"
    )

    check = multiply_matrices(leading_principal_submatrix(A, order), inverse)

    print(f"\nKiểm tra {symbol}_{order}*{symbol}_{order}^(-1):")
    print_matrix_exact(check, prefix="Kết quả = ")

    return inverse, determinant, None


# ============================================================
# QUY TRÌNH CHÍNH
# ============================================================
# ============================================================
# QUY TRÌNH CHÍNH
# ============================================================


def find_inverse_by_bordering(A, decimals):
    order = len(A)
    A_original = copy_matrix(A)

    print("\n" + "=" * 100)
    print("TÌM MA TRẬN NGHỊCH ĐẢO BẰNG PHƯƠNG PHÁP VIỀN QUANH")
    print("=" * 100)

    print("\nMa trận đầu vào:")
    print_matrix_exact(A_original, prefix="A = ")

    print_method_formula()

    determinant_A = determinant_exact(A_original)

    print("\n" + "=" * 100)
    print("BƯỚC 1. KIỂM TRA KHẢ NGHỊCH VÀ CHỌN THỨ TỰ VIỀN")
    print("=" * 100)
    print(f"\ndet(A) = {exact_number(determinant_A)}")

    if determinant_A == 0:
        rank_A = manual_rank(A_original)
        print(f"rank(A) = {rank_A} < {order}.")
        print("KẾT LUẬN: A suy biến nên không có ma trận nghịch đảo.")
        return None

    original_leading = leading_principal_determinants(A_original)
    print("\nĐịnh thức các ma trận con chính đầu của A:")

    for level, value in enumerate(original_leading, start=1):
        print(f"  det(A_{level}) = {exact_number(value)}")

    needs_permutation = any(value == 0 for value in original_leading)

    if not needs_permutation:
        print(
            "\nMọi ma trận con chính đầu đều khả nghịch. "
            "Có thể viền quanh trực tiếp theo thứ tự ban đầu."
        )
        working = A_original
        inverse_symbol = "A"
        row_order = list(range(order))
        column_order = list(range(order))
        P = identity_matrix(order)
        Q = identity_matrix(order)
    else:
        failed_level = next(
            level for level, value in enumerate(original_leading, start=1) if value == 0
        )
        print(
            f"\nA khả nghịch nhưng A_{failed_level} suy biến, "
            "nên không thể viền trực tiếp theo thứ tự ban đầu."
        )
        print(
            "Ta chọn lại độc lập thứ tự hàng và cột bằng pivot toàn phần "
            "để chỉ thay đổi thứ tự viền."
        )

        permutation = complete_pivot_permutations(A_original)

        if permutation is None:
            raise ArithmeticError(
                "A khả nghịch nhưng không tìm được hoán vị viền phù hợp."
            )

        row_order = permutation["row_order"]
        column_order = permutation["column_order"]
        working = permutation["B"]
        P = permutation["P"]
        Q = permutation["Q"]
        inverse_symbol = "B"

        print(
            "\nThứ tự hàng mới (đánh số từ 1): "
            + str(tuple(index + 1 for index in row_order))
        )
        print(
            "Thứ tự cột mới (đánh số từ 1): "
            + str(tuple(index + 1 for index in column_order))
        )

        print("\nMa trận hoán vị hàng P:")
        print_matrix_exact(P, prefix="P = ")

        print("\nMa trận hoán vị cột Q:")
        print_matrix_exact(Q, prefix="Q = ")

        print("\nLập B = P*A*Q:")
        print_matrix_exact(working, prefix="B = ")

        print("\nĐịnh thức các ma trận con chính đầu của B:")
        for level, value in enumerate(permutation["leading_determinants"], start=1):
            print(f"  det(B_{level}) = {exact_number(value)} != 0")

        print("\nDo đó phương pháp viền quanh thực hiện được trên B.")

    print("\n" + "=" * 100)
    print("BƯỚC 2. THỰC HIỆN VIỀN QUANH")
    print("=" * 100)

    inverse_working, determinant_working, error = bordering_inverse_recursive(
        working, inverse_symbol
    )

    if error is not None:
        raise ArithmeticError(
            "Dãy ma trận con chính đầu đã được chứng minh khả nghịch "
            "nhưng công thức viền quanh vẫn gặp theta = 0."
        )

    if needs_permutation:
        print("\n" + "=" * 100)
        print("BƯỚC 3. KHÔI PHỤC NGHỊCH ĐẢO CỦA MA TRẬN BAN ĐẦU")
        print("=" * 100)
        print("\nVì B = P*A*Q nên:")
        print("                    A^(-1) = Q*B^(-1)*P")

        inverse_by_product = multiply_matrices(multiply_matrices(Q, inverse_working), P)
        inverse = restore_inverse_from_permutation(
            inverse_working, row_order, column_order
        )

        if not matrices_equal(inverse, inverse_by_product):
            raise ArithmeticError("Lỗi khôi phục thứ tự của ma trận nghịch đảo.")

        print("\nB^(-1) =")
        print_matrix_exact(inverse_working, prefix="B^(-1) = ")
        print("\nSuy ra:")
        print_matrix_exact(inverse, prefix="A^(-1) = ")
    else:
        inverse = inverse_working

    print("\n" + "=" * 100)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 100)

    print("\nMa trận nghịch đảo chính xác:")
    print_matrix_exact(inverse, prefix="A^(-1) = ")

    print(f"\nDạng thập phân ({decimals} chữ số sau dấu phẩy):")
    print_matrix_decimal(inverse, decimals, prefix="A^(-1) ≈ ")

    print(f"\ndet(A) = {exact_number(determinant_A)}")
    print(f"Hạng của A: {order}")

    if needs_permutation:
        determinant_P = determinant_exact(P)
        determinant_Q = determinant_exact(Q)
        print("\nKiểm tra quan hệ định thức của phép hoán vị:")
        print(
            "det(B) = det(P)*det(A)*det(Q) = "
            f"({exact_number(determinant_P)})"
            f"({exact_number(determinant_A)})"
            f"({exact_number(determinant_Q)}) "
            f"= {exact_number(determinant_working)}"
        )

    left_check = multiply_matrices(A_original, inverse)
    right_check = multiply_matrices(inverse, A_original)

    print("\n" + "=" * 100)
    print("KIỂM TRA KẾT QUẢ")
    print("=" * 100)

    print("\nA*A^(-1):")
    print_matrix_exact(left_check, prefix="A*A^(-1) = ")

    print("\nA^(-1)*A:")
    print_matrix_exact(right_check, prefix="A^(-1)*A = ")

    identity = identity_matrix(order)

    if matrices_equal(left_check, identity) and matrices_equal(right_check, identity):
        print("\nKẾT LUẬN: A khả nghịch và ma trận nghịch đảo tìm được là chính xác.")
        print("Ta có A*A^(-1) = A^(-1)*A = I.")
    else:
        raise ArithmeticError("Kết quả viền quanh không vượt qua kiểm tra chính xác.")

    return inverse


def main():
    print("=" * 100)
    print("PHƯƠNG PHÁP VIỀN QUANH TÌM MA TRẬN NGHỊCH ĐẢO")
    print("=" * 100)

    m = input_positive_integer("Nhập số dòng của ma trận A (m): ")

    n = input_positive_integer("Nhập số cột của ma trận A (n): ")

    A = input_matrix("A", m, n)

    if m != n:
        print(
            "\nKẾT LUẬN: A không phải ma trận vuông nên "
            "không có ma trận nghịch đảo thông thường."
        )
        return

    decimals = input_nonnegative_integer(
        "\nNhập số chữ số sau dấu phẩy muốn hiển thị [Enter = 7]: ",
        default=7,
    )

    find_inverse_by_bordering(A, decimals)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except (ValueError, ArithmeticError) as error:
        print(f"\nKhông thể thực hiện: {error}")
