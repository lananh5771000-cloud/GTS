import sys
import math
import cmath
import sympy as sp


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
                f"Lỗi: Dòng phải có đúng {expected_count} phần tử. "
                "Vui lòng nhập lại."
            )
            continue

        try:
            return [sp.Rational(token) for token in tokens]
        except (ValueError, TypeError, ZeroDivisionError):
            print(
                "Lỗi: Chỉ nhập số nguyên, số thập phân hoặc phân số hợp lệ "
                "(ví dụ 2, -3, 0.25, 1/3)."
            )


def input_matrix(name, rows, columns):
    """Nhập ma trận kích thước rows x columns."""
    print(f"\nNhập ma trận {name} ({rows}x{columns}):")

    return sp.Matrix([
        input_matrix_row(
            f"Nhập dòng {i + 1} (cách nhau bởi khoảng trắng): ",
            columns
        )
        for i in range(rows)
    ])


# ============================================================
# HIỂN THỊ
# ============================================================

def expression_string(value):
    """Chuyển biểu thức SymPy thành chuỗi dễ đọc."""
    return sp.sstr(sp.simplify(value))


def clean_complex(value, tolerance=1e-12):
    """Loại bỏ phần thực hoặc phần ảo rất nhỏ."""
    value = complex(value)

    real = 0.0 if abs(value.real) < tolerance else value.real
    imag = 0.0 if abs(value.imag) < tolerance else value.imag

    return complex(real, imag)


def format_complex(value, decimals):
    """Định dạng số thực hoặc phức."""
    value = clean_complex(value, 0.5 * 10 ** (-max(decimals, 1)))

    if value.imag == 0:
        return f"{value.real:.{decimals}f}"

    if value.real == 0:
        return f"{value.imag:.{decimals}f}i"

    sign = "+" if value.imag >= 0 else "-"
    return (
        f"{value.real:.{decimals}f}"
        f" {sign} {abs(value.imag):.{decimals}f}i"
    )


def print_symbolic_matrix(matrix, prefix=""):
    """In ma trận SymPy chính xác."""
    print(prefix)
    sp.pprint(matrix)


def numeric_matrix_lines(matrix, decimals):
    """Tạo các dòng ma trận số thực/phức."""
    rows = []

    for i in range(matrix.rows):
        values = []

        for j in range(matrix.cols):
            numeric = complex(sp.N(matrix[i, j], max(20, decimals + 8)))
            text = format_complex(numeric, decimals)
            values.append(text.rjust(max(12, len(text))))

        rows.append("[" + "  ".join(values) + "]")

    return rows


def print_numeric_matrix(matrix, decimals, prefix=""):
    """In ma trận SymPy dưới dạng số gần đúng."""
    lines = numeric_matrix_lines(matrix, decimals)
    middle = len(lines) // 2
    padding = " " * len(prefix)

    for i, line in enumerate(lines):
        print((prefix if i == middle else padding) + line)


def print_vector_symbolic(vector, name):
    print(f"{name} =")
    sp.pprint(vector)


def print_vector_numeric(vector, name, decimals):
    print_numeric_matrix(vector, decimals, prefix=f"{name} ≈ ")


# ============================================================
# CÁC HÀM MA TRẬN PHỤ TRỢ
# ============================================================

def is_zero(value):
    """Kiểm tra biểu thức có bằng 0 hay không."""
    simplified = sp.simplify(value)

    if simplified == 0:
        return True

    return simplified.is_zero is True


def permutation_matrix(n, first, second):
    """Ma trận hoán vị đổi vị trí first và second."""
    matrix = sp.eye(n)
    matrix.row_swap(first, second)
    return matrix


def leading_block(matrix, start, end):
    """Lấy khối vuông matrix[start:end, start:end]."""
    return matrix[start:end, start:end]


def is_companion_block(block):
    """Kiểm tra các hàng dưới của khối Frobenius."""
    size = block.rows

    if size == 1:
        return True

    for i in range(1, size):
        for j in range(size):
            expected = 1 if j == i - 1 else 0

            if not is_zero(block[i, j] - expected):
                return False

    return True


# ============================================================
# BIẾN ĐỔI DANILEVSKI
# ============================================================

def print_method_formula():
    print("\n" + "=" * 100)
    print("PHƯƠNG PHÁP DANILEVSKI")
    print("=" * 100)

    print("\nInput:")
    print("  • Ma trận vuông A cấp n.")
    print("Output:")
    print("  • Dạng Frobenius F, đa thức đặc trưng P_A(lambda).")
    print("  • Các trị riêng, bội đại số và cơ sở vector riêng tương ứng.")
    print("\nMục tiêu: đưa A về ma trận Frobenius F bằng các phép biến đổi đồng dạng.")
    print()
    print("Tại bước đang xét, chọn pivot a_(r,r-1) != 0.")
    print("Lập M giống ma trận đơn vị, riêng hàng r-1 được thay bằng hàng r của A.")
    print()
    print("B1. Thực hiện:")
    print("                    A_mới = M * A_cũ * M^(-1)")
    print()
    print("B2. Tích lũy ma trận Q sao cho:")
    print("                    F = Q^(-1) * A * Q")
    print()
    print("B3. Lập P_A(lambda)=det(lambda*I-F) từ dạng Frobenius.")
    print("B4. Giải P_A(lambda)=0 để tìm các trị riêng.")
    print("B5. Nếu F*Y = lambda*Y thì vector riêng của A là:")
    print("                    X = Q*Y")
    print()
    print("Nếu pivot bằng 0 nhưng trên cùng hàng còn phần tử khác 0 ở bên trái,")
    print("ta hoán vị đồng thời hàng và cột để bảo toàn tính đồng dạng.")


def danilevsky_transform(A, show_steps=True):
    """
    Biến đổi Danilevski.

    Kết quả:
        F = Q^(-1) * A * Q.

    blocks chứa các khoảng [start, end) của các khối Frobenius
    trên đường chéo.
    """
    n = A.rows
    F = A.copy()
    Q = sp.eye(n)

    blocks = []
    active_end = n
    transformation_number = 0
    swap_number = 0

    if show_steps:
        print_method_formula()
        print("\nMa trận ban đầu:")
        print_symbolic_matrix(F, "A^(0) =")

    while active_end > 0:
        if active_end == 1:
            blocks.append((0, 1))
            active_end = 0
            break

        row = active_end - 1

        while row > 0:
            pivot_column = row - 1
            pivot = sp.simplify(F[row, pivot_column])

            if is_zero(pivot):
                replacement_column = None

                for column in range(pivot_column):
                    if not is_zero(F[row, column]):
                        replacement_column = column
                        break

                if replacement_column is not None:
                    swap_number += 1

                    C = permutation_matrix(
                        n,
                        replacement_column,
                        pivot_column
                    )

                    if show_steps:
                        print("\n" + "-" * 100)
                        print(
                            f"PIVOT a_({row + 1},{pivot_column + 1}) = 0"
                        )
                        print(
                            f"Chọn a_({row + 1},{replacement_column + 1}) "
                            f"= {expression_string(F[row, replacement_column])} != 0."
                        )
                        print(
                            f"Hoán vị đồng thời hàng {replacement_column + 1} "
                            f"với hàng {pivot_column + 1},"
                        )
                        print(
                            f"và cột {replacement_column + 1} "
                            f"với cột {pivot_column + 1}."
                        )
                        print_symbolic_matrix(C, f"C_{swap_number} =")

                    # C là ma trận hoán vị trực giao: C^(-1) = C^T.
                    F = sp.simplify(C * F * C.T)
                    Q = sp.simplify(Q * C.T)

                    pivot = sp.simplify(F[row, pivot_column])

                    if show_steps:
                        print("\nSau phép hoán vị đồng dạng:")
                        print_symbolic_matrix(F, "A =")
                        print(
                            f"Pivot mới a_({row + 1},{pivot_column + 1}) "
                            f"= {expression_string(pivot)}."
                        )

                else:
                    # Không còn phần tử khác 0 bên trái pivot:
                    # tách khối Frobenius phía dưới.
                    blocks.append((row, active_end))

                    if show_steps:
                        print("\n" + "-" * 100)
                        print(
                            f"Hàng {row + 1} có các phần tử từ cột 1 "
                            f"đến cột {row} đều bằng 0."
                        )
                        print(
                            f"Tách khối Frobenius cấp {active_end - row} "
                            f"từ hàng/cột {row + 1} đến {active_end}."
                        )

                    active_end = row
                    break

            # Sau hoán vị, pivot chắc chắn khác 0.
            pivot = sp.simplify(F[row, pivot_column])

            if is_zero(pivot):
                raise ArithmeticError(
                    "Không tìm được pivot khác 0 sau phép hoán vị."
                )

            transformation_number += 1

            old_row = [
                sp.simplify(F[row, column])
                for column in range(n)
            ]

            M = sp.eye(n)
            M_inverse = sp.eye(n)

            for column in range(n):
                M[pivot_column, column] = old_row[column]

                if column == pivot_column:
                    M_inverse[pivot_column, column] = sp.simplify(
                        1 / pivot
                    )
                else:
                    M_inverse[pivot_column, column] = sp.simplify(
                        -old_row[column] / pivot
                    )

            if show_steps:
                print("\n" + "-" * 100)
                print(
                    f"BIẾN ĐỔI ĐỒNG DẠNG THỨ {transformation_number}"
                )
                print(
                    f"Pivot: a_({row + 1},{pivot_column + 1}) "
                    f"= {expression_string(pivot)}"
                )

                print_symbolic_matrix(
                    M,
                    f"M_{transformation_number} ="
                )

                print_symbolic_matrix(
                    M_inverse,
                    f"M_{transformation_number}^(-1) ="
                )

                print(
                    "\nÁp dụng A_mới = M*A_cũ*M^(-1)."
                )

            F = sp.simplify(M * F * M_inverse)
            Q = sp.simplify(Q * M_inverse)

            if show_steps:
                print_symbolic_matrix(
                    F,
                    f"A^({transformation_number}) ="
                )

                print(
                    "\nMa trận tích lũy Q, thỏa F = Q^(-1)*A*Q:"
                )
                print_symbolic_matrix(
                    Q,
                    f"Q^({transformation_number}) ="
                )

            row -= 1

        else:
            blocks.append((0, active_end))
            active_end = 0

    blocks.sort(key=lambda item: item[0])

    return F, Q, blocks


# ============================================================
# ĐA THỨC ĐẶC TRƯNG TỪ CÁC KHỐI FROBENIUS
# ============================================================

def companion_polynomial(block, variable):
    """
    Nếu hàng đầu của khối Frobenius là:
        [p1, p2, ..., pm]
    thì:
        phi(lambda) = lambda^m - p1*lambda^(m-1) - ... - pm.
    """
    size = block.rows
    polynomial = variable ** size

    for column in range(size):
        polynomial -= (
            block[0, column]
            * variable ** (size - column - 1)
        )

    return sp.expand(polynomial)


def characteristic_polynomial_from_blocks(F, blocks, variable):
    """Lập đa thức đặc trưng bằng tích đa thức các khối Frobenius."""
    block_polynomials = []
    total = sp.Integer(1)

    for start, end in blocks:
        block = leading_block(F, start, end)

        if not is_companion_block(block):
            raise ArithmeticError(
                "Khối thu được không có dạng Frobenius."
            )

        polynomial = companion_polynomial(
            block,
            variable
        )

        block_polynomials.append(
            (start, end, polynomial)
        )

        total = sp.expand(total * polynomial)

    return block_polynomials, sp.expand(total)


def print_characteristic_polynomial(F, blocks, variable):
    print("\n" + "=" * 100)
    print("MA TRẬN FROBENIUS VÀ ĐA THỨC ĐẶC TRƯNG")
    print("=" * 100)

    print("\nMa trận sau biến đổi Danilevski:")
    print_symbolic_matrix(F, "F =")

    if len(blocks) == 1:
        print("\nF là một ma trận Frobenius.")
    else:
        print(
            "\nF có dạng tam giác khối, các khối đường chéo "
            "là các ma trận Frobenius."
        )

    block_polynomials, total_polynomial = (
        characteristic_polynomial_from_blocks(
            F,
            blocks,
            variable
        )
    )

    for index, (start, end, polynomial) in enumerate(
        block_polynomials,
        start=1
    ):
        block = leading_block(F, start, end)

        print(
            f"\nKhối Frobenius P_{index}, cấp {end - start}:"
        )
        print_symbolic_matrix(block, f"P_{index} =")

        print(
            f"phi_{index}(lambda) = "
            f"{sp.sstr(sp.factor(polynomial))}"
        )

    print("\nĐa thức đặc trưng của A:")
    print(
        "P_A(lambda) = "
        + " * ".join(
            f"phi_{index}(lambda)"
            for index in range(1, len(block_polynomials) + 1)
        )
    )

    print(
        f"P_A(lambda) = {sp.sstr(sp.factor(total_polynomial))}"
    )

    print(
        f"P_A(lambda) khai triển = "
        f"{sp.sstr(total_polynomial)}"
    )

    return block_polynomials, total_polynomial


def polynomial_matrix_value(polynomial, variable, matrix):
    """Tính P(A) bằng sơ đồ Horner, không khai triển lũy thừa riêng lẻ."""
    poly = sp.Poly(sp.expand(polynomial), variable)
    identity = sp.eye(matrix.rows)
    result = sp.zeros(matrix.rows)

    for coefficient in poly.all_coeffs():
        result = sp.simplify(result * matrix + coefficient * identity)

    return sp.simplify(result)


def print_characteristic_checks(A, polynomial, variable):
    """Kiểm tra độc lập đa thức đặc trưng và định lý Cayley–Hamilton."""
    print("\n" + "=" * 100)
    print("KIỂM TRA ĐỘC LẬP ĐA THỨC ĐẶC TRƯNG")
    print("=" * 100)

    polynomial = sp.expand(polynomial)
    direct = sp.expand(A.charpoly(variable).as_expr())
    difference = sp.expand(polynomial - direct)

    print("\nĐa thức thu từ các khối Frobenius:")
    print(f"P_F(lambda) = {sp.sstr(polynomial)}")
    print("\nĐa thức tính trực tiếp để kiểm tra:")
    print(f"P_A(lambda) = det(lambda*E - A) = {sp.sstr(direct)}")
    print(f"\nP_F(lambda) - P_A(lambda) = {sp.sstr(difference)}")

    if difference == 0:
        print("Kết luận: Hai đa thức trùng nhau chính xác.")
    else:
        raise ArithmeticError(
            "Đa thức từ dạng Frobenius không trùng đa thức đặc trưng trực tiếp."
        )

    poly = sp.Poly(polynomial, variable)
    coefficients = poly.all_coeffs()
    trace_coefficient = sp.simplify(coefficients[1]) if A.rows >= 1 else 0
    constant_coefficient = sp.simplify(coefficients[-1])
    expected_trace_coefficient = sp.simplify(-sp.trace(A))
    expected_constant = sp.simplify((-1) ** A.rows * A.det())

    print("\nKiểm tra hệ số theo trace và determinant:")
    print(
        "Hệ số lambda^(n-1) = "
        f"{expression_string(trace_coefficient)}, "
        f"-trace(A) = {expression_string(expected_trace_coefficient)}"
    )
    print(
        "Hệ số tự do = "
        f"{expression_string(constant_coefficient)}, "
        f"(-1)^n det(A) = {expression_string(expected_constant)}"
    )

    cayley = polynomial_matrix_value(polynomial, variable, A)
    print("\nKiểm tra định lý Cayley–Hamilton P_A(A) = 0:")
    sp.pprint(cayley)

    if cayley == sp.zeros(A.rows):
        print("Kết luận: P_A(A) = 0 chính xác.")
    else:
        raise ArithmeticError("Kiểm tra Cayley–Hamilton không cho ma trận 0.")


# ============================================================
# GIẢI ĐA THỨC ĐẶC TRƯNG
# ============================================================

def group_numeric_roots(roots, tolerance):
    """Gộp các nghiệm số gần nhau để xác định bội."""
    groups = []

    for root in roots:
        value = complex(root)

        matched = False

        for group in groups:
            if abs(value - group["value"]) <= tolerance:
                count = group["multiplicity"]
                group["value"] = (
                    group["value"] * count + value
                ) / (count + 1)
                group["multiplicity"] += 1
                matched = True
                break

        if not matched:
            groups.append({
                "value": value,
                "multiplicity": 1
            })

    groups.sort(
        key=lambda item: (
            round(item["value"].real, 12),
            round(item["value"].imag, 12)
        )
    )

    return groups


def solve_characteristic_polynomial(
    polynomial,
    variable,
    decimals
):
    """
    Ưu tiên nghiệm chính xác.
    Nếu SymPy không biểu diễn đủ nghiệm chính xác, dùng nroots.
    """
    poly = sp.Poly(polynomial, variable)
    degree = poly.degree()

    exact_dictionary = sp.roots(
        poly.as_expr(),
        variable
    )

    exact_count = sum(
        exact_dictionary.values()
    )

    root_data = []

    if exact_count == degree:
        for root, multiplicity in exact_dictionary.items():
            root_data.append({
                "exact": True,
                "value": sp.simplify(root),
                "multiplicity": int(multiplicity)
            })

        root_data.sort(
            key=lambda item: (
                float(sp.re(sp.N(item["value"], 20))),
                float(sp.im(sp.N(item["value"], 20)))
            )
        )

        mode = "exact"
    else:
        precision = max(30, decimals + 20)

        numeric_roots = sp.nroots(
            poly,
            n=precision,
            maxsteps=300
        )

        tolerance = 10 ** (-min(10, max(5, decimals)))

        grouped = group_numeric_roots(
            numeric_roots,
            tolerance
        )

        for group in grouped:
            root_data.append({
                "exact": False,
                "value": group["value"],
                "multiplicity": group["multiplicity"]
            })

        mode = "numeric"

    return root_data, mode


def print_eigenvalues(
    polynomial,
    variable,
    decimals
):
    print("\n" + "=" * 100)
    print("TÌM TRỊ RIÊNG")
    print("=" * 100)

    print("\nGiải phương trình đặc trưng:")
    print(f"{sp.sstr(polynomial)} = 0")

    root_data, mode = solve_characteristic_polynomial(
        polynomial,
        variable,
        decimals
    )

    if mode == "exact":
        print("\nCác nghiệm được biểu diễn chính xác:")
    else:
        print(
            "\nĐa thức không biểu diễn thuận tiện toàn bộ nghiệm "
            "bằng công thức chính xác."
        )
        print("Chương trình sử dụng nghiệm gần đúng của đa thức.")

    for index, item in enumerate(root_data, start=1):
        multiplicity = item["multiplicity"]

        if item["exact"]:
            value = item["value"]
            approximate = complex(
                sp.N(value, max(20, decimals + 8))
            )

            print(
                f"\nlambda_{index} = {sp.sstr(value)}"
            )
            print(
                f"lambda_{index} ≈ "
                f"{format_complex(approximate, decimals)}"
            )
        else:
            value = item["value"]

            print(
                f"\nlambda_{index} ≈ "
                f"{format_complex(value, decimals)}"
            )

        print(f"Bội đại số: {multiplicity}")

    return root_data


# ============================================================
# KHÔNG GIAN NGHIỆM CỦA (F - lambda*I)Y = 0
# ============================================================

def exact_nullspace_basis(matrix):
    """Tìm cơ sở không gian nghiệm bằng Gauss–Jordan chính xác."""
    work = sp.Matrix(matrix)
    rows = work.rows
    columns = work.cols

    pivot_row = 0
    pivot_columns = []

    for column in range(columns):
        chosen = None

        for row in range(pivot_row, rows):
            if not is_zero(work[row, column]):
                chosen = row
                break

        if chosen is None:
            continue

        if chosen != pivot_row:
            work.row_swap(chosen, pivot_row)

        pivot = sp.simplify(work[pivot_row, column])

        for j in range(column, columns):
            work[pivot_row, j] = sp.simplify(
                work[pivot_row, j] / pivot
            )

        for row in range(rows):
            if row == pivot_row:
                continue

            factor = sp.simplify(work[row, column])

            if not is_zero(factor):
                for j in range(column, columns):
                    work[row, j] = sp.simplify(
                        work[row, j]
                        - factor * work[pivot_row, j]
                    )

        pivot_columns.append(column)
        pivot_row += 1

        if pivot_row == rows:
            break

    free_columns = [
        column
        for column in range(columns)
        if column not in pivot_columns
    ]

    basis = []

    for free in free_columns:
        vector = sp.zeros(columns, 1)
        vector[free, 0] = 1

        for row_index in range(
            len(pivot_columns) - 1,
            -1,
            -1
        ):
            pivot_column = pivot_columns[row_index]

            total = sp.Integer(0)

            for column in free_columns:
                total += (
                    work[row_index, column]
                    * vector[column, 0]
                )

            vector[pivot_column, 0] = sp.simplify(
                -total
            )

        basis.append(vector)

    return basis


def normalize_exact_vector(vector):
    """Chuẩn hóa để phần tử khác 0 cuối cùng bằng 1."""
    normalized = sp.Matrix(vector)

    divisor = None

    for i in range(vector.rows - 1, -1, -1):
        if not is_zero(vector[i, 0]):
            divisor = vector[i, 0]
            break

    if divisor is None:
        return normalized

    for i in range(vector.rows):
        normalized[i, 0] = sp.simplify(
            normalized[i, 0] / divisor
        )

    return normalized


def complex_rref_nullspace(matrix, tolerance):
    """
    Tìm cơ sở không gian nghiệm gần đúng bằng Gauss–Jordan phức.
    """
    rows = len(matrix)
    columns = len(matrix[0])

    work = [
        [complex(value) for value in row]
        for row in matrix
    ]

    scale = max(
        1.0,
        max(
            abs(value)
            for row in work
            for value in row
        )
    )

    threshold = tolerance * scale

    pivot_row = 0
    pivot_columns = []

    for column in range(columns):
        chosen = None
        largest = 0.0

        for row in range(pivot_row, rows):
            value = abs(work[row][column])

            if value > largest:
                largest = value
                chosen = row

        if chosen is None or largest <= threshold:
            continue

        if chosen != pivot_row:
            work[pivot_row], work[chosen] = (
                work[chosen],
                work[pivot_row]
            )

        pivot = work[pivot_row][column]

        for j in range(column, columns):
            work[pivot_row][j] /= pivot

        for row in range(rows):
            if row == pivot_row:
                continue

            factor = work[row][column]

            if abs(factor) > threshold:
                for j in range(column, columns):
                    work[row][j] -= (
                        factor * work[pivot_row][j]
                    )

        pivot_columns.append(column)
        pivot_row += 1

        if pivot_row == rows:
            break

    free_columns = [
        column
        for column in range(columns)
        if column not in pivot_columns
    ]

    # Với nghiệm gần đúng, đôi khi sai số làm ma trận có vẻ đủ hạng.
    # Khi đó bỏ pivot yếu nhất bằng cách chọn cột cuối làm biến tự do.
    if not free_columns:
        free_columns = [columns - 1]
        pivot_columns = [
            column
            for column in pivot_columns
            if column != columns - 1
        ]

    basis = []

    for free in free_columns:
        vector = [0j for _ in range(columns)]
        vector[free] = 1 + 0j

        for row_index in range(
            len(pivot_columns) - 1,
            -1,
            -1
        ):
            pivot_column = pivot_columns[row_index]

            total = 0j

            for column in free_columns:
                total += (
                    work[row_index][column]
                    * vector[column]
                )

            vector[pivot_column] = -total

        magnitude = math.sqrt(
            sum(abs(value) ** 2 for value in vector)
        )

        if magnitude != 0:
            vector = [
                value / magnitude
                for value in vector
            ]

        # Chọn pha để phần tử có môđun lớn nhất là số thực dương.
        largest_index = max(
            range(columns),
            key=lambda index: abs(vector[index])
        )

        if abs(vector[largest_index]) > 0:
            phase = (
                vector[largest_index]
                / abs(vector[largest_index])
            )

            vector = [
                value / phase
                for value in vector
            ]

        basis.append(vector)

    return basis


# ============================================================
# VECTOR RIÊNG
# ============================================================

def canonical_frobenius_vector(value, size):
    """
    Với một khối Frobenius đầy đủ:
        Y = [lambda^(n-1), ..., lambda, 1]^T.
    """
    return sp.Matrix([
        sp.simplify(value ** power)
        for power in range(size - 1, -1, -1)
    ])


def matrix_to_complex_list(matrix):
    return [
        [
            complex(sp.N(matrix[i, j], 30))
            for j in range(matrix.cols)
        ]
        for i in range(matrix.rows)
    ]


def multiply_complex_matrix_vector(matrix, vector):
    return [
        sum(
            matrix[i][j] * vector[j]
            for j in range(len(vector))
        )
        for i in range(len(matrix))
    ]


def complex_vector_to_sympy(vector):
    return sp.Matrix([
        sp.Float(value.real, 16)
        + sp.Float(value.imag, 16) * sp.I
        for value in vector
    ])


def eigenvectors_for_root(
    F,
    Q,
    blocks,
    root_item,
    decimals
):
    n = F.rows

    if root_item["exact"]:
        eigenvalue = root_item["value"]

        # Nếu chỉ có một khối Frobenius, dùng trực tiếp công thức trong tài liệu.
        if len(blocks) == 1 and blocks[0] == (0, n):
            basis_F = [
                canonical_frobenius_vector(
                    eigenvalue,
                    n
                )
            ]
        else:
            homogeneous = (
                F - eigenvalue * sp.eye(n)
            )

            basis_F = exact_nullspace_basis(
                homogeneous
            )

        exact_results = []

        for vector_F in basis_F:
            vector_F = normalize_exact_vector(
                vector_F
            )

            vector_A = sp.simplify(
                Q * vector_F
            )

            vector_A = normalize_exact_vector(
                vector_A
            )

            residual = sp.simplify(
                (Q.inv() * 0) if False
                else sp.zeros(n, 1)
            )

            exact_results.append({
                "exact": True,
                "Y": vector_F,
                "X": vector_A
            })

        return exact_results

    eigenvalue = complex(root_item["value"])

    F_complex = matrix_to_complex_list(F)
    Q_complex = matrix_to_complex_list(Q)

    if len(blocks) == 1 and blocks[0] == (0, n):
        vector_F = [
            eigenvalue ** power
            for power in range(n - 1, -1, -1)
        ]

        magnitude = math.sqrt(
            sum(abs(value) ** 2 for value in vector_F)
        )

        if magnitude != 0:
            vector_F = [
                value / magnitude
                for value in vector_F
            ]

        basis_numeric = [vector_F]
    else:
        shifted = [
            [
                F_complex[i][j]
                - (
                    eigenvalue
                    if i == j
                    else 0
                )
                for j in range(n)
            ]
            for i in range(n)
        ]

        basis_numeric = complex_rref_nullspace(
            shifted,
            tolerance=1e-9
        )

    numeric_results = []

    for vector_F in basis_numeric:
        vector_A = multiply_complex_matrix_vector(
            Q_complex,
            vector_F
        )

        magnitude = math.sqrt(
            sum(abs(value) ** 2 for value in vector_A)
        )

        if magnitude != 0:
            vector_A = [
                value / magnitude
                for value in vector_A
            ]

        largest_index = max(
            range(n),
            key=lambda index: abs(vector_A[index])
        )

        if abs(vector_A[largest_index]) > 0:
            phase = (
                vector_A[largest_index]
                / abs(vector_A[largest_index])
            )

            vector_A = [
                value / phase
                for value in vector_A
            ]

        numeric_results.append({
            "exact": False,
            "Y": complex_vector_to_sympy(vector_F),
            "X": complex_vector_to_sympy(vector_A)
        })

    return numeric_results


def print_eigenvectors(
    A,
    F,
    Q,
    blocks,
    root_data,
    decimals
):
    print("\n" + "=" * 100)
    print("TÌM VECTOR RIÊNG")
    print("=" * 100)

    print("\nTa có:")
    print("                    F = Q^(-1)*A*Q")
    print()
    print("Nếu F*Y = lambda*Y thì:")
    print("                    X = Q*Y")
    print("là vector riêng của A ứng với cùng trị riêng lambda.")

    A_complex = matrix_to_complex_list(A)

    for root_index, root_item in enumerate(
        root_data,
        start=1
    ):
        if root_item["exact"]:
            eigenvalue_text = sp.sstr(
                root_item["value"]
            )
        else:
            eigenvalue_text = format_complex(
                root_item["value"],
                decimals
            )

        print("\n" + "-" * 100)
        print(
            f"TRỊ RIÊNG lambda_{root_index} = "
            f"{eigenvalue_text}"
        )

        results = eigenvectors_for_root(
            F,
            Q,
            blocks,
            root_item,
            decimals
        )

        if not results:
            print(
                "Không dựng được vector riêng với độ chính xác hiện tại."
            )
            continue

        print(
            f"Số vector riêng độc lập tìm được: {len(results)}"
        )

        for vector_index, result in enumerate(
            results,
            start=1
        ):
            print(
                f"\nVector riêng thứ {vector_index} của F:"
            )

            if result["exact"]:
                print_vector_symbolic(
                    result["Y"],
                    "Y"
                )
            else:
                print_vector_numeric(
                    result["Y"],
                    "Y",
                    decimals
                )

            print("\nChuyển về vector riêng của A:")
            print("X = Q*Y")

            if result["exact"]:
                print_vector_symbolic(
                    result["X"],
                    "X"
                )

                eigenvalue = root_item["value"]
                residual = sp.simplify(
                    A * result["X"]
                    - eigenvalue * result["X"]
                )

                print("\nKiểm tra A*X - lambda*X:")
                print_vector_symbolic(
                    residual,
                    "R"
                )
            else:
                print_vector_numeric(
                    result["X"],
                    "X",
                    decimals
                )

                eigenvalue = complex(
                    root_item["value"]
                )

                vector_A = [
                    complex(
                        sp.N(result["X"][i, 0], 30)
                    )
                    for i in range(A.rows)
                ]

                AX = multiply_complex_matrix_vector(
                    A_complex,
                    vector_A
                )

                residual_values = [
                    AX[i] - eigenvalue * vector_A[i]
                    for i in range(A.rows)
                ]

                residual_norm = math.sqrt(
                    sum(
                        abs(value) ** 2
                        for value in residual_values
                    )
                )

                print(
                    "\nChuẩn phần dư "
                    "||A*X - lambda*X||_2 "
                    f"= {residual_norm:.12e}"
                )


# ============================================================
# KIỂM TRA KẾT QUẢ BIẾN ĐỔI
# ============================================================

def print_transformation_check(A, F, Q):
    print("\n" + "=" * 100)
    print("KIỂM TRA PHÉP BIẾN ĐỔI ĐỒNG DẠNG")
    print("=" * 100)

    left = sp.simplify(A * Q)
    right = sp.simplify(Q * F)
    difference = sp.simplify(left - right)

    print("\nKhông cần tính Q^(-1), ta kiểm tra tương đương:")
    print("                    A*Q = Q*F")

    print("\nA*Q =")
    sp.pprint(left)

    print("\nQ*F =")
    sp.pprint(right)

    print("\nA*Q - Q*F =")
    sp.pprint(difference)

    if difference == sp.zeros(A.rows):
        print(
            "\nKết luận kiểm tra: F = Q^(-1)*A*Q (đúng)."
        )
    else:
        print(
            "\nCảnh báo: phép kiểm tra đồng dạng chưa cho ma trận 0."
        )


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main():
    print("=" * 100)
    print("PHƯƠNG PHÁP DANILEVSKI TÌM TRỊ RIÊNG VÀ VECTOR RIÊNG")
    print("=" * 100)

    m = input_positive_integer(
        "Nhập số dòng của ma trận A (m): "
    )

    n = input_positive_integer(
        "Nhập số cột của ma trận A (n): "
    )

    A = input_matrix(
        "A",
        m,
        n
    )

    if m != n:
        print(
            "\nKẾT LUẬN: Phương pháp Danilevski yêu cầu "
            "A là ma trận vuông."
        )
        return

    decimals = input_nonnegative_integer(
        "\nNhập số chữ số sau dấu phẩy muốn hiển thị [Enter = 7]: ",
        default=7,
    )

    print("\nMa trận đầu vào:")
    print_symbolic_matrix(A, "A =")

    F, Q, blocks = danilevsky_transform(
        A,
        show_steps=True
    )

    print_transformation_check(
        A,
        F,
        Q
    )

    variable = sp.Symbol("lambda")

    block_polynomials, characteristic_polynomial = (
        print_characteristic_polynomial(
            F,
            blocks,
            variable
        )
    )

    print_characteristic_checks(
        A,
        characteristic_polynomial,
        variable
    )

    root_data = print_eigenvalues(
        characteristic_polynomial,
        variable,
        decimals
    )

    print_eigenvectors(
        A,
        F,
        Q,
        blocks,
        root_data,
        decimals
    )

    print("\n" + "=" * 100)
    print("KẾT LUẬN")
    print("=" * 100)
    print(
        "\nĐã đưa A về dạng Frobenius bằng phép biến đổi đồng dạng,"
    )
    print(
        "lập đa thức đặc trưng, tìm trị riêng và vector riêng tương ứng."
    )
    print("\nTóm tắt phổ của A:")
    for index, item in enumerate(root_data, start=1):
        if item["exact"]:
            exact_text = sp.sstr(item["value"])
            approximate = format_complex(
                complex(sp.N(item["value"], max(20, decimals + 8))),
                decimals,
            )
            value_text = f"{exact_text} ≈ {approximate}"
        else:
            value_text = format_complex(item["value"], decimals)
        print(
            f"  lambda_{index} = {value_text}; "
            f"bội đại số {item['multiplicity']}."
        )
    print("Mọi cặp (lambda, X) ở trên đều đã được kiểm tra bằng A*X-lambda*X.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except Exception as error:
        print(f"\nLỗi trong quá trình tính toán: {error}")
