import sys
from exam_format import exam_print as print
import math
import re
import sympy as sp
from input_utils import MathInputError, parse_exact, split_number_row


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


def input_zero_or_one(prompt, default=0):
    while True:
        raw = input(prompt).strip()

        if raw == "":
            return default

        if raw in {"0", "1"}:
            return int(raw)

        print("Lỗi: Vui lòng nhập 0 hoặc 1.")


def input_matrix_row(prompt, expected_count):
    """
    Nhập đúng expected_count phần tử.
    Chấp nhận số nguyên, số thập phân và phân số:
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
    """Chuyển biểu thức SymPy thành chuỗi dễ đọc, không rút gọn lại khi in."""
    text = sp.sstr(value).replace("**", "^")
    text = text.replace("lambda", "λ")
    text = text.replace("*I", "i").replace("I", "i")
    text = re.sub(r"(?<=\d)\*(?=[A-Za-zλ√i(])", "", text)
    text = re.sub(r"(?<=[A-Za-zλ√i)])\*(?=[A-Za-zλ√i(])", "", text)

    def replace_sqrt(match):
        inside = match.group(1)

        if re.fullmatch(r"-?\d+", inside):
            return f"√{inside}"

        return f"√({inside})"

    previous = None

    while previous != text:
        previous = text
        text = re.sub(r"sqrt\(([^()]+)\)", replace_sqrt, text)

    return text


def print_rule(character="=", width=92):
    print(character * width)


def print_section(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def print_subsection(title):
    print("\n" + "-" * 92)
    print(title)


def matrix_lines(matrix):
    """Tạo các dòng ma trận chính xác, canh cột để dễ chép."""
    matrix = sp.Matrix(matrix)

    if matrix.rows == 0 or matrix.cols == 0:
        return ["[]"]

    values = [
        [
            expression_string(matrix[i, j])
            for j in range(matrix.cols)
        ]
        for i in range(matrix.rows)
    ]

    widths = [
        max(len(values[i][j]) for i in range(matrix.rows))
        for j in range(matrix.cols)
    ]

    lines = []

    for row in values:
        body = "  ".join(row[j].rjust(widths[j]) for j in range(matrix.cols))
        lines.append(f"[ {body} ]")

    return lines


def print_lines_with_middle_prefix(lines, prefix=""):
    if prefix and not prefix.endswith(" "):
        prefix += " "

    middle = len(lines) // 2
    padding = " " * len(prefix)

    for i, line in enumerate(lines):
        print((prefix if i == middle else padding) + line)


def sub(value):
    return str(value).translate(str.maketrans("0123456789+-()ijs", "₀₁₂₃₄₅₆₇₈₉₊₋₍₎ᵢⱼₛ"))


def sup(value):
    return str(value).translate(str.maketrans("0123456789+-()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾"))


def a_symbol(row, column):
    return f"a{sub(row)},{sub(column)}"


def A_step(step):
    return f"A{sup(f'({step})')}"


def M_symbol(step, inverse=False):
    return f"M{sub(step)}" + ("⁻¹" if inverse else "")


def lambda_symbol(index):
    return f"λ{sub(index)}"


def fast_reduce(value):
    """Rút gọn nhẹ hơn simplify, tránh treo với biểu thức Danilevski lớn."""
    try:
        return sp.cancel(value)
    except Exception:
        return value


def fast_reduce_matrix(matrix):
    return sp.Matrix(matrix).applyfunc(fast_reduce)


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
    """In ma trận SymPy chính xác, canh cột gọn để dễ chép."""
    print_lines_with_middle_prefix(matrix_lines(matrix), prefix)


def print_symbolic_numeric_matrix(matrix, decimals, prefix=""):
    exact_lines = matrix_lines(matrix)
    numeric_lines = numeric_matrix_lines(sp.Matrix(matrix), decimals)
    middle = len(exact_lines) // 2
    exact_prefix = prefix if prefix.endswith(" ") else prefix + " "
    approx_prefix = "≈ "
    padding = " " * len(exact_prefix)
    exact_width = max(len(line) for line in exact_lines)

    for i, exact_line in enumerate(exact_lines):
        left = (exact_prefix if i == middle else padding) + exact_line
        right = (approx_prefix if i == middle else " " * len(approx_prefix)) + numeric_lines[i]
        print(left.ljust(len(exact_prefix) + exact_width + 4) + right)


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


def print_vector_symbolic(vector, name, decimals=None):
    if decimals is not None:
        print_symbolic_numeric_matrix(vector, decimals, prefix=f"{name} = ")
    else:
        print_symbolic_matrix(vector, prefix=f"{name} = ")


def print_vector_numeric(vector, name, decimals):
    print_numeric_matrix(vector, decimals, prefix=f"{name} ≈ ")


# ============================================================
# CÁC HÀM MA TRẬN PHỤ TRỢ
# ============================================================

def is_zero(value):
    """Kiểm tra biểu thức có bằng 0 hay không."""
    if value == 0:
        return True

    if getattr(value, "is_zero", None) is True:
        return True

    try:
        reduced = sp.cancel(value)
        if reduced == 0 or getattr(reduced, "is_zero", None) is True:
            return True
    except Exception:
        pass

    try:
        simplified = sp.simplify(value)
    except Exception:
        return False

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
    print_section("PHƯƠNG PHÁP DANILEVSKI")
    print("\nTa biến đổi đồng dạng A để giữ nguyên đa thức đặc trưng và trị riêng.")
    print("\nA⁽ᵏ⁾ = Mₖ A⁽ᵏ⁻¹⁾ Mₖ⁻¹")
    print("F = Q⁻¹AQ")
    print("P_A(λ) = det(λI - A) = det(λI - F)")
    print("\nỞ mỗi bước xét hàng s trong khối đang làm việc.")
    print("Mₖ giống I, riêng hàng s - 1 thay bằng hàng s của A⁽ᵏ⁻¹⁾.")
    print("Mục đích: đưa hàng s về dạng [0  0  ...  1  0].")


def danilevsky_transform(A, show_steps=True):
    """
    Biến đổi Danilevski.

    Kết quả:
        F = Q^(-1) * A * Q.

    blocks chứa các khoảng [start, end) của các khối Frobenius
    trên đường chéo.
    """
    if A.rows != A.cols:
        raise ValueError("Phương pháp Danilevski chỉ áp dụng cho ma trận vuông.")

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
        print_symbolic_matrix(F, "A⁽⁰⁾ =")

    while active_end > 0:
        if active_end == 1:
            blocks.append((0, 1))
            active_end = 0
            break

        row = active_end - 1

        while row > 0:
            pivot_column = row - 1
            pivot = fast_reduce(F[row, pivot_column])

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
                        print_subsection(
                            "TH2: đổi hàng/cột vì pivot bằng 0"
                        )
                        print(
                            f"\nTH2. {a_symbol(row + 1, pivot_column + 1)} = 0, "
                            f"nhưng ∃j < s - 1 sao cho {a_symbol(row + 1, 'j')} ≠ 0"
                        )
                        print(
                            "\nVì pivot đang bằng 0 nên chưa lập được Mₖ⁻¹."
                        )
                        print(
                            "Ta đổi đồng thời hàng và cột để đưa phần tử khác 0 "
                            "về vị trí pivot."
                        )
                        print(f"\nChọn j = {replacement_column + 1}")
                        print(
                            "C = ma trận hoán vị đổi hàng/cột "
                            f"{replacement_column + 1} và {pivot_column + 1}"
                        )
                        print_symbolic_matrix(C, "C =")
                        print("\nA ← C A Cᵀ")

                    # C là ma trận hoán vị trực giao: C^(-1) = C^T.
                    F = fast_reduce_matrix(C * F * C.T)
                    Q = fast_reduce_matrix(Q * C.T)

                    pivot = fast_reduce(F[row, pivot_column])

                    if show_steps:
                        print("\nKhi đó pivot mới:")
                        print(f"{a_symbol(row + 1, pivot_column + 1)} = {expression_string(pivot)} ≠ 0")
                        print("\nQuay về TH1.")
                        print_symbolic_matrix(F, "A =")

                else:
                    # Không còn phần tử khác 0 bên trái pivot:
                    # tách khối Frobenius phía dưới.
                    blocks.append((row, active_end))

                    if show_steps:
                        print_subsection("TH3: tách khối")
                        print(
                            f"\nTH3. {a_symbol(row + 1, 'j')} = 0, "
                            f"∀j = 1, 2, ..., {row}"
                        )
                        print(
                            "\nVì phía trái hàng s toàn 0 nên không có pivot để chia."
                        )
                        print(
                            "Do đó không tiếp tục biến đổi trên toàn ma trận, "
                            "mà tách khối."
                        )
                        print("\nKhi đó A có dạng tam giác khối:")
                        print("\nA = [ A₁   B")
                        print("      0    A₂ ]")
                        print("\nSuy ra:")
                        print("P_A(λ) = P_A₁(λ) · P_A₂(λ)")
                        if active_end - row == 1:
                            print("\nNếu A₂ = [aₛₛ] thì:")
                            print(
                                "P_A₂(λ) = λ - "
                                f"{expression_string(F[row, row])}"
                            )
                            print("\nDo đó:")
                            print(
                                "P_A(λ) = P_A₁(λ)"
                                f"(λ - {expression_string(F[row, row])})"
                            )
                        print(
                            "\nSau đó tiếp tục Danilevsky cho khối A₁, "
                            "không ép toàn bộ A thành một khối Frobenius."
                        )

                    active_end = row
                    break

            # Sau hoán vị, pivot chắc chắn khác 0.
            pivot = fast_reduce(F[row, pivot_column])

            if is_zero(pivot):
                raise ArithmeticError(
                    "Không tìm được pivot khác 0 sau phép hoán vị."
                )

            transformation_number += 1

            old_row = [
                fast_reduce(F[row, column])
                for column in range(n)
            ]

            M = sp.eye(n)
            M_inverse = sp.eye(n)

            for column in range(n):
                M[pivot_column, column] = old_row[column]

                if column == pivot_column:
                    M_inverse[pivot_column, column] = fast_reduce(
                        1 / pivot
                    )
                else:
                    M_inverse[pivot_column, column] = fast_reduce(
                        -old_row[column] / pivot
                    )

            if show_steps:
                print_subsection(
                    f"BƯỚC {transformation_number}"
                )
                print(
                    f"\nTH1. {a_symbol(row + 1, pivot_column + 1)} ≠ 0"
                )
                print(f"{a_symbol(row + 1, pivot_column + 1)} = {expression_string(pivot)}")
                print(
                    "\nVì pivot khác 0 nên lập được Mₖ và Mₖ⁻¹."
                )
                print(
                    "Phép đồng dạng dưới đây giữ nguyên đa thức đặc trưng."
                )
                print("\nLập Mₖ, Mₖ⁻¹:")

                print_symbolic_matrix(
                    M,
                    f"{M_symbol(transformation_number)} ="
                )

                print_symbolic_matrix(
                    M_inverse,
                    f"{M_symbol(transformation_number, inverse=True)} ="
                )

                print(
                    f"\n{A_step(transformation_number)} = "
                    f"{M_symbol(transformation_number)} "
                    f"{A_step(transformation_number - 1)} "
                    f"{M_symbol(transformation_number, inverse=True)}"
                )

            F = fast_reduce_matrix(M * F * M_inverse)
            Q = fast_reduce_matrix(Q * M_inverse)

            if show_steps:
                print_symbolic_matrix(
                    F,
                    f"{A_step(transformation_number)} ="
                )
                active_row = [
                    F[row, column]
                    for column in range(active_end)
                ]
                print("\nSau biến đổi:")
                print(
                    f"hàng {row + 1} có dạng "
                    f"[ {'  '.join(expression_string(value) for value in active_row)} ]"
                )
                print(
                    "Như vậy trong khối đang xét đã tạo được một hàng "
                    "của dạng Frobenius."
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
    print_section("MA TRẬN FROBENIUS VÀ ĐA THỨC ĐẶC TRƯNG")

    print_symbolic_matrix(F, "F =")

    if len(blocks) == 1:
        print("\nF là ma trận Frobenius.")
        print("Đa thức đặc trưng đọc trực tiếp từ hàng đầu của F.")
    else:
        print(
            "\nF là ma trận tam giác khối với các khối Frobenius "
            "trên đường chéo."
        )
        print(
            "Vì F tam giác khối nên det(λI - F) bằng tích định thức "
            "các khối đường chéo."
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

        print_subsection(f"Khối {index}: hàng/cột {start + 1}..{end}")
        print_symbolic_matrix(block, f"F{sub(index)} =")
        print(
            f"Với khối Frobenius F{sub(index)}, đa thức P{sub(index)}(λ) "
            "lấy theo hàng đầu."
        )

        print(
            f"P{sub(index)}(λ) = "
            f"{expression_string(sp.factor(polynomial))}"
        )

    print_subsection("Đa thức đặc trưng")
    print("P_A(λ) = det(λI - A)")
    print("Do các phép biến đổi là đồng dạng nên P_A(λ) = P_F(λ).")

    if len(block_polynomials) == 1:
        size = block_polynomials[0][1] - block_polynomials[0][0]
        terms = [f"λ{sup(size)}"]
        for index in range(1, size + 1):
            power = size - index
            if power == 0:
                terms.append(f"- p{sub(index)}")
            elif power == 1:
                terms.append(f"- p{sub(index)}λ")
            else:
                terms.append(f"- p{sub(index)}λ{sup(power)}")
        print("P_A(λ) = " + " ".join(terms))
    else:
        print(
            "P_A(λ) = "
            + " · ".join(
                f"P{sub(index)}(λ)"
                for index in range(1, len(block_polynomials) + 1)
            )
        )

    print(
        "P_A(λ) = "
        + " · ".join(
            f"({expression_string(sp.factor(polynomial))})"
            for _start, _end, polynomial in block_polynomials
        )
    )

    print(f"P_A(λ) = {expression_string(sp.factor(total_polynomial))}")
    print(f"P_A(λ) = {expression_string(total_polynomial)}")

    return block_polynomials, total_polynomial


def print_block_summary(blocks):
    print_section("TÓM TẮT KHỐI")

    for index, (start, end) in enumerate(blocks, start=1):
        print(f"Khối {index}: hàng/cột {start + 1}..{end}")

    if len(blocks) == 1:
        print("\nF là ma trận Frobenius.")
        print("Đa thức đặc trưng đọc từ hàng đầu của F.")
    else:
        print(
            "\nF là ma trận tam giác khối với các khối Frobenius "
            "trên đường chéo."
        )
        print("Đa thức đặc trưng bằng tích các đa thức khối.")


def polynomial_matrix_value(polynomial, variable, matrix):
    """Tính P(A) bằng sơ đồ Horner, không khai triển lũy thừa riêng lẻ."""
    poly = sp.Poly(sp.expand(polynomial), variable)
    identity = sp.eye(matrix.rows)
    result = sp.zeros(matrix.rows)

    for coefficient in poly.all_coeffs():
        result = fast_reduce_matrix(result * matrix + coefficient * identity)

    return fast_reduce_matrix(result)


def print_characteristic_checks(A, polynomial, variable, exact_check=True):
    """Kiểm tra độc lập đa thức đặc trưng và định lý Cayley–Hamilton."""
    print_section("KIỂM TRA NGẮN")

    if not exact_check:
        print("\nBỏ qua kiểm tra symbolic trực tiếp để chương trình chạy nhanh.")
        print("Đa thức đặc trưng đã được lập từ dạng Frobenius ở trên.")
        return

    polynomial = sp.expand(polynomial)
    direct = sp.expand(A.charpoly(variable).as_expr())
    difference = sp.expand(polynomial - direct)

    print("\nSo sánh với det(λI - A):")
    print(f"  P_Danilevski(λ) = {expression_string(polynomial)}")
    print(f"  det(λI - A)     = {expression_string(direct)}")
    print(f"  Hiệu                 = {expression_string(difference)}")

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

    print("\nĐối chiếu hệ số:")
    print(f"  Hệ số λⁿ⁻¹ = {expression_string(trace_coefficient)}")
    print(
        "  Hệ số tự do        = "
        f"{expression_string(constant_coefficient)}, "
        f"(-1)^n det(A) = {expression_string(expected_constant)}"
    )

    cayley = polynomial_matrix_value(polynomial, variable, A)
    print("\nKiểm tra Cayley-Hamilton:")
    print_symbolic_matrix(cayley, "P_A(A) = ")

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

    root_data = []
    exact_dictionary = {}
    exact_count = 0

    if degree <= 3:
        exact_dictionary = sp.roots(
            poly.as_expr(),
            variable
        )

        exact_count = sum(
            exact_dictionary.values()
        )

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
    print_section("TÌM TRỊ RIÊNG")

    print("\nGiải phương trình đặc trưng:")
    print(f"{expression_string(polynomial)} = 0")

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

            print(f"\n{lambda_symbol(index)} = {expression_string(value)}")
            print(f"          ≈ {format_complex(approximate, decimals)}")
        else:
            value = item["value"]

            print(f"\n{lambda_symbol(index)} ≈ {format_complex(value, decimals)}")

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
    Ghi chú: nếu tài liệu PDF dùng quy ước ngược [1, lambda, ..., lambda^(n-1)]^T
    thì đó chỉ là cách viết khác; code giữ quy ước phù hợp với khối Frobenius đang tạo.
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


def normalize_complex_vector(vector):
    magnitude = math.sqrt(
        sum(abs(value) ** 2 for value in vector)
    )

    if magnitude != 0:
        vector = [
            value / magnitude
            for value in vector
        ]

    largest_index = max(
        range(len(vector)),
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

    return vector


def complex_residual_norm(matrix, eigenvalue, vector):
    AX = multiply_complex_matrix_vector(
        matrix,
        vector
    )

    residual_values = [
        AX[i] - eigenvalue * vector[i]
        for i in range(len(vector))
    ]

    return math.sqrt(
        sum(
            abs(value) ** 2
            for value in residual_values
        )
    )


def complex_matrix_norm(matrix):
    return math.sqrt(
        sum(
            abs(value) ** 2
            for row in matrix
            for value in row
        )
    )


def direct_exact_eigenvectors(A, eigenvalue):
    homogeneous = (
        A - eigenvalue * sp.eye(A.rows)
    )

    basis = exact_nullspace_basis(
        homogeneous
    )

    results = []

    for vector_A in basis:
        vector_A = normalize_exact_vector(
            vector_A
        )

        residual = fast_reduce_matrix(
            A * vector_A
            - eigenvalue * vector_A
        )

        residual_norm = 0.0
        if residual != sp.zeros(A.rows, 1):
            residual_norm = math.sqrt(
                sum(
                    abs(complex(sp.N(residual[i, 0], 30))) ** 2
                    for i in range(A.rows)
                )
            )

        results.append({
            "exact": True,
            "Y": None,
            "X": vector_A,
            "source": "Giải trực tiếp (A − λI)X = 0 trên ma trận gốc A.",
            "residual": residual,
            "residual_norm": residual_norm,
            "certified": residual_norm <= 1e-10,
        })

    return results


def direct_numeric_eigenvectors(A, eigenvalue, decimals):
    A_complex = matrix_to_complex_list(A)
    shifted = [
        [
            A_complex[i][j]
            - (
                eigenvalue
                if i == j
                else 0
            )
            for j in range(A.cols)
        ]
        for i in range(A.rows)
    ]

    tolerance = 10 ** (-min(9, max(6, decimals)))
    basis_numeric = complex_rref_nullspace(
        shifted,
        tolerance=tolerance
    )

    matrix_norm = complex_matrix_norm(
        A_complex
    )

    results = []

    for vector_A in basis_numeric:
        vector_A = normalize_complex_vector(
            vector_A
        )

        residual_norm = complex_residual_norm(
            A_complex,
            eigenvalue,
            vector_A
        )

        denominator = max(
            1.0,
            matrix_norm + abs(eigenvalue)
        )

        relative_residual = residual_norm / denominator

        results.append({
            "exact": False,
            "Y": None,
            "X": complex_vector_to_sympy(vector_A),
            "source": "Giải gần đúng (A − λI)X = 0 bằng Gauss-Jordan phức có pivot.",
            "residual": None,
            "residual_norm": residual_norm,
            "relative_residual": relative_residual,
            "certified": relative_residual <= max(1e-7, 10 ** (-max(4, decimals - 2))),
        })

    return results


def eigenvectors_for_root(
    A,
    F,
    Q,
    blocks,
    root_item,
    decimals
):
    n = F.rows

    if root_item["exact"]:
        eigenvalue = root_item["value"]
        direct_results = direct_exact_eigenvectors(
            A,
            eigenvalue
        )

        if direct_results:
            return direct_results

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

            residual = fast_reduce_matrix(
                A * vector_A
                - eigenvalue * vector_A
            )
            residual_norm = 0.0
            if residual != sp.zeros(A.rows, 1):
                residual_norm = math.sqrt(
                    sum(
                        abs(complex(sp.N(residual[i, 0], 30))) ** 2
                        for i in range(A.rows)
                    )
                )

            exact_results.append({
                "exact": True,
                "Y": vector_F,
                "X": vector_A,
                "source": "Dựng qua F = Q⁻¹AQ rồi kiểm tra lại trên A gốc.",
                "residual": residual,
                "residual_norm": residual_norm,
                "certified": residual_norm <= 1e-10,
            })

        return exact_results

    eigenvalue = complex(root_item["value"])
    direct_results = direct_numeric_eigenvectors(
        A,
        eigenvalue,
        decimals
    )

    if direct_results and any(result["certified"] for result in direct_results):
        return direct_results

    F_complex = matrix_to_complex_list(F)
    Q_complex = matrix_to_complex_list(Q)

    if len(blocks) == 1 and blocks[0] == (0, n):
        vector_F = [
            eigenvalue ** power
            for power in range(n - 1, -1, -1)
        ]

        vector_F = normalize_complex_vector(
            vector_F
        )

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
    A_complex = matrix_to_complex_list(A)
    matrix_norm = complex_matrix_norm(A_complex)

    for vector_F in basis_numeric:
        vector_A = multiply_complex_matrix_vector(
            Q_complex,
            vector_F
        )

        vector_A = normalize_complex_vector(
            vector_A
        )

        residual_norm = complex_residual_norm(
            A_complex,
            eigenvalue,
            vector_A
        )
        relative_residual = residual_norm / max(1.0, matrix_norm + abs(eigenvalue))

        numeric_results.append({
            "exact": False,
            "Y": complex_vector_to_sympy(vector_F),
            "X": complex_vector_to_sympy(vector_A),
            "source": "Dựng qua F = Q⁻¹AQ rồi kiểm tra lại trên A gốc.",
            "residual": None,
            "residual_norm": residual_norm,
            "relative_residual": relative_residual,
            "certified": relative_residual <= max(1e-7, 10 ** (-max(4, decimals - 2))),
        })

    if direct_results:
        numeric_results.extend(direct_results)

    return numeric_results


def print_eigenvectors(
    A,
    F,
    Q,
    blocks,
    root_data,
    decimals
):
    print_section("TÌM VECTOR RIÊNG")

    print("\nTa có:")
    print("  F = Q^(-1)*A*Q")
    print("  Nếu F*Y = lambda*Y thì X = Q*Y.")
    print("  Kết quả cuối ưu tiên giải trực tiếp (A − λI)X = 0 trên A gốc.")
    print(
        "  Lưu ý: với khối Frobenius, code dùng vector [λ^(r-1), ..., λ, 1]^T; "
        "nếu PDF ghi ngược chiều thì đó là quy ước trình bày khác."
    )

    conclusion_vectors = []

    for root_index, root_item in enumerate(
        root_data,
        start=1
    ):
        if root_item["exact"]:
            eigenvalue_text = expression_string(
                root_item["value"]
            )
        else:
            eigenvalue_text = format_complex(
                root_item["value"],
                decimals
            )

        print_subsection(f"Với {lambda_symbol(root_index)} = {eigenvalue_text}")

        results = eigenvectors_for_root(
            A,
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

        algebraic_multiplicity = root_item["multiplicity"]
        geometric_multiplicity = len(results)
        print(f"Bội đại số: {algebraic_multiplicity}")
        print(f"Bội hình học tìm được: {geometric_multiplicity}")
        if geometric_multiplicity < algebraic_multiplicity:
            print(
                "λ có thiếu vector riêng độc lập; "
                "A có thể không chéo hóa được tại λ này."
            )

        for vector_index, result in enumerate(
            results,
            start=1
        ):
            print(f"\nVector riêng thứ {vector_index}:")
            print(result["source"])

            if result.get("Y") is not None:
                print("\nVector riêng của F:")
                if result["exact"]:
                    print_vector_symbolic(
                        result["Y"],
                        "Y",
                        decimals
                    )
                else:
                    print_vector_numeric(
                        result["Y"],
                        "Y",
                        decimals
                    )

                print("\nSuy ra vector riêng của A:")
                print("X = Q*Y")

            if result["exact"]:
                print_vector_symbolic(
                    result["X"],
                    "X",
                    decimals
                )

                eigenvalue = root_item["value"]
                residual = result["residual"]

                print("\nKiểm tra:")
                print_vector_symbolic(
                    residual,
                    "A*X - λX"
                )
                print(
                    "Chuẩn phần dư "
                    "||A*X - λX||_2 "
                    f"= {result['residual_norm']:.12e}"
                )
            else:
                print_vector_numeric(
                    result["X"],
                    "X",
                    decimals
                )

                print(
                    "\nChuẩn phần dư "
                    "||A*X - λX||_2 "
                    f"= {result['residual_norm']:.12e}"
                )
                print(f"Phần dư tương đối = {result['relative_residual']:.12e}")

            if result["certified"]:
                print("Kết luận: vector riêng đạt kiểm tra phần dư.")
            else:
                print("Kết luận: không chứng nhận được vector riêng trên A gốc.")
                continue

            conclusion_vectors.append({
                "root_index": root_index,
                "vector": result["X"],
                "exact": result["exact"],
                "certified": result["certified"],
            })

    return conclusion_vectors


# ============================================================
# KIỂM TRA KẾT QUẢ BIẾN ĐỔI
# ============================================================

def print_transformation_check(A, F, Q, exact_check=True):
    print_section("KIỂM TRA PHÉP BIẾN ĐỔI ĐỒNG DẠNG")

    if not exact_check:
        print("\nBỏ qua kiểm tra A*Q = Q*F dạng symbolic để tránh chạy quá lâu.")
        print("Kết quả F và Q vẫn được tính trong quá trình biến đổi Danilevski.")
        return

    left = fast_reduce_matrix(A * Q)
    right = fast_reduce_matrix(Q * F)
    difference = fast_reduce_matrix(left - right)

    print("\nKiểm tra tương đương:")
    print("  F = Q^(-1)*A*Q  <=>  A*Q = Q*F")
    print()
    print_symbolic_matrix(difference, "A*Q - Q*F = ")

    if difference == sp.zeros(A.rows):
        print(
            "\nKết luận kiểm tra: F = Q^(-1)*A*Q (đúng)."
        )
    else:
        print(
            "\nCảnh báo: phép kiểm tra đồng dạng chưa cho ma trận 0."
        )


def _assert_danilevsky_charpoly(A, label):
    variable = sp.Symbol("lambda")
    F, _Q, blocks = danilevsky_transform(
        A,
        show_steps=False
    )
    _parts, polynomial = characteristic_polynomial_from_blocks(
        F,
        blocks,
        variable
    )
    direct = A.charpoly(variable).as_expr()
    if sp.expand(polynomial - direct) != 0:
        raise AssertionError(f"{label}: đa thức đặc trưng không khớp.")
    return F, blocks, polynomial


def run_self_tests():
    print_section("SELF-TEST DANILEVSKI")

    A1 = sp.Matrix([
        [4, 1, 0],
        [0, 3, 1],
        [2, 0, 1],
    ])
    _assert_danilevsky_charpoly(A1, "Test 1 TH1")

    A2 = sp.Matrix([
        [1, 2, 3],
        [4, 5, 6],
        [1, 0, 0],
    ])
    _assert_danilevsky_charpoly(A2, "Test 2 TH2")

    A3 = sp.Matrix([
        [1, 2, 0],
        [0, 3, 4],
        [0, 0, 5],
    ])
    _F3, blocks3, _poly3 = _assert_danilevsky_charpoly(A3, "Test 3 TH3")
    if len(blocks3) <= 1:
        raise AssertionError("Test 3 TH3: phải tách khối.")

    A4 = sp.diag(1, 2, 3, 4)
    _F4, blocks4, _poly4 = _assert_danilevsky_charpoly(A4, "Test 4 nhiều khối")
    if len(blocks4) != 4:
        raise AssertionError("Test 4: ma trận chéo cấp 4 phải tách 4 khối 1x1.")

    A5 = sp.Matrix([
        [2, 1],
        [0, 2],
    ])
    _F5, _blocks5, polynomial5 = _assert_danilevsky_charpoly(A5, "Test 5 nghiệm bội")
    variable = sp.Symbol("lambda")
    root_data5, _mode5 = solve_characteristic_polynomial(
        polynomial5,
        variable,
        decimals=7
    )
    root_2 = next(
        item for item in root_data5
        if sp.simplify(item["value"] - 2) == 0
    )
    vectors5 = direct_exact_eigenvectors(
        A5,
        sp.Integer(2)
    )
    if root_2["multiplicity"] != 2 or len(vectors5) != 1:
        raise AssertionError("Test 5: bội đại số/hình học của nghiệm bội sai.")

    A6 = sp.Matrix([
        [0, -1],
        [1, 0],
    ])
    _F6, _blocks6, polynomial6 = _assert_danilevsky_charpoly(A6, "Test 6 nghiệm phức")
    root_data6, _mode6 = solve_characteristic_polynomial(
        polynomial6,
        variable,
        decimals=7
    )
    if not any(abs(complex(sp.N(item["value"], 20)).imag) > 0.5 for item in root_data6):
        raise AssertionError("Test 6: phải có nghiệm phức.")

    try:
        danilevsky_transform(
            sp.Matrix([
                [1, 2, 3],
                [4, 5, 6],
            ]),
            show_steps=False
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Test 7: ma trận không vuông phải bị từ chối.")

    print("Đã qua 7 self-test bắt buộc.")


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main():
    print("=" * 92)
    print("PHƯƠNG PHÁP DANILEVSKI TÌM TRỊ RIÊNG VÀ VECTOR RIÊNG")
    print("=" * 92)

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

    if n > 5:
        print(
            "\nCảnh báo: Danilevski symbolic exact với n > 5 có thể chạy lâu."
        )
        continue_large = input_zero_or_one(
            "Bạn có muốn tiếp tục không? 1 = có, 0 = không [Enter = 0]: ",
            default=0,
        )
        if continue_large == 0:
            print("Dừng theo lựa chọn của người dùng.")
            return

    decimals = input_nonnegative_integer(
        "\nNhập số chữ số sau dấu phẩy muốn hiển thị [Enter = 7]: ",
        default=7,
    )

    show_steps = bool(input_zero_or_one(
        "In chi tiết từng bước biến đổi? 1 = có, 0 = không [Enter = 0]: ",
        default=0,
    ))

    find_vectors = bool(input_zero_or_one(
        "Tìm vector riêng? 1 = có, 0 = không [Enter = 0]: ",
        default=0,
    ))

    exact_check = bool(input_zero_or_one(
        "Bật kiểm tra symbolic nặng? 1 = có, 0 = không [Enter = 0]: ",
        default=0,
    ))

    print("\nMa trận đầu vào:")
    print_symbolic_matrix(A, "A =")

    if not show_steps:
        print_method_formula()
        print("\nĐang biến đổi Danilevski, vui lòng chờ...")

    F, Q, blocks = danilevsky_transform(
        A,
        show_steps=show_steps
    )

    if not show_steps:
        print_block_summary(blocks)

    print_transformation_check(
        A,
        F,
        Q,
        exact_check=exact_check
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
        variable,
        exact_check=exact_check
    )

    root_data = print_eigenvalues(
        characteristic_polynomial,
        variable,
        decimals
    )

    conclusion_vectors = []

    if find_vectors:
        conclusion_vectors = print_eigenvectors(
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
    print("\nĐã xử lý TH1/TH2/TH3.")
    if len(blocks) == 1:
        print("F là ma trận Frobenius.")
    else:
        print(f"Có {len(blocks)} khối.")
        print(
            "P_A(λ) = "
            + " · ".join(
                f"P{sub(index)}(λ)"
                for index in range(1, len(blocks) + 1)
            )
        )
    print(f"P_A(λ) = {expression_string(sp.factor(characteristic_polynomial))}")

    for index, item in enumerate(root_data, start=1):
        if item["exact"]:
            exact_text = expression_string(item["value"])
            approximate = format_complex(
                complex(sp.N(item["value"], max(20, decimals + 8))),
                decimals,
            )
            value_text = f"{exact_text} ≈ {approximate}"
        else:
            value_text = format_complex(item["value"], decimals)
        print(f"{lambda_symbol(index)} = {value_text}")

    if find_vectors:
        for index, item in enumerate(conclusion_vectors, start=1):
            vector = item["vector"]
            if item["exact"]:
                print_vector_symbolic(vector, f"X{sub(index)}")
            else:
                print_vector_numeric(vector, f"X{sub(index)}", decimals)
    else:
        print("Không tìm vector riêng.")


if __name__ == "__main__":
    try:
        if "--self-test" in sys.argv:
            run_self_tests()
        else:
            main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except Exception as error:
        print(f"\nLỗi trong quá trình tính toán: {error}")
