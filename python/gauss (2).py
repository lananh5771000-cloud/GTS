import sys
from exam_format import exam_print as print
from fractions import Fraction
from input_utils import MathInputError, parse_exact, split_number_row


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def display_number(value, decimals):
    """Chỉ đổi Fraction sang float tại thời điểm định dạng để hiển thị."""
    if value == 0:
        value = Fraction(0)
    return f"{float(value):8.{decimals}f}"


def print_matrix(matrix, m, n, k, decimals, prefix=""):
    padding = " " * len(prefix)
    mid_row = m // 2
    for i, row in enumerate(matrix):
        A_part = "  ".join(display_number(val, decimals) for val in row[:n])
        B_part = "  ".join(display_number(val, decimals) for val in row[n : n + k])
        row_str = f"[{A_part}  |  {B_part}]"
        print(f"{prefix if i == mid_row else padding}{row_str}")


def format_matrix_to_lines(mat, decimals):
    """In nhiều ma trận nằm ngang hàng nhau trong phần kết luận nghiệm."""
    return [
        "[" + "  ".join(display_number(val, decimals) for val in row) + "]"
        for row in mat
    ]


def input_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt).strip())
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Lỗi: Vui lòng nhập một số nguyên dương.")


def input_decimals(prompt, default=7):
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "":
                return default
            value = int(raw)
            if value < 0:
                raise ValueError
            return value
        except ValueError:
            print("Lỗi: Số chữ số sau dấu phẩy phải là số nguyên không âm.")


def input_matrix_row(prompt, expected_count):
    while True:
        try:
            tokens = split_number_row(input(prompt), expected_count)
            return [parse_exact(token) for token in tokens]
        except (MathInputError, ValueError, ZeroDivisionError):
            print(
                "Lỗi: Chỉ nhập số nguyên, số thập phân hoặc phân số hợp lệ "
                "(ví dụ 2, 0.25, 1/3)."
            )


def input_matrix(name, rows, columns):
    print(f"\nNhập ma trận {name}:")
    return [
        input_matrix_row(f"Nhập dòng {i + 1} (cách nhau bởi khoảng trắng): ", columns)
        for i in range(rows)
    ]


def choose_pivot_row(matrix, candidates, column):
    """Ưu tiên pivot bằng 1 hoặc -1; nếu không có thì lấy trị tuyệt đối lớn nhất."""
    unit_candidates = [r for r in candidates if abs(matrix[r][column]) == 1]
    if unit_candidates:
        return unit_candidates[0]
    return max(candidates, key=lambda r: abs(matrix[r][column]))


def matrix_rank(matrix):
    """Tính hạng bằng khử Gauss chính xác trên một bản sao."""
    if not matrix:
        return 0
    work = [row[:] for row in matrix]
    rows, columns = len(work), len(work[0])
    pivot_row = 0
    for column in range(columns):
        candidates = [r for r in range(pivot_row, rows) if work[r][column] != 0]
        if not candidates:
            continue
        chosen = choose_pivot_row(work, candidates, column)
        work[pivot_row], work[chosen] = work[chosen], work[pivot_row]
        for r in range(pivot_row + 1, rows):
            if work[r][column] != 0:
                factor = work[r][column] / work[pivot_row][column]
                for j in range(column, columns):
                    work[r][j] -= factor * work[pivot_row][j]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def multiply_matrices(left, right):
    rows, middle, columns = len(left), len(right), len(right[0])
    return [
        [
            sum((left[i][p] * right[p][j] for p in range(middle)), Fraction(0))
            for j in range(columns)
        ]
        for i in range(rows)
    ]


def gauss_jordan(A, B, decimals):
    """Đưa phần S=[A|B] về dạng bậc thang rút gọn theo kiểu PDF."""
    m, n, k = len(A), len(A[0]), len(B[0])
    aug = [A[i][:] + B[i][:] for i in range(m)]
    pivot_pos = {}
    pivot_values = [None] * m
    row = [0] * m
    col = [0] * n
    idx = [0] * m
    pivot_row = 0
    step = 1

    while pivot_row < m:
        chosen = None
        max_abs = 0
        max_pos = None

        for column in range(n):
            if col[column] == 1:
                continue
            for i in range(m):
                if row[i] == 1:
                    continue
                value = aug[i][column]
                if value == 0:
                    continue
                if abs(value) == 1:
                    chosen = (i, column)
                    break
                if abs(value) > max_abs:
                    max_abs = abs(value)
                    max_pos = (i, column)
            if chosen is not None:
                break

        if chosen is None:
            chosen = max_pos
        if chosen is None:
            break

        i0, j0 = chosen
        print(f"\nBước {step}: Chọn pivot s[{i0 + 1},{j0 + 1}] theo quy tắc PDF.")
        if i0 != pivot_row:
            aug[pivot_row], aug[i0] = aug[i0], aug[pivot_row]
            row[pivot_row], row[i0] = row[i0], row[pivot_row]
            print(f"R{pivot_row + 1} <-> R{i0 + 1}")
            i0 = pivot_row

        row[pivot_row] = 1
        row[i0] = 1
        col[j0] = 1
        idx[pivot_row] = j0 + 1
        pivot_value = aug[pivot_row][j0]
        pivot_values[pivot_row] = pivot_value

        for r in range(m):
            if r == pivot_row or aug[r][j0] == 0:
                continue
            factor = aug[r][j0]
            print(f"R{r + 1} <- R{r + 1} - ({factor})*R{pivot_row + 1}")
            aug[r] = [aug[r][j] - factor * aug[pivot_row][j] for j in range(n + k)]

        pivot_pos[j0] = pivot_row
        print_matrix(aug, m, n, k, decimals, prefix=f"S^({step}) = ")
        pivot_row += 1
        step += 1

    for i in range(m):
        if idx[i] > 0 and pivot_values[i] not in {None, 0} and pivot_values[i] != 1:
            print(f"Chuẩn hóa hàng mốc R{i + 1} / ({pivot_values[i]}) sau khi khử xong.")
            aug[i] = [value / pivot_values[i] for value in aug[i]]

    print(f"idx = {idx}")
    return aug, pivot_pos, step - 1


def print_solution(aug, pivot_pos, n, k, decimals):
    free_cols = [column for column in range(n) if column not in pivot_pos]
    X0 = [[Fraction(0) for _ in range(k)] for _ in range(n)]
    for column, row in pivot_pos.items():
        for j in range(k):
            X0[column][j] = aug[row][n + j]

    if not free_cols:
        print("Hệ có nghiệm duy nhất X:\n")
        lines = format_matrix_to_lines(X0, decimals)
        for i, line in enumerate(lines):
            print(("X = " if i == n // 2 else "    ") + line)
        return X0

    print(f"Hệ có VÔ SỐ NGHIỆM ({len(free_cols)} biến tự do).\n")
    vectors = []
    for free in free_cols:
        vector = [[Fraction(0)] for _ in range(n)]
        vector[free][0] = Fraction(1)
        for column, row in pivot_pos.items():
            vector[column][0] = -aug[row][free]
        vectors.append(vector)

    V_names = [f"V_{free + 1}" for free in free_cols]
    T_names = [f"T_{free + 1}" for free in free_cols]
    x_names = [f"x_{free + 1}" for free in free_cols]
    print(
        f"Trong đó, X_0 là ma trận nghiệm riêng (khi cho các biến tự do bằng 0), "
        f"{', '.join(V_names)} là các vector cơ sở tương ứng với {', '.join(x_names)}, "
        f"và {', '.join(T_names)} là các ma trận dòng chứa tham số tự do cho {k} cột.\n"
    )

    X0_lines = format_matrix_to_lines(X0, decimals)
    vector_lines = [format_matrix_to_lines(vector, decimals) for vector in vectors]
    parameter_rows = [
        "[" + "  ".join(f"t_{free + 1},{j + 1}" for j in range(k)) + "]"
        for free in free_cols
    ]
    for i in range(n):
        middle = i == n // 2
        line = ("X = " if middle else "    ") + X0_lines[i]
        for index in range(len(free_cols)):
            line += (" + " if middle else "   ") + vector_lines[index][i]
            line += " " + (
                parameter_rows[index] if middle else " " * len(parameter_rows[index])
            )
        print(line)

    names = [f"t_{free + 1},{j + 1}" for free in free_cols for j in range(k)]
    print(f"\n(với {', '.join(names)} ∈ ℝ là các tham số).\n")
    print(
        "Lưu ý: Vì hệ vô số nghiệm nên nghiệm tổng quát sẽ thay đổi "
        "tùy thuộc vào biến tự do bạn chọn."
    )
    return X0, vectors


def process(A, B, decimals, inverse_mode=False):
    m, n, k = len(A), len(A[0]), len(B[0])
    A_original = [row[:] for row in A]
    initial_aug = [A[i][:] + B[i][:] for i in range(m)]

    print("\n" + "=" * 80)
    print("Ta sử dụng phương pháp Gauss - Jordan giải hệ phương trình AX = B:")
    print("Khởi tạo ma trận mở rộng S = [A|B]\n")
    print_matrix(initial_aug, m, n, k, decimals, prefix="S = ")

    aug, pivot_pos, last_step = gauss_jordan(A, B, decimals)
    rank_A = matrix_rank(A_original)
    rank_aug = matrix_rank(initial_aug)

    print("\n" + "=" * 80)
    print("Kết quả\n")
    print_matrix(aug, m, n, k, decimals, prefix=f"S^({last_step}) = ")
    print(f"\nHạng của A: {rank_A}")
    print(f"Hạng của ma trận bổ sung [A|B]: {rank_aug}\n")

    inconsistent = any(
        all(aug[i][j] == 0 for j in range(n))
        and any(aug[i][n + j] != 0 for j in range(k))
        for i in range(m)
    )
    if inverse_mode:
        if len(pivot_pos) != n or inconsistent:
            print("Ma trận A không khả nghịch.")
            return None
        inverse = [[aug[i][n + j] for j in range(n)] for i in range(n)]
        print("Ma trận A khả nghịch và A^(-1) là:\n")
        for line in format_matrix_to_lines(inverse, decimals):
            print(line)
        print("\nKiểm tra A*A^(-1):")
        for line in format_matrix_to_lines(
            multiply_matrices(A_original, inverse), decimals
        ):
            print(line)
        return inverse

    if inconsistent or rank_A < rank_aug:
        print("Hệ phương trình VÔ NGHIỆM.")
        return None
    return print_solution(aug, pivot_pos, n, k, decimals)


def solve_system():
    m = input_positive_integer("Nhập số dòng của ma trận A (m): ")
    n = input_positive_integer("Nhập số cột của ma trận A (n - số biến): ")
    A = input_matrix("A", m, n)
    k = input_positive_integer("\nNhập số cột của ma trận B (k): ")
    B = input_matrix("B", m, k)
    decimals = input_decimals("\nSố chữ số sau dấu phẩy [Enter = 7]: ")
    process(A, B, decimals)


def find_inverse():
    m = input_positive_integer("Nhập số dòng của ma trận A (m): ")
    n = input_positive_integer("Nhập số cột của ma trận A (n): ")
    A = input_matrix("A", m, n)
    decimals = input_decimals("\nSố chữ số sau dấu phẩy [Enter = 7]: ")
    if m != n:
        print("\nMa trận không vuông nên không có ma trận nghịch đảo thông thường.")
        return
    identity = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    process(A, identity, decimals, inverse_mode=True)


def solve_gauss_jordan():
    print("=" * 88)
    print("GAUSS–JORDAN – BÀI GIẢI CHI TIẾT")
    print("Input: ma trận A, vế phải B (hoặc I khi tìm nghịch đảo).")
    print("Output: dạng bậc thang rút gọn, nghiệm/họ nghiệm và phép kiểm tra.")
    print("Thuật toán: chọn pivot, đổi hàng nếu cần, chuẩn hóa pivot và khử toàn bộ cột pivot.")
    print("=" * 88)
    print("1. Giải hệ phương trình AX = B bằng phương pháp Gauss - Jordan")
    print("2. Tìm ma trận nghịch đảo A^(-1) bằng phương pháp Gauss - Jordan")
    print("0. Thoát")
    while True:
        choice = input("Chọn [Enter = 1]: ").strip() or "1"
        if choice == "0":
            return
        if choice == "1":
            solve_system()
            return
        if choice == "2":
            find_inverse()
            return
        print("Lỗi: Vui lòng chỉ chọn 1 hoặc 2.")


if __name__ == "__main__":
    try:
        solve_gauss_jordan()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except Exception as error:
        print(f"\nKhông thể thực hiện: {error}")
