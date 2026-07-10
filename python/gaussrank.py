import sys
from exam_format import exam_print as print

import sympy as sp
from input_utils import MathInputError, parse_exact, split_number_row


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def input_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt).strip())
            if value > 0:
                return value
        except ValueError:
            pass
        print("Lỗi: Vui lòng nhập một số nguyên dương.")


def input_nonnegative_integer(prompt, default=None):
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "" and default is not None:
                return default
            value = int(raw)
            if value >= 0:
                return value
        except ValueError:
            pass
        print("Lỗi: Vui lòng nhập một số nguyên không âm.")


def input_matrix_row(prompt, size):
    while True:
        try:
            tokens = split_number_row(input(prompt), size)
            return [parse_exact(token) for token in tokens]
        except (MathInputError, ValueError, TypeError, ZeroDivisionError):
            print("Lỗi: Hãy nhập số nguyên, số thập phân hoặc phân số (ví dụ: 2, 0.25, 1/3).")


def input_matrix(name, rows, columns):
    print(f"\nNhập ma trận {name} ({rows} x {columns}), các phần tử cách nhau bởi dấu cách:")
    return sp.Matrix([input_matrix_row(f"  Hàng {i + 1}: ", columns) for i in range(rows)])


def fmt(value, digits):
    """Định dạng số để trình bày; mọi phép tính bên trong vẫn hoàn toàn chính xác."""
    value = sp.N(value, max(15, digits + 8))
    number = float(value)
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:.{digits}f}"


def print_matrix(matrix, digits, title=None):
    if title:
        print(title)
    strings = [[fmt(matrix[i, j], digits) for j in range(matrix.cols)] for i in range(matrix.rows)]
    widths = [max(len(strings[i][j]) for i in range(matrix.rows)) for j in range(matrix.cols)]
    for row in strings:
        print("  [ " + "  ".join(value.rjust(widths[j]) for j, value in enumerate(row)) + " ]")


def rank_exact(matrix):
    return matrix.rank()


def forward_elimination(A, B, digits):
    aug = A.row_join(B)
    rows, variables = A.shape
    pivot_row = 0
    pivot_columns = []
    idx = [0] * rows
    step = 1

    print("\n========== QUÁ TRÌNH KHỬ THUẬN ==========")
    for column in range(variables):
        candidates = [r for r in range(pivot_row, rows) if aug[r, column] != 0]
        if not candidates:
            print(f"\nCột {column + 1} không có phần tử trụ, chuyển sang cột kế tiếp.")
            continue
        chosen = candidates[0]
        if chosen != pivot_row:
            aug.row_swap(chosen, pivot_row)
            print(f"\nBước {step}: R{pivot_row + 1} ↔ R{chosen + 1}")
            print_matrix(aug, digits)
            step += 1

        pivot_columns.append(column)
        idx[pivot_row] = column + 1
        print(f"\nChọn phần tử trụ a[{pivot_row + 1},{column + 1}] = {fmt(aug[pivot_row, column], digits)}.")
        for r in range(pivot_row + 1, rows):
            if aug[r, column] == 0:
                continue
            multiplier = sp.cancel(aug[r, column] / aug[pivot_row, column])
            aug[r, :] = aug[r, :] - multiplier * aug[pivot_row, :]
            print(f"\nBước {step}: R{r + 1} ← R{r + 1} - ({fmt(multiplier, digits)})R{pivot_row + 1}")
            print_matrix(aug, digits)
            step += 1
        pivot_row += 1
        if pivot_row == rows:
            break
    if step == 1:
        print("Không cần thực hiện phép biến đổi hàng nào.")
    print(f"\nidx (vị trí pivot theo hàng, 1-based) = {idx}")
    return aug, pivot_columns, idx


def back_substitute(echelon, pivot_columns, variables, rhs_columns, digits, show=True):
    X = sp.zeros(variables, rhs_columns)
    if show:
        print("\n========== QUÁ TRÌNH THẾ NGƯỢC ==========")
    for rhs in range(rhs_columns):
        if show:
            print(f"\n--- Giải cột vế phải thứ {rhs + 1} ---")
        for row in range(len(pivot_columns) - 1, -1, -1):
            col = pivot_columns[row]
            terms = [echelon[row, j] * X[j, rhs] for j in range(col + 1, variables)]
            known_sum = sum(terms, sp.S.Zero)
            numerator = echelon[row, variables + rhs] - known_sum
            X[col, rhs] = sp.cancel(numerator / echelon[row, col])
            if show:
                pieces = " + ".join(
                    f"({fmt(echelon[row, j], digits)})({fmt(X[j, rhs], digits)})"
                    for j in range(col + 1, variables) if echelon[row, j] != 0
                ) or "0"
                print(
                    f"x{col + 1} = [{fmt(echelon[row, variables + rhs], digits)} - ({pieces})] "
                    f"/ {fmt(echelon[row, col], digits)} = {fmt(X[col, rhs], digits)}"
                )
    return X


def homogeneous_vector(echelon, pivots, variables, free):
    vector = sp.zeros(variables, 1)
    vector[free] = 1
    for row in range(len(pivots) - 1, -1, -1):
        col = pivots[row]
        total = sum((echelon[row, j] * vector[j] for j in range(col + 1, variables)), sp.S.Zero)
        vector[col] = sp.cancel(-total / echelon[row, col])
    return vector


def solve_from_matrices(A, B, digits, inverse_mode=False):
    original_A, original_B = A.copy(), B.copy()
    variables, rhs_columns = A.cols, B.cols
    print_matrix(A.row_join(B), digits, "\nMa trận bổ sung ban đầu [A | B]:")
    echelon, pivots, idx = forward_elimination(A, B, digits)
    print_matrix(echelon, digits, "\n========== MA TRẬN BẬC THANG ==========")

    rank_A = rank_exact(original_A)
    rank_aug = rank_exact(original_A.row_join(original_B))
    print(f"\nrank(A) = {rank_A}; rank([A | B]) = {rank_aug}; số ẩn = {variables}.")
    print(f"idx = {idx}")
    if rank_A < rank_aug:
        print("\nKẾT LUẬN: " + ("A không khả nghịch." if inverse_mode else "Hệ phương trình vô nghiệm."))
        return None
    if inverse_mode and rank_A < variables:
        print("\nKẾT LUẬN: A suy biến nên không có ma trận nghịch đảo.")
        return None

    if rank_A == variables:
        X = back_substitute(echelon, pivots, variables, rhs_columns, digits)
        title = "\nKẾT QUẢ: A⁻¹ = X:" if inverse_mode else "\nKẾT QUẢ: Ma trận nghiệm X:"
        print_matrix(X, digits, title)
        if inverse_mode:
            print_matrix(original_A * X, digits, "\nKiểm tra A·A⁻¹ = I:")
        return X

    free = [j for j in range(variables) if j not in pivots]
    print(f"\nHệ có vô số nghiệm với {len(free)} biến tự do: " + ", ".join(f"x{j + 1}" for j in free) + ".")
    X0 = back_substitute(echelon, pivots, variables, rhs_columns, digits)
    print_matrix(X0, digits, "\nNghiệm riêng X₀ (cho các biến tự do bằng 0):")
    print("\nNghiệm tổng quát: X = X₀ + Σ(VᵢTᵢ), trong đó:")
    for free_col in free:
        vector = homogeneous_vector(echelon, pivots, variables, free_col)
        print_matrix(vector, digits, f"  Vector V ứng với x{free_col + 1}:")
    print("Các phần tử của Tᵢ là tham số thực tùy ý (mỗi cột vế phải có một tham số riêng).")
    return X0


def solve_system(digits):
    print("\n--- GIẢI HỆ AX = B BẰNG PHƯƠNG PHÁP GAUSS ---")
    m = input_positive_integer("Nhập số hàng m: ")
    n = input_positive_integer("Nhập số cột của A (số ẩn n): ")
    A = input_matrix("A", m, n)
    k = input_positive_integer("Nhập số cột của B (k): ")
    B = input_matrix("B", m, k)
    solve_from_matrices(A, B, digits)


def find_inverse(digits):
    print("\n--- TÌM MA TRẬN NGHỊCH ĐẢO BẰNG GAUSS ---")
    n = input_positive_integer("Nhập cấp của ma trận vuông A (n): ")
    A = input_matrix("A", n, n)
    solve_from_matrices(A, sp.eye(n), digits, inverse_mode=True)


def main():
    print("========== PHƯƠNG PHÁP GAUSS ==========")
    print("Input: A, B. Output: rank(A), rank(A|B), nghiệm duy nhất/họ nghiệm/vô nghiệm.")
    print("Các phép biến đổi sơ cấp và vị trí pivot được in đầy đủ để chép bài.")
    digits = input_nonnegative_integer(
        "Số chữ số sau dấu phẩy [Enter = 7]: ", default=7
    )
    print("\n1. Giải hệ phương trình AX = B")
    print("2. Tìm ma trận nghịch đảo A⁻¹")
    print("0. Thoát")
    while True:
        choice = input("Chọn [Enter = 1]: ").strip() or "1"
        if choice == "0":
            return
        if choice == "1":
            solve_system(digits)
            return
        if choice == "2":
            find_inverse(digits)
            return
        print("Lỗi: Vui lòng chỉ chọn 1 hoặc 2.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except Exception as error:
        print(f"\nKhông thể thực hiện: {error}")
