"""Giải hệ phương trình phi tuyến F(x)=0 bằng phương pháp Newton."""

from __future__ import annotations

import math
from exam_format import exam_print as print
import sys
from dataclasses import dataclass
from input_utils import MathInputError, parse_math_expression, parse_real, split_number_row

import numpy as np
import sympy as sp


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")


LINE = "=" * 100
SAFE_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}


def parse_number(text: str) -> float:
    text = text.strip().replace("−", "-")
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    value = parse_real(text)
    if not math.isfinite(value):
        raise ValueError
    return value


def read_int(prompt: str, minimum: int = 0, default: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            value = int(raw)
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            print(f"  Lỗi: hãy nhập số nguyên >= {minimum}.")


def read_positive(prompt: str, default: float | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            value = parse_number(raw)
            if value <= 0.0:
                raise ValueError
            return value
        except (ValueError, ZeroDivisionError):
            print("  Lỗi: hãy nhập số dương hữu hạn.")


def read_vector(prompt: str, size: int) -> np.ndarray:
    while True:
        try:
            parts = split_number_row(input(prompt), size)
            return np.array([parse_number(part) for part in parts], dtype=float)
        except (MathInputError, ValueError, ZeroDivisionError):
            print("  Lỗi: vector chứa dữ liệu không hợp lệ.")


def clean(value: float, decimals: int) -> float:
    threshold = 0.5 * 10.0 ** (-decimals) if decimals else 0.5
    return 0.0 if abs(value) < threshold else float(value)


def fmt(value: float, decimals: int) -> str:
    return f"{clean(value, decimals):.{decimals}f}"


def print_matrix(name: str, matrix: np.ndarray, decimals: int) -> None:
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    cells = [[fmt(value, decimals) for value in row] for row in matrix]
    widths = [max(len(cells[i][j]) for i in range(len(cells))) for j in range(len(cells[0]))]
    middle = len(cells) // 2
    padding = " " * (len(name) + 3)
    for i, row in enumerate(cells):
        body = "  ".join(value.rjust(widths[j]) for j, value in enumerate(row))
        if len(cells) == 1:
            left, right = "[", "]"
        elif i == 0:
            left, right = "⎡", "⎤"
        elif i == len(cells) - 1:
            left, right = "⎣", "⎦"
        else:
            left, right = "⎢", "⎥"
        print((f"{name} = " if i == middle else padding) + f"{left} {body} {right}")


def print_vector(name: str, vector: np.ndarray, decimals: int) -> None:
    text = "  ".join(fmt(value, decimals) for value in np.asarray(vector).reshape(-1))
    print(f"{name} = [{text}]^T")


def section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def gaussian_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Giải hệ vuông bằng khử Gauss có pivot từng phần, không gọi solve/inv."""
    A = np.asarray(matrix, dtype=float).copy()
    b = np.asarray(rhs, dtype=float).copy()
    n = len(b)
    scale = max(1.0, float(np.linalg.norm(A, np.inf)))
    tolerance = 100.0 * np.finfo(float).eps * scale

    for column in range(n):
        pivot = column + int(np.argmax(np.abs(A[column:, column])))
        if abs(A[pivot, column]) <= tolerance:
            raise ArithmeticError(
                f"Jacobian suy biến hoặc gần suy biến tại cột {column + 1}."
            )
        if pivot != column:
            A[[column, pivot]] = A[[pivot, column]]
            b[[column, pivot]] = b[[pivot, column]]
        for row in range(column + 1, n):
            factor = A[row, column] / A[column, column]
            A[row, column:] -= factor * A[column, column:]
            b[row] -= factor * b[column]

    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        x[row] = (b[row] - A[row, row + 1 :] @ x[row + 1 :]) / A[row, row]
    return x


def evaluate_system(
    expressions: list[sp.Expr],
    jacobian: sp.Matrix,
    variables: list[sp.Symbol],
    point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    substitutions = {variable: float(value) for variable, value in zip(variables, point)}
    try:
        F = np.array([float(expression.evalf(subs=substitutions)) for expression in expressions])
        J = np.array(jacobian.evalf(subs=substitutions), dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArithmeticError(f"không đánh giá được F hoặc Jacobian: {error}") from error
    if not np.all(np.isfinite(F)) or not np.all(np.isfinite(J)):
        raise ArithmeticError("F hoặc Jacobian sinh NaN/vô cùng")
    return F, J


def evaluate_functions(
    expressions: list[sp.Expr], variables: list[sp.Symbol], point: np.ndarray
) -> np.ndarray:
    substitutions = {variable: float(value) for variable, value in zip(variables, point)}
    try:
        evaluated = [expression.subs(substitutions) for expression in expressions]
        if any(item.has(sp.zoo, sp.oo, -sp.oo, sp.nan) for item in evaluated):
            raise ArithmeticError("F không xác định hoặc vô hạn tại điểm đang xét")
        values = np.array([float(sp.N(item, 17)) for item in evaluated], dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArithmeticError(f"không đánh giá được F: {error}") from error
    if not np.all(np.isfinite(values)):
        raise ArithmeticError("F sinh NaN hoặc vô cùng")
    return values


def evaluate_jacobian(
    jacobian: sp.Matrix, variables: list[sp.Symbol], point: np.ndarray
) -> np.ndarray:
    substitutions = {variable: float(value) for variable, value in zip(variables, point)}
    try:
        evaluated = jacobian.subs(substitutions)
        if any(item.has(sp.zoo, sp.oo, -sp.oo, sp.nan) for item in evaluated):
            raise ArithmeticError("Jacobian không xác định hoặc vô hạn tại điểm đang xét")
        matrix = np.array(sp.N(evaluated, 17), dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArithmeticError(f"không đánh giá được Jacobian: {error}") from error
    if not np.all(np.isfinite(matrix)):
        raise ArithmeticError("Jacobian sinh NaN hoặc vô cùng")
    return matrix


@dataclass
class Record:
    k: int
    x_old: np.ndarray
    F_old: np.ndarray
    J_old: np.ndarray
    delta: np.ndarray
    x_new: np.ndarray
    step_norm: float
    residual_norm: float


@dataclass
class Result:
    x: np.ndarray
    records: list[Record]
    converged: bool
    reason: str
    status: str = "unknown"
    residual_norm: float = math.inf


def newton_system(
    expressions: list[sp.Expr],
    variables: list[sp.Symbol],
    initial: np.ndarray,
    epsilon: float,
    maximum_iterations: int,
    fixed_iterations: int = 0,
    maximum_step_norm: float | None = None,
    method: str = "classic",
) -> Result:
    if len(expressions) == 0 or len(expressions) != len(variables):
        raise ValueError("Số phương trình phải bằng số ẩn và khác 0.")
    if epsilon <= 0 or maximum_iterations <= 0 or fixed_iterations < 0:
        raise ValueError("epsilon, max_iter phải dương; fixed_iterations không âm.")
    if maximum_step_norm is not None and (
        not math.isfinite(maximum_step_norm) or maximum_step_norm <= 0
    ):
        raise ValueError("maximum_step_norm phải dương và hữu hạn.")
    method = method.strip().lower()
    if method not in {"classic", "modified"}:
        raise ValueError("method chi nhan 'classic' hoac 'modified'.")
    jacobian = sp.Matrix(expressions).jacobian(variables)
    x = np.asarray(initial, dtype=float).copy()
    if x.shape != (len(variables),) or not np.all(np.isfinite(x)):
        raise ValueError("Vector đầu sai kích thước hoặc chứa NaN/vô cùng.")
    fixed_jacobian = None
    if method == "modified":
        try:
            fixed_jacobian = evaluate_jacobian(jacobian, variables, x)
        except ArithmeticError as error:
            return Result(
                x,
                [],
                False,
                f"Modified Newton that bai vi J0 khong hop le: {error}",
                "singular_jacobian",
                math.inf,
            )
    records: list[Record] = []
    limit = fixed_iterations if fixed_iterations > 0 else maximum_iterations
    converged = False
    reason = f"đã thực hiện đúng k={fixed_iterations} bước"
    status = "fixed_steps"
    residual_norm = math.inf

    for k in range(limit):
        # Thứ tự bắt buộc: F -> hữu hạn -> phần dư -> rồi mới Jacobian.
        try:
            F_old = evaluate_functions(expressions, variables, x)
        except ArithmeticError as error:
            return Result(x, records, False, str(error), "numerical_failure", math.inf)
        residual_old = float(np.linalg.norm(F_old, np.inf))
        if residual_old <= epsilon and fixed_iterations == 0:
            reason = (
                "Nghiệm ban đầu đã thỏa hệ."
                if k == 0
                else f"||F(x^{k})||_∞ <= epsilon={epsilon:.3e}."
            )
            return Result(x, records, True, reason, "converged", residual_old)
        try:
            J_old = fixed_jacobian.copy() if fixed_jacobian is not None else evaluate_jacobian(jacobian, variables, x)
            delta = gaussian_solve(J_old, -F_old)
        except ArithmeticError as error:
            label = "J0" if fixed_jacobian is not None else "Jacobian"
            return Result(
                x,
                records,
                False,
                f"Newton thất bại do {label} suy biến tại điểm chưa phải nghiệm: {error}",
                "singular_jacobian",
                residual_old,
            )
        if not np.all(np.isfinite(delta)):
            return Result(x, records, False, "Bước Newton sinh NaN/vô cùng.", "numerical_failure", residual_old)
        step_norm = float(np.linalg.norm(delta, np.inf))
        if maximum_step_norm is not None and step_norm > maximum_step_norm:
            return Result(
                x, records, False,
                f"Bước Newton quá lớn: ||Delta||_∞={step_norm:.3e}>{maximum_step_norm:.3e}.",
                "step_too_large", residual_old,
            )
        x_new = x + delta
        try:
            F_new = evaluate_functions(expressions, variables, x_new)
        except ArithmeticError as error:
            return Result(x, records, False, str(error), "numerical_failure", residual_old)
        residual_norm = float(np.linalg.norm(F_new, np.inf))
        records.append(Record(k, x.copy(), F_old, J_old, delta, x_new.copy(), step_norm, residual_norm))
        x = x_new
        if residual_norm <= epsilon and fixed_iterations == 0:
            converged = True
            status = "converged"
            reason = f"||F(x)||_∞ <= epsilon={epsilon:.3e}"
            break

    if fixed_iterations > 0:
        converged = bool(records and records[-1].residual_norm <= epsilon)
        status = "fixed_steps"
        reason = (
            f"đã thực hiện đúng k={fixed_iterations} bước; residual cuối đạt epsilon"
            if converged
            else f"đã thực hiện đúng k={fixed_iterations} bước; residual cuối chưa đạt epsilon"
        )
    elif not converged:
        status = "max_iter_reached"
        reason = f"đạt k_max={maximum_iterations} nhưng chưa thỏa điều kiện dừng"
    return Result(x, records, converged, reason, status, residual_norm)


def modified_newton_system(
    expressions: list[sp.Expr],
    variables: list[sp.Symbol],
    initial: np.ndarray,
    epsilon: float,
    maximum_iterations: int,
    fixed_iterations: int = 0,
    maximum_step_norm: float | None = None,
) -> Result:
    return newton_system(
        expressions,
        variables,
        initial,
        epsilon,
        maximum_iterations,
        fixed_iterations,
        maximum_step_norm,
        method="modified",
    )


def print_theory(method: str = "classic") -> None:
    section("A. THUẬT TOÁN NEWTON CHO HỆ PHƯƠNG TRÌNH PHI TUYẾN")
    method = method.strip().lower()
    if method == "modified":
        print("Nhánh dùng trong bài: Modified Newton.")
        print("Tính J0 = J(X0) một lần; nếu det(J0)=0 thì dừng.")
        print("Lặp X_(k+1)=X_k - J0^(-1)F(X_k).")
    else:
        print("Nhánh dùng trong bài: Newton cổ điển.")
        print("Mỗi vòng tính J(X_k); nếu det(J(X_k))=0 thì dừng.")
        print("Lặp X_(k+1)=X_k - J(X_k)^(-1)F(X_k).")
    print("Input:")
    print("  • Hệ F(x)=0 gồm n phương trình và điểm đầu x^(0).")
    print("  • Sai số epsilon hoặc số vòng lặp k.")
    print("Output: nghiệm gần đúng, Jacobian và hệ tuyến tính hóa ở từng bước.")
    print("Các bước:")
    print("  B1. Lập ma trận Jacobian J(x)=[∂f_i/∂x_j].")
    print("  B2. Tại x^(k), giải hệ tuyến tính")
    print("                   J(x^(k)) Delta^(k) = -F(x^(k)).")
    print("  B3. Cập nhật x^(k+1)=x^(k)+Delta^(k).")
    print("  B4. Kiểm tra ||Delta^(k)||_∞ và ||F(x^(k+1))||_∞.")
    print("Lưu ý: độ nhỏ của bước và phần dư là tiêu chuẩn dừng số;")
    print("không tự động là chặn sai số nghiệm nếu chưa có giả thiết Newton–Kantorovich.")


def print_records(records: list[Record], decimals: int) -> None:
    for record in records:
        section(f"BƯỚC NEWTON k={record.k} → k={record.k + 1}")
        print_vector(f"x^({record.k})", record.x_old, decimals)
        print_vector(f"F(x^({record.k}))", record.F_old, decimals)
        print_matrix(f"J(x^({record.k}))", record.J_old, decimals)
        augmented = np.column_stack((record.J_old, -record.F_old))
        print("\nHệ tuyến tính hóa [J | -F]:")
        print_matrix("[J|-F]", augmented, decimals)
        print_vector(f"Delta^({record.k})", record.delta, decimals)
        print(
            f"x^({record.k + 1})=x^({record.k})+Delta^({record.k})"
        )
        print_vector(f"x^({record.k + 1})", record.x_new, decimals)
        print(f"||Delta^({record.k})||_∞={record.step_norm:.7e}")
        print(f"||F(x^({record.k + 1}))||_∞={record.residual_norm:.7e}")

    if not records:
        return
    print("\nBảng tổng hợp:")
    n = records[0].x_new.size
    headers = ["k"] + [f"x{i + 1}^(k)" for i in range(n)] + ["||Delta||inf", "||F||inf"]
    rows = []
    for record in records:
        rows.append([
            str(record.k + 1),
            *[fmt(value, decimals) for value in record.x_new],
            f"{record.step_norm:.2e}",
            f"{record.residual_norm:.2e}",
        ])
    widths = [max(len(headers[j]), *(len(row[j]) for row in rows)) for j in range(len(headers))]
    print("  " + " | ".join(headers[j].rjust(widths[j]) for j in range(len(headers))))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print("  " + " | ".join(row[j].rjust(widths[j]) for j in range(len(row))))


def main() -> None:
    print(LINE)
    print("PHƯƠNG PHÁP NEWTON GIẢI HỆ PHƯƠNG TRÌNH PHI TUYẾN")
    print(LINE)
    print("Nhập 0 để thoát, dùng ** cho lũy thừa; hỗ trợ sin, cos, exp, log, sqrt.")
    n = read_int("Số phương trình/số ẩn n (0 để thoát): ", 0)
    if n == 0:
        return

    while True:
        raw_names = input(
            f"Nhập {n} tên biến cách nhau bằng dấu cách [Enter = x1 ... x{n}]: "
        ).strip()
        names = raw_names.split() if raw_names else [f"x{i + 1}" for i in range(n)]
        if len(names) == n and len(set(names)) == n and all(name.isidentifier() for name in names):
            break
        print("  Lỗi: tên biến phải hợp lệ, khác nhau và đủ số lượng.")
    variables = list(sp.symbols(" ".join(names), seq=True))
    expressions: list[sp.Expr] = []
    print("\nNhập vế trái f_i(x)=0:")
    for i in range(n):
        while True:
            text = input(f"  f_{i + 1} = ").strip().replace("^", "**")
            try:
                expression = parse_math_expression(text, dict(zip(names, variables)))
                if expression.free_symbols - set(variables):
                    raise ValueError("có biến chưa khai báo")
                expressions.append(expression)
                break
            except (sp.SympifyError, TypeError, ValueError) as error:
                print(f"  Lỗi biểu thức: {error}.")

    initial = read_vector(f"\nNhập x^(0) gồm {n} số: ", n)
    print("\nChọn nhánh Newton theo PDF:")
    print("1. Newton cổ điển")
    print("2. Modified Newton")
    method_choice = input("Chọn [Enter = 1]: ").strip() or "1"
    method = "modified" if method_choice == "2" else "classic"
    print("\nChế độ dừng:")
    print("1. Theo epsilon")
    print("2. Thực hiện đúng k bước")
    stop_choice = input("Chọn [Enter = 1]: ").strip() or "1"
    epsilon = read_positive("epsilon dùng để kiểm tra [Enter = 1e-8]: ", 1e-8)
    fixed_iterations = 0
    maximum_iterations = 100
    if stop_choice == "2":
        fixed_iterations = read_int("Số bước k: ", 1)
    decimals = read_int("Số chữ số sau dấu phẩy [Enter = 7]: ", 0, 7)

    jacobian = sp.Matrix(expressions).jacobian(variables)
    print_theory(method)
    section("B. DỮ KIỆN VÀ JACOBIAN")
    print("Hệ đã nhập:")
    for i, expression in enumerate(expressions, start=1):
        print(f"  f_{i}({', '.join(names)})={sp.sstr(expression)}=0")
    print("\nMa trận Jacobian ký hiệu:")
    sp.pprint(jacobian)
    print_vector("x^(0)", initial, decimals)

    try:
        result = newton_system(
            expressions,
            variables,
            initial,
            epsilon,
            maximum_iterations,
            fixed_iterations,
            method=method,
        )
    except ArithmeticError as error:
        print(f"\nDừng: {error}")
        return

    print_records(result.records, decimals)
    section("C. KẾT LUẬN")
    print(f"Dừng vì {result.reason}.")
    print_vector("x*", result.x, decimals)
    final_F, _ = evaluate_system(expressions, jacobian, variables, result.x)
    print_vector("F(x*)", final_F, decimals)
    print(f"||F(x*)||_∞={np.linalg.norm(final_F, np.inf):.7e}")
    if result.converged:
        print("KẾT LUẬN: nghiệm gần đúng thỏa tiêu chuẩn dừng đã nhập.")
    elif fixed_iterations:
        print("KẾT LUẬN: đây là giá trị gần đúng sau đúng số bước đề yêu cầu.")
    else:
        print("KẾT LUẬN: chưa xác nhận hội tụ; cần đổi x^(0) hoặc tăng k_max.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except Exception as error:
        print(f"\nLỗi trong quá trình tính toán: {error}")
