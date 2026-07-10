"""Giải hệ tuyến tính Ax=b bằng phương pháp lặp đơn (Richardson)."""

from __future__ import annotations

import math
from exam_format import exam_print as print
import sys
from dataclasses import dataclass
from input_utils import MathInputError, parse_real, split_number_row

import numpy as np


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")


LINE = "=" * 100


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
            print("  Lỗi: hãy nhập số dương, ví dụ 1e-7, 0.01 hoặc 1/10.")


def read_row(prompt: str, size: int) -> np.ndarray:
    while True:
        try:
            parts = split_number_row(input(prompt), size)
            return np.array([parse_number(part) for part in parts], dtype=float)
        except (MathInputError, ValueError, ZeroDivisionError):
            print("  Lỗi: dữ liệu không hợp lệ.")


def read_matrix(name: str, rows: int, columns: int) -> np.ndarray:
    print(f"\nNhập ma trận {name} kích thước {rows}×{columns}:")
    return np.vstack([read_row(f"  Dòng {i + 1}: ", columns) for i in range(rows)])


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


def matrix_norm(matrix: np.ndarray, norm_kind: str) -> float:
    order = np.inf if norm_kind == "inf" else 1
    return float(np.linalg.norm(matrix, order))


def vector_norm(vector: np.ndarray, norm_kind: str) -> float:
    order = np.inf if norm_kind == "inf" else 1
    return float(np.linalg.norm(vector, order))


def relative_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    numerator = float(np.linalg.norm(A @ x - b, np.inf))
    denominator = float(np.linalg.norm(A, np.inf) * np.linalg.norm(x, np.inf) + np.linalg.norm(b, np.inf))
    return numerator / denominator if denominator else numerator


def symmetric_jacobi_eigenvalues(
    matrix: np.ndarray,
    *,
    tolerance: float | None = None,
    maximum_rotations: int | None = None,
) -> np.ndarray:
    """Tính trị riêng ma trận đối xứng bằng phép quay Jacobi tự cài đặt."""

    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("Ma trận phải vuông và khác rỗng.")
    if not np.all(np.isfinite(A)):
        raise ValueError("Ma trận chứa NaN hoặc vô cực.")
    scale = max(1.0, float(np.linalg.norm(A, np.inf)))
    symmetry_tolerance = 100.0 * np.finfo(float).eps * scale
    if not np.allclose(A, A.T, rtol=0.0, atol=symmetry_tolerance):
        raise ValueError("Phép quay Jacobi yêu cầu ma trận đối xứng.")
    work = 0.5 * (A + A.T)
    n = A.shape[0]
    threshold = tolerance if tolerance is not None else symmetry_tolerance
    limit = maximum_rotations if maximum_rotations is not None else max(1, 50 * n * n)
    for _ in range(limit):
        upper = np.triu(np.abs(work), 1)
        p, q = np.unravel_index(int(np.argmax(upper)), upper.shape)
        if upper[p, q] <= threshold:
            return np.sort(np.diag(work).copy())
        app, aqq, apq = work[p, p], work[q, q], work[p, q]
        tau = (aqq - app) / (2.0 * apq)
        t = (
            math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
            if tau != 0.0
            else 1.0
        )
        cosine = 1.0 / math.sqrt(1.0 + t * t)
        sine = t * cosine
        for k in range(n):
            if k in (p, q):
                continue
            akp, akq = work[k, p], work[k, q]
            work[k, p] = work[p, k] = cosine * akp - sine * akq
            work[k, q] = work[q, k] = sine * akp + cosine * akq
        work[p, p] = app - t * apq
        work[q, q] = aqq + t * apq
        work[p, q] = work[q, p] = 0.0
    raise ArithmeticError("Phép quay Jacobi chưa hội tụ trong giới hạn số vòng quay.")


@dataclass
class Record:
    k: int
    x: np.ndarray
    difference: float
    error_bound: float
    residual: float
    relative_residual: float


@dataclass
class Result:
    x: np.ndarray
    records: list[Record]
    q: float
    converged: bool
    reason: str


def simple_iteration(
    A: np.ndarray,
    b: np.ndarray,
    B: np.ndarray,
    d: np.ndarray,
    x0: np.ndarray,
    norm_kind: str,
    epsilon: float,
    maximum_iterations: int,
    fixed_iterations: int,
) -> Result:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    B = np.asarray(B, dtype=float)
    d = np.asarray(d, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    n = A.shape[0] if A.ndim == 2 else 0
    if n == 0 or A.shape != (n, n) or B.shape != (n, n):
        raise ValueError("A và B phải là ma trận vuông cùng cấp.")
    if b.shape != (n,) or d.shape != (n,) or x0.shape != (n,):
        raise ValueError("b, d, x0 phải là vector đúng kích thước.")
    if not all(np.all(np.isfinite(item)) for item in (A, b, B, d, x0)):
        raise ValueError("Dữ liệu không được chứa NaN hoặc vô cùng.")
    if epsilon <= 0 or maximum_iterations <= 0 or fixed_iterations < 0:
        raise ValueError("epsilon, max_iter phải dương; fixed_iterations không âm.")
    q = matrix_norm(B, norm_kind)
    if fixed_iterations == 0 and q >= 1.0:
        raise ValueError(
            f"||B||={q:.7g} >= 1 nên chưa có chứng nhận co; "
            "hãy đổi tau/dạng lặp hoặc chọn chế độ đúng k bước."
        )
    limit = fixed_iterations if fixed_iterations > 0 else maximum_iterations
    x = x0.copy()
    records: list[Record] = []
    converged = False
    reason = f"đã thực hiện đúng k={fixed_iterations} bước"

    initial_residual = float(np.linalg.norm(A @ x - b, np.inf))
    if initial_residual <= epsilon and fixed_iterations == 0:
        return Result(x, records, q, True, "Nghiệm ban đầu đã thỏa phần dư.")

    for k in range(1, limit + 1):
        x_new = B @ x + d
        difference = vector_norm(x_new - x, norm_kind)
        error_bound = q / (1.0 - q) * difference if q < 1.0 else math.inf
        residual = float(np.linalg.norm(A @ x_new - b, np.inf))
        eta = relative_residual(A, x_new, b)
        records.append(Record(k, x_new.copy(), difference, error_bound, residual, eta))
        x = x_new
        # Chế độ PDF dừng theo đúng chặn sai số hậu nghiệm:
        #     E_k = q/(1-q)||X_k-X_{k-1}|| <= epsilon.
        # Phần dư ||Ax-b||_∞ chỉ in thêm để kiểm tra, không dùng làm điều kiện dừng.
        if fixed_iterations == 0 and error_bound <= epsilon:
            converged = True
            reason = f"E_k <= epsilon = {epsilon:.3e} theo công thức hậu nghiệm PDF"
            break

    if fixed_iterations > 0:
        # Đúng k bước: chỉ kết luận chứng nhận nếu chặn hậu nghiệm đã đạt epsilon;
        # không ép thêm điều kiện phần dư vì PDF không dùng phần dư để dừng.
        converged = bool(records and records[-1].error_bound <= epsilon)
    elif not converged:
        reason = f"đạt k_max={maximum_iterations} nhưng chưa thỏa epsilon"
    return Result(x, records, q, converged, reason)


def print_theory() -> None:
    section("A. THUẬT TOÁN PHƯƠNG PHÁP LẶP ĐƠN TUYẾN TÍNH")
    print("Input:")
    print("  • Hệ Ax=b và vector đầu x^(0).")
    print("  • Tham số tau để đưa hệ về x=Bx+d với B=I-tau*A, d=tau*b.")
    print("Output: nghiệm gần đúng, bảng lặp và chặn sai số theo PDF.")
    print("Các bước:")
    print("  B1. Chọn tau và lập B=I-tau*A, d=tau*b.")
    print("  B2. Kiểm tra q=||B||<1 trong chuẩn 1 hoặc chuẩn vô cùng.")
    print("  B3. Lặp x^(k+1)=B x^(k)+d.")
    print("  B4. Chặn hậu nghiệm theo PDF:")
    print("            E_k = q/(1-q)||x^(k)-x^(k-1)||.")
    print("            Dừng khi E_k <= epsilon.")
    print("  B5. Có thể in thêm phần dư r^(k)=b-Ax^(k) để kiểm tra, nhưng không dùng làm điều kiện dừng PDF.")
    print("Với A đối xứng xác định dương, tau tối ưu theo chuẩn phổ là")
    print("            tau=2/(lambda_min+lambda_max).")


def print_table(records: list[Record], decimals: int) -> None:
    if not records:
        return
    n = records[0].x.size
    headers = ["k"] + [f"x{i + 1}^(k)" for i in range(n)] + ["||dx||", "E_k", "||r||∞", "eta"]
    rows = []
    for record in records:
        rows.append([
            str(record.k),
            *[fmt(value, decimals) for value in record.x],
            f"{record.difference:.2e}",
            f"{record.error_bound:.2e}" if math.isfinite(record.error_bound) else "N/A",
            f"{record.residual:.2e}",
            f"{record.relative_residual:.2e}",
        ])
    widths = [max(len(headers[j]), *(len(row[j]) for row in rows)) for j in range(len(headers))]
    print("  " + " | ".join(headers[j].rjust(widths[j]) for j in range(len(headers))))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print("  " + " | ".join(row[j].rjust(widths[j]) for j in range(len(row))))


def main() -> None:
    print(LINE)
    print("LẶP ĐƠN TUYẾN TÍNH – DẠNG x=Bx+d")
    print(LINE)
    print("Nhập 0 ở menu để thoát; có thể nhập phân số như 1/3.")
    print("1. Nhập A,b rồi tự tạo dạng lặp x=Bx+d")
    print("2. Nhập trực tiếp ma trận lặp B và vector d")
    print("0. Thoát")
    source_choice = input("Chọn [Enter = 1]: ").strip() or "1"
    if source_choice == "0":
        return
    if source_choice not in {"1", "2"}:
        print("Lựa chọn không hợp lệ.")
        return

    n = read_int("Cấp ma trận n: ", 1)
    tau = None
    eigenvalues = None
    direct_mode = source_choice == "2"
    if direct_mode:
        print("\nChế độ nhập trực tiếp B,d. Không yêu cầu nhập A hoặc b.")
        B = read_matrix("B", n, n)
        d = read_row(f"Nhập vector d gồm {n} số: ", n)
        A = np.eye(n) - B
        b = d.copy()
    else:
        A = read_matrix("A", n, n)
        b = read_row(f"Nhập vector b gồm {n} số: ", n)
        print("\nCách tạo B,d từ A,b:")
        print("1. Tự động chọn tau cho A đối xứng xác định dương")
        print("2. Tự nhập tau")
        tau_choice = input("Chọn [Enter = 1]: ").strip() or "1"
        if tau_choice not in {"1", "2"}:
            print("Lựa chọn không hợp lệ.")
            return
    if not direct_mode and tau_choice == "1":
        if not np.allclose(A, A.T, rtol=0.0, atol=1e-12 * max(1.0, np.linalg.norm(A, np.inf))):
            print("Kết luận: A không đối xứng, không áp dụng được lựa chọn tự động này.")
            return
        eigenvalues = symmetric_jacobi_eigenvalues(A)
        if eigenvalues[0] <= 0.0:
            print("Kết luận: A không xác định dương, không áp dụng được công thức tau tự động.")
            return
        tau = 2.0 / float(eigenvalues[0] + eigenvalues[-1])
        B = np.eye(n) - tau * A
        d = tau * b
    elif not direct_mode:
        tau = read_positive("Nhập tau > 0: ")
        B = np.eye(n) - tau * A
        d = tau * b

    print("\nChọn chuẩn dùng để chứng nhận co:")
    print("1. Chuẩn vô cùng (khuyên dùng)")
    print("2. Chuẩn 1")
    norm_choice = input("Chọn [Enter = 1]: ").strip() or "1"
    norm_kind = "inf" if norm_choice == "1" else "one"

    print("\nVector đầu:")
    start_choice = input("Dùng x^(0)=0? [C/k, Enter=C]: ").strip().lower()
    x0 = np.zeros(n) if start_choice not in {"k", "khong", "không", "n", "no"} else read_row(
        f"Nhập x^(0) gồm {n} số: ", n
    )

    print("\nChế độ dừng theo PDF:")
    print("1. Sai số tiên nghiệm")
    print("2. Sai số hậu nghiệm")
    print("3. Đúng k bước")
    stop_choice = input("Chọn [Enter = 1]: ").strip() or "1"
    fixed_iterations = 0
    maximum_iterations = 10000
    if stop_choice == "1":
        epsilon = read_positive("Sai số epsilon [Enter = 1e-7]: ", 1e-7)
        maximum_iterations = 10000
        q_preview = matrix_norm(B, norm_kind)
        initial_gap = vector_norm(B @ x0 + d - x0, norm_kind)
        if not (0.0 < q_preview < 1.0):
            print("Cảnh báo: ||C|| không nhỏ hơn 1 nên công thức tiên nghiệm PDF chưa áp dụng được.")
        elif initial_gap == 0.0:
            fixed_iterations = 0
            maximum_iterations = 1
        else:
            argument = (1.0 - q_preview) * epsilon / initial_gap
            if argument <= 0.0:
                fixed_iterations = 0
                maximum_iterations = 1
            else:
                fixed_iterations = max(0, math.ceil(math.log(argument) / math.log(q_preview)))
                maximum_iterations = max(1, fixed_iterations)
                print(f"Số bước tiên nghiệm theo PDF: n = {fixed_iterations}.")
    elif stop_choice == "2":
        epsilon = read_positive("Sai số epsilon [Enter = 1e-7]: ", 1e-7)
        maximum_iterations = 10000
    else:
        fixed_iterations = read_int("Số bước k: ", 1)
        epsilon = 1e-12
        maximum_iterations = fixed_iterations
    decimals = read_int("Số chữ số sau dấu phẩy [Enter = 7]: ", 0, 7)

    print_theory()
    section("B. DỮ KIỆN VÀ DẠNG LẶP")
    if direct_mode:
        print("Dữ liệu nhập trực tiếp theo x^(k+1)=B x^(k)+d; không nhập A,b.")
    else:
        print_matrix("A", A, decimals)
        print_vector("b", b, decimals)
    if tau is not None:
        if eigenvalues is not None:
            print("Các trị riêng của A dùng để chọn tau:")
            print("  " + ", ".join(fmt(value, decimals) for value in eigenvalues))
            print(f"tau=2/(lambda_min+lambda_max)={tau:.{decimals}f}")
        else:
            print(f"tau={tau:.{decimals}f}")
    print_matrix("B", B, decimals)
    print_vector("d", d, decimals)
    q = matrix_norm(B, norm_kind)
    symbol = "∞" if norm_kind == "inf" else "1"
    print(f"q=||B||_{symbol}={q:.{decimals}f}.")
    print("B là ma trận co trong chuẩn đã chọn." if q < 1 else "Cảnh báo: q>=1, chưa có chứng nhận hội tụ.")

    try:
        result = simple_iteration(A, b, B, d, x0, norm_kind, epsilon, maximum_iterations, fixed_iterations)
    except ValueError as error:
        print(f"\nKhông thể thực hiện: {error}")
        return

    section("C. BẢNG LẶP")
    print_table(result.records, decimals)
    section("D. KẾT LUẬN")
    print(f"Dừng vì {result.reason}.")
    print_vector(f"x^({len(result.records)})", result.x, decimals)
    residual = A @ result.x - b
    if direct_mode:
        print_vector("x-Bx-d", residual, decimals)
        print(f"||x-Bx-d||_∞={np.linalg.norm(residual, np.inf):.7e}")
    else:
        print_vector("A x-b", residual, decimals)
        print(f"||Ax-b||_∞={np.linalg.norm(residual, np.inf):.7e}")
    if result.records and math.isfinite(result.records[-1].error_bound):
        print(f"Chặn sai số hậu nghiệm E_k={result.records[-1].error_bound:.7e}.")
        print("Theo PDF, điều kiện dừng của lặp đơn là E_k <= epsilon; phần dư chỉ là kiểm tra thêm.")
    if result.converged:
        print("KẾT LUẬN: nghiệm gần đúng đạt sai số epsilon theo chặn hậu nghiệm PDF.")
    elif fixed_iterations:
        print("KẾT LUẬN: đây là nghiệm gần đúng sau đúng số bước đề yêu cầu.")
    else:
        print("KẾT LUẬN: chưa xác nhận đạt sai số yêu cầu.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except Exception as error:
        print(f"\nLỗi trong quá trình tính toán: {error}")
