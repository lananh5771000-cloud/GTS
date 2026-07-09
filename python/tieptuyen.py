import sympy as sp
import numpy as np
import math
import sys
from exam_format import exam_print as print
from truyen_sai_so import propagate_bound, symbolic_derivative_bound
from dataclasses import dataclass
from typing import Callable
from input_utils import parse_math_expression, parse_real

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class NewtonResult:
    root: float
    converged: bool
    certified: bool
    reason: str
    iterations: list[tuple[int, float, float, float, str, float | None]]
    residual: float
    error_bound: float | None
    bracket: tuple[float, float]


def safeguarded_newton(
    f: Callable[[float], float],
    derivative: Callable[[float], float],
    a: float,
    b: float,
    epsilon: float,
    *,
    x0: float | None = None,
    second_derivative: Callable[[float], float] | None = None,
    max_iter: int = 100,
    derivative_tolerance: float = 1e-14,
    assumptions_verified: bool = False,
    samples: int = 1001,
    fixed_iterations: int | None = None,
) -> NewtonResult:
    """Newton bảo vệ bởi khoảng đổi dấu và bước chia đôi."""
    if not all(math.isfinite(v) for v in (a, b, epsilon, derivative_tolerance)):
        raise ValueError("Dữ liệu phải hữu hạn.")
    if not a < b or epsilon <= 0 or derivative_tolerance <= 0 or max_iter <= 0:
        raise ValueError("Cần a < b; epsilon, tolerance và max_iter dương.")

    if fixed_iterations is not None and fixed_iterations < 0:
        raise ValueError("fixed_iterations phai khong am.")

    def value(fun: Callable[[float], float], point: float, name: str) -> float:
        result = float(fun(point))
        if not math.isfinite(result):
            raise ArithmeticError(f"{name}({point}) không hữu hạn.")
        return result

    fixed_mode = fixed_iterations is not None
    fa, fb = value(f, a, "f"), value(f, b, "f")
    if (fa == 0.0 if fixed_mode else abs(fa) <= derivative_tolerance):
        exact = fa == 0.0
        return NewtonResult(a, True, exact, "Đầu mút a là nghiệm chính xác." if exact else "Đầu mút a chỉ đạt ngưỡng phần dư.", [], abs(fa), 0.0 if exact else None, (a, a) if exact else (a, b))
    if (fb == 0.0 if fixed_mode else abs(fb) <= derivative_tolerance):
        exact = fb == 0.0
        return NewtonResult(b, True, exact, "Đầu mút b là nghiệm chính xác." if exact else "Đầu mút b chỉ đạt ngưỡng phần dư.", [], abs(fb), 0.0 if exact else None, (b, b) if exact else (a, b))
    if fa * fb > 0:
        raise ValueError("[a,b] không đổi dấu.")

    grid = [a + (b - a) * i / (samples - 1) for i in range(samples)]
    derivative_values = [value(derivative, p, "f'") for p in grid]
    m1 = min(abs(v) for v in derivative_values)
    derivative_nonzero_sampled = m1 > derivative_tolerance
    curvature_constant_sampled = True
    if second_derivative is not None:
        second_values = [value(second_derivative, p, "f''") for p in grid]
        curvature_constant_sampled = min(second_values) >= 0 or max(second_values) <= 0

    if x0 is None:
        if second_derivative is not None and fa * value(second_derivative, a, "f''") > 0:
            x = a
        elif second_derivative is not None and fb * value(second_derivative, b, "f''") > 0:
            x = b
        else:
            x = (a + b) / 2.0
    else:
        if not math.isfinite(x0) or not a <= x0 <= b:
            raise ValueError("x0 phải hữu hạn và thuộc [a,b].")
        x = x0

    rows: list[tuple[int, float, float, float, str, float | None]] = []
    last_bound = None
    loop_limit = fixed_iterations if fixed_iterations is not None else max_iter
    for k in range(loop_limit + 1):
        fx = value(f, x, "f")
        residual = abs(fx)
        last_bound = residual / m1 if derivative_nonzero_sampled else None
        rows.append((k, x, fx, 0.0 if k == 0 else rows[-1][3], "khởi tạo" if k == 0 else rows[-1][4], last_bound))
        if residual <= derivative_tolerance and (fixed_iterations is None or fx == 0.0):
            exact = fx == 0.0
            return NewtonResult(
                x,
                True,
                exact or assumptions_verified,
                "Tìm được nghiệm chính xác." if exact else "Phần dư đạt tolerance; chứng nhận phụ thuộc các giả thiết đã xác minh.",
                rows,
                residual,
                0.0 if residual == 0.0 else last_bound,
                (x, x) if residual == 0.0 else (a, b),
            )
        if fixed_iterations is None and last_bound is not None and last_bound <= epsilon:
            certified = assumptions_verified and derivative_nonzero_sampled
            reason = "Đạt chặn hậu nghiệm |f(x_k)|/m1."
            if not certified:
                reason += " Các giả thiết mới được khảo sát số, chưa phải chứng minh tuyệt đối."
            return NewtonResult(x, True, certified, reason, rows, residual, last_bound, (a, b))
        if fixed_iterations is None and (b - a) / 2.0 <= epsilon:
            midpoint = (a + b) / 2.0
            fm = abs(value(f, midpoint, "f"))
            certified = assumptions_verified
            return NewtonResult(midpoint, True, certified, "Đạt chặn sai số từ khoảng đổi dấu.", rows, fm, (b - a) / 2.0, (a, b))

        if fixed_iterations is not None and k >= fixed_iterations:
            reason = (
                f"Da thuc hien dung k={fixed_iterations} buoc Newton; "
                "chua dung tieu chuan epsilon de chung nhan."
            )
            return NewtonResult(x, True, assumptions_verified and last_bound is not None and last_bound <= epsilon, reason, rows, residual, last_bound, (a, b))

        dfx = value(derivative, x, "f'")
        method = "Newton"
        proposal = math.nan
        if abs(dfx) > derivative_tolerance:
            proposal = x - fx / dfx
        if not math.isfinite(proposal) or not a < proposal < b:
            proposal, method = (a + b) / 2.0, "chia đôi bảo vệ"
        else:
            fp = value(f, proposal, "f")
            if abs(fp) > residual:
                proposal, method = (a + b) / 2.0, "chia đôi bảo vệ"
        f_new = value(f, proposal, "f")
        if fa * f_new <= 0:
            b, fb = proposal, f_new
        else:
            a, fa = proposal, f_new
        step = abs(proposal - x)
        x = proposal
        rows[-1] = (k, rows[-1][1], fx, step, method, last_bound)

    residual = abs(value(f, x, "f"))
    reason = "Đã đạt max_iter."
    if not curvature_constant_sampled:
        reason += " f'' đổi dấu trên lưới kiểm tra."
    return NewtonResult(x, False, False, reason, rows, residual, last_bound, (a, b))


def print_custom_algorithm(
    f_expr,
    a,
    b,
    x0,
    m1,
    M2,
    cond_type,
    target_val,
    final_n,
    final_xn,
    precision,
    G_name="",
):
    fmt = f".{precision}f"
    print("\n" + "=" * 50)
    print("MÔ TẢ THUẬT TOÁN TIẾP TUYẾN (NEWTON-RAPHSON)")
    print("=" * 50)
    print(f"1. Thiết lập hàm số: f(x) = {f_expr}")
    print(f"2. Khoảng phân ly ban đầu: [{a}, {b}]")
    print("3. Khảo sát đạo hàm:")
    print("   - f'(x) và f''(x) không đổi dấu trên khoảng phân ly.")
    print(f"   - m1 = min|f'(x)| = {m1:{fmt}}")
    print(f"   - M2 = max|f''(x)| = {M2:{fmt}}")
    print("4. Chọn điểm xuất phát x0 (Điều kiện Fourier):")
    print("   - Phải thỏa mãn f(x0) * f''(x) > 0")
    print(f"   - Trích xuất được điểm xuất phát x0 = {x0}")

    if cond_type == "x_abs":
        print("5. Điều kiện dừng: Sai số tuyệt đối của x (Hậu nghiệm):")
        print(f"   \u0394x = (M2 / 2m1) * |x_n - x_{{n-1}}|^2 \u2264 {target_val}")
    elif cond_type == "x_rel":
        print("5. Điều kiện dừng: Sai số tương đối của x:")
        print(f"   \u03b4x \u2264 \u0394x/(|x_n|-\u0394x) \u2264 {target_val}, cần |x_n|>\u0394x")
    elif cond_type == "G_abs":
        print(
            f"5. Điều kiện dừng: Sai số tuyệt đối của hàm {G_name}: \u0394{G_name} \u2264 {target_val}"
        )
    elif cond_type == "G_rel":
        print(
            f"5. Điều kiện dừng: B_G/(|{G_name}(x_n)|-B_G) \u2264 {target_val}"
        )
    elif cond_type == "iter":
        print(f"5. Điều kiện dừng: Lặp đủ {int(target_val)} lần.")

    print("6. Tiến trình lặp:")
    print("   - Sử dụng công thức: x_{n+1} = x_n - f(x_n) / f'(x_n)")
    print("7. Kết luận:")
    print(f"   - Sau {final_n} bước lặp, điều kiện dừng đã được thỏa mãn.")
    print(f"   - Nghiệm gần đúng cuối cùng: x \u2248 {final_xn:{fmt}}")
    print("=" * 50)


def newton_method(max_iter=1000):
    print("=== GIẢI PHƯƠNG TRÌNH BẰNG PHƯƠNG PHÁP TIẾP TUYẾN (NEWTON-RAPHSON) ===\n")
    print("Input: f(x), khoảng phân ly, x_0 và điều kiện dừng.")
    print("Output: bảng lặp, nghiệm gần đúng và đánh giá sai số.")
    print("Công thức: x_(k+1)=x_k-f(x_k)/f'(x_k).\n")
    x, pi_sym, e_sym = sp.symbols("x pi e")

    # 1. Nhập phương trình gốc
    f_input = input("Nhập phương trình f(x) = 0 (vd: x**3 - x - 1): ")

    # 2. Xử lý hàm liên quan
    use_G = (
        input(
            "Bài toán có điều kiện dừng theo một hàm khác phụ thuộc vào x không? (y/n): "
        )
        .strip()
        .lower()
    )

    G_expr_raw = None
    G_name = "G"
    g_input = ""
    if use_G == "y":
        G_name = input("Nhập ký hiệu của hàm đó (vd: V, S, F...): ").strip() or "G"
        g_input = input(f"Nhập biểu thức {G_name}(x) (vd: 1/6 * pi * x**3): ")
        G_expr_raw = parse_math_expression(g_input, {"x": x, "pi": pi_sym, "e": e_sym})

    # 3. Xử lý hằng số Pi, E
    combined_input = f_input + " " + g_input
    pi_val, delta_pi = math.pi, 0.0
    e_val, delta_e = math.e, 0.0

    if "pi" in combined_input:
        k_pi = int(
            input("\nPhát hiện có số \u03c0. Nhập số chữ số sau dấu phẩy của \u03c0: ")
        )
        pi_val = round(math.pi, k_pi)
        delta_pi = 0.5 * 10 ** (-k_pi)

    if "e" in combined_input:
        k_e = int(
            input("\nPhát hiện có hằng số e. Nhập số chữ số sau dấu phẩy của e: ")
        )
        e_val = round(math.e, k_e)
        delta_e = 0.5 * 10 ** (-k_e)

    # 4. Thay số và thiết lập hàm f(x), f'(x), f''(x)
    try:
        f_expr = parse_math_expression(f_input, {"x": x, "pi": sp.Float(pi_val), "e": sp.Float(e_val)})
        f_prime = sp.diff(f_expr, x)
        f_double_prime = sp.diff(f_prime, x)

        f_func = sp.lambdify(x, f_expr, "numpy")
        f_prime_func = sp.lambdify(x, f_prime, "numpy")
        f_double_prime_func = sp.lambdify(x, f_double_prime, "numpy")
    except Exception as e:
        print("Lỗi cú pháp hàm số:", e)
        return

    # 5. Xây dựng công thức truyền sai số cho G(x)
    G_func = None
    delta_G_func = None
    G_prime_func = None
    if use_G == "y":
        dG_dx = sp.diff(G_expr_raw, x)
        dG_dpi = sp.diff(G_expr_raw, pi_sym)
        dG_de = sp.diff(G_expr_raw, e_sym)

        delta_x_sym = sp.Symbol("delta_x")
        delta_G_expr = (
            sp.Abs(dG_dx) * delta_x_sym
            + sp.Abs(dG_dpi) * delta_pi
            + sp.Abs(dG_de) * delta_e
        )

        subs_dict = {pi_sym: pi_val, e_sym: e_val}
        G_eval_expr = G_expr_raw.subs(subs_dict)
        delta_G_eval_expr = delta_G_expr.subs(subs_dict)

        G_func = sp.lambdify(x, G_eval_expr, "numpy")
        G_prime_func = sp.lambdify(x, sp.diff(G_eval_expr, x), "numpy")
        delta_G_func = sp.lambdify((x, delta_x_sym), delta_G_eval_expr, "numpy")

    # --- TÌM NGHIỆM GẦN ĐÚNG ---
    try:
        precision = int(
            input("\nĐộ chính xác hiển thị (số chữ số sau dấu phẩy, vd: 8): ")
        )
    except ValueError:
        precision = 8
    fmt = f".{precision}f"

    preview_choice = (
        input("\nBạn có muốn quét tìm trước khoảng phân ly chứa nghiệm không? (y/n): ")
        .strip()
        .lower()
    )
    if preview_choice == "y":
        print("Đang quét không gian [-50, 50]...")
        X_scan = np.linspace(-50, 50, 2000)
        Y_scan = f_func(X_scan)
        roots_preview = []
        for i in range(len(X_scan) - 1):
            if Y_scan[i] * Y_scan[i + 1] <= 0:
                try:
                    root = sp.nsolve(f_expr, x, X_scan[i])
                    roots_preview.append(round(float(root), 6))
                except Exception:
                    pass
        if roots_preview:
            print(
                f"-> Gợi ý: Phát hiện các nghiệm gần đúng quanh: {sorted(list(set(roots_preview)))}"
            )
        else:
            print("-> Không tìm thấy nghiệm thực nào trong khoảng [-50, 50].")

    # --- BƯỚC 1: KHOẢNG CÁCH LY VÀ XÉT DẤU ---
    print("\n--- BƯỚC 1: KHOẢNG CÁCH LY VÀ XÉT DẤU ---")
    a = parse_real(input("Nhập cận dưới a = "))
    b = parse_real(input("Nhập cận trên b = "))

    print(f"\nf'(x) = {f_prime}")
    print(f"f''(x) = {f_double_prime}")

    X_vals = np.linspace(a, b, 1000)
    Y_prime = f_prime_func(X_vals)
    Y_double_prime = f_double_prime_func(X_vals)

    abs_Y_prime = np.abs(Y_prime)
    abs_Y_double_prime = np.abs(Y_double_prime)

    m1 = np.min(abs_Y_prime)
    M2 = np.max(abs_Y_double_prime)

    sign_f_prime = np.sign(Y_prime)
    sign_f_double_prime = np.sign(Y_double_prime)
    print("Kiểm tra đạo hàm trên lưới chỉ là kiểm tra số, không phải chứng minh tuyệt đối.")

    if not np.all(sign_f_prime == sign_f_prime[0]):
        print(
            "[!] CẢNH BÁO: Đạo hàm bậc 1 đổi dấu trên (a, b). Khoảng phân ly không hợp lệ!"
        )
    if not np.all(sign_f_double_prime == sign_f_double_prime[0]):
        print(
            "[!] CẢNH BÁO: Đạo hàm bậc 2 đổi dấu trên (a, b). Hàm không lồi/lõm hoàn toàn!"
        )

    sign_f2 = 1 if sign_f_double_prime[0] > 0 else -1
    print(f"-> m1 = min|f'(x)| = {m1:{fmt}}")
    print(f"-> M2 = max|f''(x)| = {M2:{fmt}}")

    if m1 == 0:
        print(
            "[X] LỖI: m1 = 0, đạo hàm bậc 1 triệt tiêu. Thuật toán có thể chia cho 0."
        )
        return

    # --- BƯỚC 2: CHỌN ĐIỂM XUẤT PHÁT (x0) ---
    print("\n--- BƯỚC 2: CHỌN ĐIỂM XUẤT PHÁT (x0) ---")
    fa = float(f_expr.subs(x, a))
    fb = float(f_expr.subs(x, b))

    if abs(fa) <= 1e-14:
        print(f"Nghiệm ở đầu mút: x = a = {a}.")
        return
    if abs(fb) <= 1e-14:
        print(f"Nghiệm ở đầu mút: x = b = {b}.")
        return

    if fa * fb > 0:
        print(
            "[X] LỖI: f(a) * f(b) > 0. (a, b) không phải khoảng cách ly nghiệm hợp lệ!"
        )
        return

    if fa * sign_f2 > 0:
        x0 = a
    else:
        x0 = b

    print("Điều kiện chọn điểm xuất phát (Fourier): f(x0) * f''(x) > 0")
    print(f"=> CHỌN: Điểm xuất phát x0 = {x0}")

    # --- BƯỚC 3: ĐIỀU KIỆN DỪNG ---
    print("\n--- BƯỚC 3: CHỌN ĐIỀU KIỆN DỪNG ---")
    target_val = 0
    cond_type = ""

    if use_G == "y":
        print(f"1. Sai số tuyệt đối của hàm {G_name} (\u0394{G_name} \u2264 \u03b5)")
        print(f"2. Chặn tương đối B_G/(|{G_name}(x_n)|-B_G) \u2264 \u03b5")
        print("3. Dừng theo số lần lặp cố định")
        choice = input("Lựa chọn (1/2/3): ")
        if choice == "1":
            target_val = parse_real(input("Nhập \u03b5 mục tiêu (vd: 0.5e-3): "))
            cond_type = "G_abs"
        elif choice == "2":
            target_val = parse_real(input("Nhập \u03b5 mục tiêu (vd: 0.5e-3): "))
            cond_type = "G_rel"
        else:
            target_val = int(input("Nhập số lần lặp tối đa: "))
            cond_type = "iter"
    else:
        print(
            "1. Sai số tuyệt đối của x (\u0394x = (M2/2m1)*|x_n - x_{n-1}|^2 \u2264 \u03b5)"
        )
        print("2. Chặn tương đối của x (\u03b4x \u2264 \u0394x/(|x_n|-\u0394x) \u2264 \u03b5)")
        print("3. Dừng theo số lần lặp cố định")
        choice = input("Lựa chọn (1/2/3): ")
        if choice == "1":
            target_val = parse_real(input("Nhập sai số tuyệt đối epsilon = "))
            cond_type = "x_abs"
        elif choice == "2":
            target_val = parse_real(input("Nhập sai số tương đối epsilon = "))
            cond_type = "x_rel"
        else:
            target_val = int(input("Nhập số lần lặp tối đa: "))
            cond_type = "iter"

    # --- BƯỚC 4: QUÁ TRÌNH LẶP ---
    print("\n--- BƯỚC 4: QUÁ TRÌNH LẶP ---")

    if use_G == "y":
        header = f"{'n':<3} | {'x_n':<15} | {'f(x_n)':<15} | {'E_x':<15} | {G_name:<15} | {'B_G/XP_G':<15} | {'B_G tương đối':<15}"
    else:
        header = f"{'n':<3} | {'x_n':<15} | {'f(x_n)':<15} | {'\u0394x (Tuyệt đối)':<18} | {'\u03b4x (Tương đối)':<18}"

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    xn = x0
    x_prev = None
    n = 0
    error_coeff = M2 / (2 * m1)

    while True:
        if n >= max_iter:
            print(f"[X] Dừng tại max_iter = {max_iter}; chưa chứng nhận đạt sai số.")
            return
        fxn = float(f_expr.subs(x, xn))
        fpxn = float(f_prime.subs(x, xn))

        # Đánh giá sai số (bỏ qua tính sai số ở n=0 do chưa có x_prev)
        if n == 0:
            str_dx = "-"
            str_dx_rel = "-"
            str_dG = "-"
            str_dG_rel = "-"
            delta_x = float("inf")
            delta_x_rel = float("inf")
        else:
            delta_x = error_coeff * (abs(xn - x_prev) ** 2)
            delta_x_rel = (
                delta_x / (abs(xn) - delta_x)
                if abs(xn) > delta_x
                else float("inf")
            )
            str_dx = (
                f"{delta_x:<15.{precision}e}"
                if use_G == "y"
                else f"{delta_x:<18.{precision}e}"
            )
            str_dx_rel = f"{delta_x_rel:<18.{precision}e}"

        # Tính và in bảng
        if use_G == "y":
            G_val = float(G_func(xn))
            if n > 0:
                interval_left = max(a, xn - delta_x)
                interval_right = min(b, xn + delta_x)
                M_G = symbolic_derivative_bound(G_eval_expr, x, interval_left, interval_right)
                G_bound_certified = M_G is not None and delta_pi == 0.0 and delta_e == 0.0
                if M_G is not None:
                    propagated = propagate_bound(
                        G_func, G_prime_func, xn, delta_x, (a, b),
                        derivative_bound=M_G,
                        derivative_bound_verified=G_bound_certified,
                    )
                    delta_G_val = propagated.absolute_bound
                    delta_G_rel = propagated.relative_bound if propagated.relative_bound is not None else float("inf")
                else:
                    delta_G_val = float(delta_G_func(xn, delta_x))
                    delta_G_rel = float("inf")
                    G_bound_certified = False
                str_dG = f"{delta_G_val:<15.{precision}e}"
                str_dG_rel = f"{delta_G_rel:<15.{precision}e}"
            else:
                delta_G_val = float("inf")
                delta_G_rel = float("inf")

            print(
                f"{n:<3} | {xn:<15.{precision}f} | {fxn:<15.{precision}e} | {str_dx} | {G_val:<15.{precision}f} | {str_dG} | {str_dG_rel}"
            )

            if n > 0:
                if cond_type == "iter" and n >= target_val:
                    break
                elif cond_type == "G_abs" and G_bound_certified and delta_G_val <= target_val:
                    break
                elif cond_type == "G_rel" and G_bound_certified and delta_G_rel <= target_val:
                    break
                elif not G_bound_certified:
                    print("    ΔG chỉ là xấp xỉ bậc nhất/kiểm tra số; không dùng để chứng nhận dừng.")
        else:
            print(
                f"{n:<3} | {xn:<15.{precision}f} | {fxn:<15.{precision}e} | {str_dx} | {str_dx_rel}"
            )

            if n > 0:
                if cond_type == "iter" and n >= target_val:
                    break
                elif cond_type == "x_abs" and delta_x <= target_val:
                    break
                elif cond_type == "x_rel" and delta_x_rel <= target_val:
                    break

        # Nếu đạt số bước (khi chọn dừng theo số bước nhưng n=0)
        if n == 0 and cond_type == "iter" and n >= target_val:
            break

        force_bisection = abs(fpxn) <= 1e-14 * max(1.0, abs(fxn))
        if force_bisection:
            print(f"\n[!] f'(x_{n}) gần 0; chuyển sang chia đôi bảo vệ.")

        # Công thức lặp tiếp tuyến
        x_next = (a + b) / 2.0 if force_bisection else xn - (fxn / fpxn)
        method = "chia đôi bảo vệ" if force_bisection else "Newton"
        if not math.isfinite(x_next) or not a <= x_next <= b:
            x_next = (a + b) / 2.0
            method = "chia đôi bảo vệ"
        f_next = float(f_expr.subs(x, x_next))
        if not math.isfinite(f_next):
            print("[X] f tại bước mới không hữu hạn; dừng.")
            return
        if abs(f_next) > abs(fxn):
            x_next = (a + b) / 2.0
            f_next = float(f_expr.subs(x, x_next))
            method = "chia đôi bảo vệ"
        if method != "Newton":
            print(f"  Bước {n + 1}: chuyển sang {method}.")
        if fa * f_next <= 0:
            b, fb = x_next, f_next
        else:
            a, fa = x_next, f_next

        x_prev = xn
        xn = x_next
        n += 1

    print("-" * len(header))
    print(f"=> Quá trình lặp hoàn tất tại bước {n}.")

    # In thuật toán
    print_algo = (
        input("\nIn phần thuật toán để chép bài? [C/k]: ")
        .strip()
        .lower()
    )
    if print_algo in {"", "c", "co", "có", "y", "yes"}:
        print_custom_algorithm(
            f_input, a, b, x0, m1, M2, cond_type, target_val, n, xn, precision, G_name
        )


if __name__ == "__main__":
    try:
        newton_method()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã dừng chương trình; không có dữ liệu đầu vào đầy đủ.")
    except Exception as error:
        print(f"\nKhông thể thực hiện: {error}")
