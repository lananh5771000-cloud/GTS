import sympy as sp
import numpy as np
import math
import sys
from exam_format import exam_print as print
from truyen_sai_so import propagate_bound, symbolic_derivative_bound
from dataclasses import dataclass
from typing import Callable, Literal
from input_utils import parse_math_expression, parse_real

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class ChordResult:
    root: float
    converged: bool
    certified: bool
    variant: str
    reason: str
    iterations: list[tuple[int, float, float, float]]
    residual: float
    error_bound: float | None
    bracket: tuple[float, float]


def chord_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    epsilon: float,
    *,
    variant: Literal["regula_falsi", "fixed_endpoint"] = "regula_falsi",
    second_derivative: Callable[[float], float] | None = None,
    m1: float | None = None,
    max_iter: int = 1000,
    denominator_tolerance: float = 1e-14,
    assumptions_verified: bool = False,
    fixed_iterations: int | None = None,
) -> ChordResult:
    """Hai biến thể dây cung được tách rõ, không dùng lẫn điều kiện hội tụ."""
    if variant not in {"regula_falsi", "fixed_endpoint"}:
        raise ValueError("Biến thể dây cung không hợp lệ.")
    if not all(math.isfinite(v) for v in (a, b, epsilon, denominator_tolerance)):
        raise ValueError("Dữ liệu phải hữu hạn.")
    if not a < b or epsilon <= 0 or max_iter <= 0 or denominator_tolerance <= 0:
        raise ValueError("Cần a < b và các ngưỡng dương.")

    if fixed_iterations is not None and fixed_iterations < 0:
        raise ValueError("fixed_iterations phai khong am.")

    def value(point: float) -> float:
        result = float(f(point))
        if not math.isfinite(result):
            raise ArithmeticError(f"f({point}) không hữu hạn.")
        return result

    fixed_mode = fixed_iterations is not None
    fa, fb = value(a), value(b)
    if (fa == 0.0 if fixed_mode else abs(fa) <= denominator_tolerance):
        exact = fa == 0.0
        return ChordResult(a, True, exact, variant, "Đầu mút a là nghiệm chính xác." if exact else "Đầu mút a chỉ đạt ngưỡng phần dư.", [], abs(fa), 0.0 if exact else None, (a, a) if exact else (a, b))
    if (fb == 0.0 if fixed_mode else abs(fb) <= denominator_tolerance):
        exact = fb == 0.0
        return ChordResult(b, True, exact, variant, "Đầu mút b là nghiệm chính xác." if exact else "Đầu mút b chỉ đạt ngưỡng phần dư.", [], abs(fb), 0.0 if exact else None, (b, b) if exact else (a, b))
    if fa * fb > 0:
        raise ValueError("[a,b] không phải khoảng đổi dấu.")
    if m1 is not None and (not math.isfinite(m1) or m1 <= 0):
        raise ValueError("m1 phải là chặn dưới dương của |f'|.")

    fixed = moving = None
    f_fixed = None
    if variant == "fixed_endpoint":
        if second_derivative is None:
            raise ValueError("Biến thể cố định một đầu cần f'' để chọn đúng đầu cố định.")
        f2a, f2b = float(second_derivative(a)), float(second_derivative(b))
        if not math.isfinite(f2a) or not math.isfinite(f2b) or f2a * f2b < 0:
            raise ValueError("Chưa đủ cơ sở xác nhận f'' không đổi dấu.")
        if fa * f2a > 0:
            fixed, f_fixed, moving = a, fa, b
        elif fb * f2b > 0:
            fixed, f_fixed, moving = b, fb, a
        else:
            raise ValueError("Không chọn được đầu cố định theo f(d)f''(d)>0.")

    records: list[tuple[int, float, float, float]] = []
    root = (a + b) / 2.0
    error = None
    loop_limit = fixed_iterations if fixed_iterations is not None else max_iter
    for k in range(1, loop_limit + 1):
        if variant == "regula_falsi":
            denominator = fb - fa
            scale = max(1.0, abs(fa), abs(fb))
            if abs(denominator) <= denominator_tolerance * scale:
                return ChordResult(root, False, False, variant, "Mẫu số dây cung gần 0.", records, abs(value(root)), None, (a, b))
            root = (a * fb - b * fa) / denominator
        else:
            if moving is None or fixed is None or f_fixed is None:
                raise ArithmeticError("Chua chon duoc dau co dinh cho phuong phap day cung.")
            fm = value(moving)
            denominator = fm - f_fixed
            scale = max(1.0, abs(fm), abs(f_fixed))
            if abs(denominator) <= denominator_tolerance * scale:
                return ChordResult(moving, False, False, variant, "Mẫu số dây cung gần 0.", records, abs(fm), None, (a, b))
            root = moving - fm * (moving - fixed) / denominator
        if not math.isfinite(root) or root < a or root > b:
            return ChordResult(root, False, False, variant, "Bước dây cung ra ngoài khoảng phân ly.", records, math.inf, None, (a, b))
        fr = value(root)
        error = abs(fr) / m1 if m1 is not None else None
        records.append((k, root, fr, error if error is not None else math.nan))
        if fixed_iterations is None and (
            abs(fr) <= denominator_tolerance or (error is not None and error <= epsilon)
        ):
            certified = fr == 0.0 or (assumptions_verified and error is not None)
            reason = "Đạt chặn hậu nghiệm |f(x_k)|/m1." if error is not None else "Phần dư bằng 0 trong số học máy."
            if not certified:
                reason += " Kết quả chưa được chứng nhận do giả thiết chỉ kiểm tra số."
            return ChordResult(root, True, certified, variant, reason, records, abs(fr), error, (a, b))
        if fixed_iterations is not None and fr == 0.0:
            return ChordResult(root, True, True, variant, "Nghiem chinh xac.", records, abs(fr), 0.0, (root, root))
        if fa * fr <= 0:
            b, fb = root, fr
        else:
            a, fa = root, fr
        if variant == "fixed_endpoint":
            moving = root
    if fixed_iterations is not None:
        reason = (
            f"Da thuc hien dung k={fixed_iterations} buoc day cung; "
            "chua dung tieu chuan epsilon de chung nhan."
        )
        return ChordResult(root, True, assumptions_verified and error is not None and error <= epsilon, variant, reason, records, abs(value(root)), records[-1][3] if records and math.isfinite(records[-1][3]) else None, (a, b))
    return ChordResult(root, False, False, variant, "Đã đạt max_iter.", records, abs(value(root)), records[-1][3] if records and math.isfinite(records[-1][3]) else None, (a, b))


def print_custom_algorithm(
    f_expr,
    a,
    b,
    d,
    x0,
    m1,
    cond_type,
    target_val,
    final_n,
    final_xn,
    precision,
    G_name="",
):
    fmt = f".{precision}f"
    print("\n" + "=" * 45)
    print("MÔ TẢ THUẬT TOÁN DÂY CUNG ĐÃ SỬ DỤNG")
    print("=" * 45)
    print(f"1. Thiết lập hàm số: f(x) = {f_expr}")
    print(f"2. Khoảng phân ly ban đầu: [{a}, {b}]")
    print("3. Khảo sát đạo hàm:")
    print("   - Đạo hàm f'(x) không đổi dấu trên khoảng phân ly.")
    print("   - Đạo hàm f''(x) không đổi dấu trên khoảng phân ly.")
    print(f"   - min|f'(x)| trên [{a}, {b}] là m1 = {m1:{fmt}}")
    print("4. Chọn điểm mốc (d) và điểm xuất phát (x0):")
    print("   - Điều kiện chọn mốc: f(d) * f''(d) > 0")
    print(f"   - Suy ra: Điểm mốc d = {d}, Điểm xuất phát x0 = {x0}")

    if cond_type == "x_abs":
        print(
            f"5. Điều kiện dừng: Sai số tuyệt đối của x: \u0394x = |f(x_k)|/m1 \u2264 {target_val}"
        )
    elif cond_type == "x_rel":
        print(
            f"5. Điều kiện dừng: \u03b4x \u2264 \u0394x/(|x_k|-\u0394x) \u2264 {target_val}, cần |x_k|>\u0394x"
        )
    elif cond_type == "G_abs":
        print(
            f"5. Điều kiện dừng: Sai số tuyệt đối của hàm {G_name}: \u0394{G_name} \u2264 {target_val}"
        )
    elif cond_type == "G_rel":
        print(
            f"5. Điều kiện dừng: B_G/(|{G_name}(x_k)|-B_G) \u2264 {target_val}"
        )
    elif cond_type == "iter":
        print(f"5. Điều kiện dừng: Lặp đủ {int(target_val)} lần.")

    print("6. Tiến trình lặp:")
    print(
        "   - Sử dụng công thức: x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ − f(x⁽ᵏ⁾)(x⁽ᵏ⁾ − d)/[f(x⁽ᵏ⁾) − f(d)]"
    )
    print("7. Kết luận:")
    print(f"   - Sau {final_n} bước lặp, điều kiện dừng đã được thỏa mãn.")
    print(f"   - Nghiệm gần đúng cuối cùng: x \u2248 {final_xn:{fmt}}")
    print("=" * 45)


def secant_method(max_iter=1000):
    print("=== GIẢI PHƯƠNG TRÌNH BẰNG PHƯƠNG PHÁP DÂY CUNG ===\n")
    print("Input: f(x), khoảng phân ly, điểm mốc và điều kiện dừng.")
    print("Output: bảng lặp, nghiệm gần đúng và đánh giá sai số.")
    print("Công thức: x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ − f(x⁽ᵏ⁾)(x⁽ᵏ⁾−d)/[f(x⁽ᵏ⁾)−f(d)].\n")
    x, pi_sym, e_sym = sp.symbols("x pi e")

    # 1. Nhập phương trình gốc
    f_input = input("Nhập phương trình f(x) = 0 (vd: x**3 - x - 1): ")

    # 2. Xử lý hàm liên quan (Thể tích, Diện tích...)
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

    # 4. Thay số và thiết lập hàm f(x)
    try:
        f_expr = parse_math_expression(f_input, {"x": x, "pi": sp.Float(pi_val), "e": sp.Float(e_val)})
        f_func = sp.lambdify(x, f_expr, "numpy")
    except Exception as e:
        print("Lỗi cú pháp hàm số:", e)
        return

    # 5. Xây dựng công thức truyền sai số tự động cho G(x)
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
            input("\nBạn muốn hiển thị bao nhiêu số sau dấu phẩy (vd: 6, 8, 10): ")
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

    f_prime = sp.diff(f_expr, x)
    f_double_prime = sp.diff(f_prime, x)

    print(f"\nf'(x) = {f_prime}")
    print(f"f''(x) = {f_double_prime}")

    X_vals = np.linspace(a, b, 1000)
    f_prime_func = sp.lambdify(x, f_prime, "numpy")
    f_double_prime_func = sp.lambdify(x, f_double_prime, "numpy")

    Y_prime = f_prime_func(X_vals)
    Y_double_prime = f_double_prime_func(X_vals)

    abs_Y_prime = np.abs(Y_prime)
    m1 = np.min(abs_Y_prime)
    M1 = np.max(abs_Y_prime)

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
    print(f"-> M1 = max|f'(x)| = {M1:{fmt}}")
    if not math.isfinite(float(m1)) or m1 <= 1e-14:
        print("[X] Không có chặn dưới dương đáng tin cậy cho |f'|; dừng.")
        return

    # --- BƯỚC 2: CHỌN ĐIỂM MỐC (d) VÀ ĐIỂM XUẤT PHÁT (x0) ---
    print("\n--- BƯỚC 2: CHỌN ĐIỂM MỐC (d) VÀ ĐIỂM XUẤT PHÁT (x0) ---")
    fa = float(f_expr.subs(x, a))
    fb = float(f_expr.subs(x, b))

    if abs(fa) <= 1e-14:
        print(f"Nghiệm ở đầu mút: x = a = {a}.")
        return
    if abs(fb) <= 1e-14:
        print(f"Nghiệm ở đầu mút: x = b = {b}.")
        return

    if fa * fb > 0:
        print("[X] LỖI: f(a) * f(b) > 0. (a, b) không phải là khoảng cách ly nghiệm!")
        return

    if fa * sign_f2 > 0:
        d, x0 = a, b
    else:
        d, x0 = b, a

    print("Điều kiện chọn mốc: f(d) * f''(x) > 0")
    print(f"=> CHỌN: Điểm mốc d = {d}, Điểm xuất phát x0 = {x0}")

    # --- BƯỚC 3: ĐIỀU KIỆN DỪNG ---
    print("\n--- BƯỚC 3: CHỌN ĐIỂM KIỆN DỪNG ---")
    target_val = 0
    cond_type = ""

    if use_G == "y":
        print(f"1. Sai số tuyệt đối của hàm {G_name} (\u0394{G_name} \u2264 \u03b5)")
        print(f"2. Chặn tương đối B_G/(|{G_name}(x_k)|-B_G) \u2264 \u03b5")
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
        print("1. Sai số tuyệt đối của x (\u0394x = |f(x_k)|/m1 \u2264 \u03b5)")
        print("2. Chặn tương đối của x (\u03b4x \u2264 \u0394x/(|x_k|-\u0394x) \u2264 \u03b5)")
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
        header = f"{'k':<3} | {'x⁽ᵏ⁾':<15} | {'f(x⁽ᵏ⁾)':<15} | {'Eₓ':<15} | {G_name:<15} | {'Bᴳ/XPᴳ':<15} | {'Bᴳ tương đối':<15}"
    else:
        header = f"{'k':<3} | {'x⁽ᵏ⁾':<15} | {'f(x⁽ᵏ⁾)':<15} | {'Δx (Tuyệt đối)':<18} | {'δx (Tương đối)':<18}"

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    xk = x0
    k = 0
    fd = float(f_expr.subs(x, d))

    while True:
        if k >= max_iter:
            print(f"[X] Dừng tại max_iter = {max_iter}; chưa chứng nhận đạt sai số.")
            return
        fxk = float(f_expr.subs(x, xk))
        if not math.isfinite(fxk):
            print("[X] f(x⁽ᵏ⁾) không hữu hạn; dừng.")
            return

        # Chặn hậu nghiệm theo phần dư, chỉ đúng khi m1 đã được bảo đảm.
        delta_x = abs(fxk) / m1
        delta_x_rel = (
            delta_x / (abs(xk) - delta_x)
            if abs(xk) > delta_x
            else float("inf")
        )

        if use_G == "y":
            G_val = float(G_func(xk))
            interval_left = max(a, xk - delta_x)
            interval_right = min(b, xk + delta_x)
            M_G = symbolic_derivative_bound(G_eval_expr, x, interval_left, interval_right)
            G_bound_certified = M_G is not None and delta_pi == 0.0 and delta_e == 0.0
            if M_G is not None:
                propagated = propagate_bound(
                    G_func, G_prime_func, xk, delta_x, (a, b),
                    derivative_bound=M_G,
                    derivative_bound_verified=G_bound_certified,
                )
                delta_G_val = propagated.absolute_bound
                delta_G_rel = propagated.relative_bound if propagated.relative_bound is not None else float("inf")
            else:
                delta_G_val = float(delta_G_func(xk, delta_x))
                delta_G_rel = float("inf")
                G_bound_certified = False

            print(
                f"{k:<3} | {xk:<15.{precision}f} | {fxk:<15.{precision}e} | {delta_x:<15.{precision}e} | {G_val:<15.{precision}f} | {delta_G_val:<15.{precision}e} | {delta_G_rel:<15.{precision}e}"
            )

            if cond_type == "iter" and k >= target_val:
                break
            elif cond_type == "G_abs" and G_bound_certified and delta_G_val <= target_val:
                break
            elif cond_type == "G_rel" and G_bound_certified and delta_G_rel <= target_val:
                break
            elif not G_bound_certified:
                print("    ΔG chỉ là xấp xỉ bậc nhất/kiểm tra số; không dùng để chứng nhận dừng.")
        else:
            print(
                f"{k:<3} | {xk:<15.{precision}f} | {fxk:<15.{precision}e} | {delta_x:<18.{precision}e} | {delta_x_rel:<18.{precision}e}"
            )

            if cond_type == "iter" and k >= target_val:
                break
            elif cond_type == "x_abs" and delta_x <= target_val:
                break
            elif cond_type == "x_rel" and delta_x_rel <= target_val:
                break

        # Công thức lặp dây cung (1 đầu mút d cố định)
        denominator = fxk - fd
        if abs(denominator) <= 1e-14 * max(1.0, abs(fxk), abs(fd)):
            print("\n[!] LỖI: Chia cho 0 do f(x⁽ᵏ⁾) − f(d) = 0. Thuật toán dừng sớm.")
            break
        x_next = xk - (fxk * (xk - d)) / denominator
        if not math.isfinite(x_next) or not a <= x_next <= b:
            print("[X] Bước dây cung ra ngoài khoảng phân ly hoặc không hữu hạn; dừng.")
            return

        xk = x_next
        k += 1

    print("-" * len(header))
    print(f"=> Quá trình lặp hoàn tất tại bước {k}.")

    # In thuật toán
    print_algo = (
        input("\nIn phần thuật toán để chép bài? [C/k]: ")
        .strip()
        .lower()
    )
    if print_algo in {"", "c", "co", "có", "y", "yes"}:
        print_custom_algorithm(
            f_input, a, b, d, x0, m1, cond_type, target_val, k, xk, precision, G_name
        )


if __name__ == "__main__":
    try:
        secant_method()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã dừng chương trình; không có dữ liệu đầu vào đầy đủ.")
    except Exception as error:
        print(f"\nKhông thể thực hiện: {error}")
