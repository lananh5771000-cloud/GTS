import sympy as sp
import numpy as np
import math
import sys
from truyen_sai_so import propagate_bound, symbolic_derivative_bound
from dataclasses import dataclass
from typing import Callable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class BisectionResult:
    root: float
    converged: bool
    certified: bool
    reason: str
    iterations: list[tuple[int, float, float, float, float, float]]
    bracket: tuple[float, float]
    error_bound: float
    residual: float
    required_steps: int


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    epsilon: float,
    *,
    function_tolerance: float = 1e-14,
    max_iter: int = 1000,
    continuity_verified: bool = False,
    continuity_samples: int = 257,
) -> BisectionResult:
    """Lõi chia đôi kiểm thử được; không tự nhận việc lấy mẫu là chứng minh liên tục."""
    if not all(math.isfinite(v) for v in (a, b, epsilon, function_tolerance)):
        raise ValueError("Các tham số phải là số hữu hạn.")
    if not a < b:
        raise ValueError("Phải có a < b.")
    if epsilon <= 0 or function_tolerance < 0 or max_iter <= 0:
        raise ValueError("epsilon, tolerance và max_iter phải hợp lệ.")

    def value(x: float) -> float:
        y = float(f(x))
        if not math.isfinite(y):
            raise ArithmeticError(f"f({x}) không hữu hạn; không được kết luận có nghiệm.")
        return y

    fa, fb = value(a), value(b)
    if abs(fa) <= function_tolerance:
        return BisectionResult(a, True, continuity_verified, "Nghiệm ở đầu mút a.", [], (a, a), 0.0, abs(fa), 0)
    if abs(fb) <= function_tolerance:
        return BisectionResult(b, True, continuity_verified, "Nghiệm ở đầu mút b.", [], (b, b), 0.0, abs(fb), 0)
    if fa * fb > 0:
        raise ValueError("f(a) và f(b) không trái dấu.")

    # Chỉ là bộ phát hiện điểm không xác định, không thay thế giả thiết liên tục.
    for i in range(max(2, continuity_samples)):
        value(a + (b - a) * i / (max(2, continuity_samples) - 1))

    ratio = (b - a) / (2.0 * epsilon)
    required = max(0, math.ceil(math.log2(ratio))) if ratio > 1.0 else 0
    records: list[tuple[int, float, float, float, float, float]] = []
    root = (a + b) / 2.0
    residual = abs(value(root))
    for k in range(max_iter + 1):
        root = (a + b) / 2.0
        fm = value(root)
        error = (b - a) / 2.0
        records.append((k, a, b, root, fm, error))
        residual = abs(fm)
        if residual <= function_tolerance:
            return BisectionResult(root, True, continuity_verified, "Trung điểm là nghiệm theo tolerance.", records, (root, root), 0.0, residual, required)
        if error <= epsilon:
            reason = "Đạt chặn sai số của khoảng chia đôi."
            if not continuity_verified:
                reason += " Kiểm tra số không thay thế chứng minh tính liên tục."
            return BisectionResult(root, True, continuity_verified, reason, records, (a, b), error, residual, required)
        if fa * fm < 0:
            b, fb = root, fm
        else:
            a, fa = root, fm
    return BisectionResult(root, False, False, "Đã đạt max_iter.", records, (a, b), (b - a) / 2.0, residual, required)


def print_custom_algorithm(
    f_expr,
    a_init,
    b_init,
    cond_type,
    target_val,
    final_n,
    final_xn,
    precision,
    G_name="",
):
    fmt = f".{precision}f"
    print("\n" + "=" * 40)
    print("MÔ TẢ THUẬT TOÁN ĐÃ SỬ DỤNG")
    print("=" * 40)
    print(f"1. Thiết lập hàm số: f(x) = {f_expr}")
    print(f"2. Khoảng phân ly ban đầu: [{a_init}, {b_init}]")

    if cond_type == "x_abs":
        print(f"3. Điều kiện dừng: Sai số tuyệt đối của x: \u0394x < {target_val}")
    elif cond_type == "x_rel":
        print(
            f"3. Điều kiện dừng: \u03b4x <= \u0394x/(|x_n|-\u0394x) < {target_val}, cần |x_n|>\u0394x"
        )
    elif cond_type == "G_abs":
        print(
            f"3. Điều kiện dừng: Sai số tuyệt đối của hàm {G_name}: \u0394{G_name} \u2264 {target_val}"
        )
    elif cond_type == "G_rel":
        print(
            f"3. Điều kiện dừng: B_G/(|{G_name}(x_n)|-B_G) \u2264 {target_val}, cần |{G_name}(x_n)|>B_G"
        )

    print("4. Tiến trình thực thi:")
    print("   - Lặp lại việc chia đôi khoảng cách ly hiện tại.")
    print(
        "   - Sau mỗi bước, chọn nửa khoảng mà tại đó hàm số đổi dấu (f(a)*f(x) < 0)."
    )
    print("   - Ở mỗi vòng lặp, tính sai số tương ứng để kiểm tra điều kiện dừng.")
    print("5. Kết luận:")
    print(f"   - Sau {final_n} bước lặp, điều kiện sai số đã được thỏa mãn.")
    print(f"   - Nghiệm gần đúng cuối cùng: x \u2248 {final_xn:{fmt}}")
    print("=" * 40)


def bisection_method(max_iter=10000):
    print("=== GIẢI PHƯƠNG TRÌNH & ĐÁNH GIÁ SAI SỐ ===\n")
    print("Input: f(x), khoảng phân ly [a,b], điều kiện dừng.")
    print("Output: bảng chia đôi, nghiệm gần đúng và chặn sai số.")
    print("Công thức: x_n=(a_n+b_n)/2; giữ nửa khoảng còn đổi dấu.\n")
    x, pi_sym, e_sym = sp.symbols("x pi e")

    # 1. Nhập phương trình gốc
    f_input = input(
        "Nhập phương trình f(x) = 0 (vd: 3*sin(x) + x**3 - 8*x**2 + 8*x + 1): "
    )

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
        G_expr_raw = sp.sympify(g_input, locals={"pi": pi_sym, "e": e_sym})

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
    f_expr = sp.sympify(f_input, locals={"pi": pi_val, "e": e_val})
    f = sp.lambdify(x, f_expr, "numpy")

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
    preview_choice = (
        input("\nBạn có muốn quét tìm trước khoảng phân ly chứa nghiệm không? (y/n): ")
        .strip()
        .lower()
    )
    if preview_choice == "y":
        print("Đang quét không gian [-50, 50]...")
        X_scan = np.linspace(-50, 50, 2000)
        Y_scan = f(X_scan)
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
                f"-> Gợi ý: Phát hiện nghiệm gần đúng quanh: {sorted(list(set(roots_preview)))}"
            )
        else:
            print("-> Không tìm thấy nghiệm thực trong khoảng này.")

    # --- NHẬP THÔNG SỐ LẶP ---
    try:
        precision = int(
            input("\nĐộ chính xác hiển thị (số chữ số sau dấu phẩy, vd: 8): ")
        )
    except (TypeError, ValueError):
        precision = 8

    a_orig = float(input("\nNhập đầu mút a: "))
    b_orig = float(input("Nhập đầu mút b: "))
    a, b = a_orig, b_orig

    if max_iter <= 0 or not (math.isfinite(a) and math.isfinite(b)) or not a < b:
        print("[X] Lỗi: cần a < b hữu hạn và max_iter > 0.")
        return
    fa, fb = float(f(a)), float(f(b))
    if not (math.isfinite(fa) and math.isfinite(fb)):
        print("[X] Lỗi: f(a), f(b) phải hữu hạn.")
        return
    if abs(fa) <= 1e-14:
        print(f"Nghiệm ở đầu mút: x = a = {a}.")
        return
    if abs(fb) <= 1e-14:
        print(f"Nghiệm ở đầu mút: x = b = {b}.")
        return
    if fa * fb > 0:
        print("[X] Lỗi: f(a) và f(b) phải trái dấu.")
        return

    # --- CHỌN ĐIỀU KIỆN DỪNG ---
    if use_G == "y":
        print(f"\nCHỌN ĐIỀU KIỆN DỪNG DỰA TRÊN HÀM {G_name}:")
        print(f"1. Sai số tuyệt đối (\u0394{G_name} \u2264 \u03b5)")
        print(
            f"2. Chặn tương đối (B_G/(|{G_name}(x_n)|-B_G) \u2264 \u03b5)"
        )
        stop_choice = input("Lựa chọn (1/2): ")
        target_epsilon = float(input("Nhập \u03b5 mục tiêu (vd: 0.5e-3): "))
        cond_type = "G_abs" if stop_choice == "1" else "G_rel"
    else:
        print("\nCHỌN ĐIỀU KIỆN DỪNG THEO x:")
        print("1. Sai số tuyệt đối (\u0394x \u2264 \u03b5)")
        print("2. Chặn tương đối (\u03b4x \u2264 \u0394x/(|x_n|-\u0394x) \u2264 \u03b5)")
        print("3. Theo số chữ số đáng tin của x (tuyệt đối)")
        stop_choice = input("Lựa chọn (1/2/3): ")
        if stop_choice == "1":
            target_epsilon = float(input("Nhập sai số tuyệt đối epsilon = "))
            cond_type = "x_abs"
        elif stop_choice == "2":
            target_epsilon = float(input("Nhập sai số tương đối epsilon = "))
            cond_type = "x_rel"
        else:
            k = int(input("Nhập số chữ số đáng tin: "))
            target_epsilon = 0.5 * (10 ** (-k))
            print(f"-> Epsilon tuyệt đối tương đương: {target_epsilon}")
            cond_type = "x_abs"

    # --- QUÁ TRÌNH LẶP ---
    print("\n--- BẢNG QUÁ TRÌNH LẶP ---")
    if use_G == "y":
        header = f"{'n':<3} | {'a':<12} | {'b':<12} | {'d':<12} | {'Dấu f(d)':<8} | {'E_x':<12} | {G_name:<12} | {'B_G/XP_G':<12}"
    else:
        # Cập nhật header cho trường hợp tính theo x
        header = f"{'n':<3} | {'a':<12} | {'b':<12} | {'xn':<12} | {'Dấu f(xn)':<9} | {'\u0394x (Tuyệt đối)':<16} | {'\u03b4x (Tương đối)':<16}"

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    n = 0
    final_xn = 0

    if not math.isfinite(target_epsilon) or target_epsilon <= 0:
        print("[X] Lỗi: epsilon phải dương và hữu hạn.")
        return
    print("Chặn sai số trung điểm: |x_n-x*| <= (b-a)/2^(n+1).")
    if cond_type == "x_abs":
        required_steps = max(
            0, math.ceil(math.log2((b - a) / (2 * target_epsilon)))
        ) if (b - a) > 2 * target_epsilon else 0
        print(f"Số bước tối thiểu theo ngưỡng của x: n >= {required_steps}.")
    elif cond_type == "x_rel":
        print("Dừng tương đối: chưa thể suy số bước chỉ từ epsilon nếu chưa có chặn dưới dương cho |x*|.")
    else:
        print("Dừng theo G(x): số bước phải suy từ chặn truyền sai số của G, không dùng trực tiếp epsilon_G cho x.")

    while True:
        if n >= max_iter:
            print(f"[X] Dừng tại max_iter = {max_iter}; chưa chứng nhận đạt sai số.")
            return
        xn = (a + b) / 2
        fxn = float(f(xn))
        if not math.isfinite(fxn):
            print("[X] f tại trung điểm không hữu hạn; không được kết luận có nghiệm.")
            return
        delta_x = abs(b - a) / 2

        # Tính sai số tương đối của x (tránh lỗi chia cho 0)
        delta_x_rel = (
            delta_x / (abs(xn) - delta_x)
            if abs(xn) > delta_x
            else float("inf")
        )

        # Đánh giá điều kiện dừng & In bảng
        if use_G == "y":
            G_val = G_func(xn)
            interval_left = max(a_orig, xn - delta_x)
            interval_right = min(b_orig, xn + delta_x)
            M_G = symbolic_derivative_bound(
                G_eval_expr, x, interval_left, interval_right
            )
            G_bound_certified = (
                M_G is not None and delta_pi == 0.0 and delta_e == 0.0
            )
            if M_G is not None:
                propagated = propagate_bound(
                    G_func,
                    G_prime_func,
                    xn,
                    delta_x,
                    (a_orig, b_orig),
                    derivative_bound=M_G,
                    derivative_bound_verified=G_bound_certified,
                )
                delta_G_val = propagated.absolute_bound
                delta_G_rel = propagated.relative_bound
            else:
                delta_G_val = float(delta_G_func(xn, delta_x))
                delta_G_rel = None
                G_bound_certified = False

            dau_f = "+" if fxn > 0 else "-"
            print(
                f"{n:<3} | {a:<12.{precision}f} | {b:<12.{precision}f} | {xn:<12.{precision}f} | {dau_f:^8} | {delta_x:<12.{precision}e} | {G_val:<12.{precision}f} | {delta_G_val:<12.{precision}e}"
            )

            if cond_type == "G_abs" and G_bound_certified and delta_G_val <= target_epsilon:
                final_xn = xn
                break
            elif cond_type == "G_rel" and G_bound_certified and delta_G_rel is not None and delta_G_rel <= target_epsilon:
                final_xn = xn
                break
            elif not G_bound_certified:
                print("    ΔG chỉ là xấp xỉ bậc nhất/kiểm tra số; không dùng để chứng nhận dừng.")
        else:
            dau_f = "+" if fxn > 0 else "-"
            print(
                f"{n:<3} | {a:<12.{precision}f} | {b:<12.{precision}f} | {xn:<12.{precision}f} | {dau_f:^9} | {delta_x:<16.{precision}e} | {delta_x_rel:<16.{precision}e}"
            )

            if cond_type == "x_abs" and delta_x < target_epsilon:
                final_xn = xn
                break
            elif cond_type == "x_rel" and delta_x_rel < target_epsilon:
                final_xn = xn
                break

        # Cập nhật mút
        if fa * fxn < 0:
            b = xn
            fb = fxn
        elif fb * fxn < 0:
            a = xn
            fa = fxn
        else:  # f(xn) == 0
            final_xn = xn
            break

        n += 1

    print("-" * len(header))
    print(f"=> Quá trình lặp hoàn tất tại bước {n}.")
    print(f"Khoảng cuối cùng chứa nghiệm: [{a}, {b}].")
    print(f"|f(x)| = {abs(float(f(final_xn))):.{precision}e}.")
    print("Lưu ý: tính liên tục của f trên đoạn vẫn là giả thiết người dùng phải kiểm tra.")

    # In thuật toán theo yêu cầu
    print_algo = (
        input("\nIn phần thuật toán để chép bài? [C/k]: ")
        .strip()
        .lower()
    )
    if print_algo in {"", "c", "co", "có", "y", "yes"}:
        print_custom_algorithm(
            f_input,
            a_orig,
            b_orig,
            cond_type,
            target_epsilon,
            n,
            final_xn,
            precision,
            G_name,
        )


if __name__ == "__main__":
    bisection_method()
