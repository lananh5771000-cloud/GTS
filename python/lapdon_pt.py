import sympy as sp
import numpy as np
import math
import sys
from exam_format import exam_print as print
from dataclasses import dataclass
from typing import Callable
from input_utils import parse_math_expression, parse_real

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class FixedPointResult:
    root: float
    converged: bool
    certified: bool
    reason: str
    q: float
    mapping_ok: bool
    iterations: list[tuple[int, float, float, float | None, float | None]]
    error_bound: float | None
    relative_error_bound: float | None
    residual: float
    status: str = "unknown"
    left_domain: bool = False
    max_iter_reached: bool = False
    invalid_contraction: bool = False
    numerical_failure: bool = False


@dataclass
class FunctionErrorBound:
    value: float
    bound: float | None
    derivative_bound: float | None
    interval: tuple[float, float]
    rigorous: bool
    description: str


def priori_iteration_count(q: float, epsilon: float, first_difference: float) -> int:
    """n=ceil(log_q(epsilon*(1-q)/|phi(x0)-x0|)) cho sai so tien nghiem."""
    if not all(math.isfinite(v) for v in (q, epsilon, first_difference)):
        raise ValueError("q, epsilon va first_difference phai huu han.")
    if not (0 <= q < 1) or epsilon <= 0 or first_difference < 0:
        raise ValueError("Can 0<=q<1, epsilon>0 va first_difference>=0.")
    if first_difference == 0 or q == 0:
        return 1
    ratio = epsilon * (1 - q) / first_difference
    if ratio >= 1:
        return 1
    return max(1, math.ceil(math.log(ratio) / math.log(q)))


def posteriori_error_bound(q: float, difference: float) -> float | None:
    """Delta=q/(1-q)*|x_k-x_{k-1}| cho sai so hau nghiem."""
    if not math.isfinite(q) or q < 0 or q >= 1:
        return None
    return q / (1 - q) * abs(difference)


def bound_function_error(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    x_approx: float,
    error_bound_x: float,
    domain: tuple[float, float],
    *,
    derivative_bound: float | None = None,
    derivative_bound_verified: bool = False,
    samples: int = 2001,
) -> FunctionErrorBound:
    """Truyền chặn qua G trên toàn I=[x-E,x+E] giao với miền xét."""
    a, b = domain
    if not all(math.isfinite(v) for v in (a, b, x_approx, error_bound_x)):
        raise ValueError("Dữ liệu truyền sai số phải hữu hạn.")
    if not a <= x_approx <= b or error_bound_x < 0 or samples < 2:
        raise ValueError("Miền, xấp xỉ hoặc chặn sai số không hợp lệ.")
    left, right = max(a, x_approx - error_bound_x), min(b, x_approx + error_bound_x)
    value = float(function(x_approx))
    if not math.isfinite(value):
        raise ArithmeticError("G(xấp xỉ) không hữu hạn.")
    if derivative_bound is None:
        slopes = [
            abs(float(derivative(left + (right - left) * i / (samples - 1))))
            for i in range(samples)
        ]
        if not all(math.isfinite(slope) for slope in slopes):
            raise ArithmeticError("G' không hữu hạn trên khoảng sai số.")
        M = max(slopes, default=0.0)
        rigorous = False
        description = "Ước lượng M_G bằng lưới; không phải chặn giải tích được chứng minh."
    else:
        M = float(derivative_bound)
        if not math.isfinite(M) or M < 0:
            raise ValueError("Chặn M_G phải không âm và hữu hạn.")
        rigorous = derivative_bound_verified
        description = (
            "Chặn M_G đã được xác nhận trên toàn khoảng."
            if rigorous
            else "M_G chưa được chứng minh trên toàn khoảng; chỉ dùng tham khảo."
        )
    return FunctionErrorBound(
        value=value,
        bound=M * error_bound_x,
        derivative_bound=M,
        interval=(left, right),
        rigorous=rigorous,
        description=description,
    )


def symbolic_derivative_bound(
    expression: sp.Expr,
    symbol: sp.Symbol,
    left: float,
    right: float,
) -> float | None:
    """Cố gắng lấy max |G'| bằng SymPy; None nghĩa là chưa chứng minh được."""
    try:
        derivative = sp.diff(expression, symbol)
        domain = sp.Interval(left, right)
        lower = sp.minimum(derivative, symbol, domain)
        upper = sp.maximum(derivative, symbol, domain)
        value = max(abs(float(sp.N(lower, 17))), abs(float(sp.N(upper, 17))))
    except (ValueError, TypeError, NotImplementedError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def fixed_point(
    phi: Callable[[float], float],
    derivative: Callable[[float], float],
    a: float,
    b: float,
    x0: float,
    epsilon: float,
    *,
    relative: bool = False,
    max_iter: int = 1000,
    samples: int = 2001,
    analytic_conditions_verified: bool = False,
    fixed_iterations: int | None = None,
) -> FixedPointResult:
    """Lặp đơn với chặn hậu nghiệm Banach và nhãn chứng nhận trung thực."""
    if not all(math.isfinite(v) for v in (a, b, x0, epsilon)):
        raise ValueError("Dữ liệu phải hữu hạn.")
    if not a < b or not a <= x0 <= b:
        raise ValueError("Cần a < b và x0 thuộc [a,b].")
    if epsilon <= 0 or max_iter <= 0 or samples < 2:
        raise ValueError("epsilon, max_iter và số mẫu phải hợp lệ.")

    if fixed_iterations is not None and fixed_iterations < 0:
        raise ValueError("fixed_iterations phai khong am.")

    grid = [a + (b - a) * i / (samples - 1) for i in range(samples)]
    images: list[float] = []
    derivatives: list[float] = []
    for point in grid:
        image, slope = float(phi(point)), float(derivative(point))
        if not math.isfinite(image) or not math.isfinite(slope):
            raise ArithmeticError("phi hoặc phi' không hữu hạn trên lưới kiểm tra.")
        images.append(image)
        derivatives.append(abs(slope))
    mapping_ok = min(images) >= a and max(images) <= b
    q = max(derivatives)
    contraction_sampled = mapping_ok and q < 1.0
    certified_conditions = contraction_sampled and analytic_conditions_verified

    x = x0
    rows: list[tuple[int, float, float, float | None, float | None]] = [(0, x, abs(phi(x) - x), None, None)]
    last_error = last_relative = None
    if fixed_iterations == 0:
        residual0 = abs(float(phi(x)) - x)
        return FixedPointResult(
            x, True, False,
            "Da thuc hien dung k=0 buoc lap don; chua dung tieu chuan epsilon de chung nhan.",
            q, mapping_ok, rows, None, None, residual0,
            status="fixed_steps", invalid_contraction=not contraction_sampled,
        )
    loop_limit = fixed_iterations if fixed_iterations is not None else max_iter
    for k in range(1, loop_limit + 1):
        new = float(phi(x))
        if not math.isfinite(new):
            return FixedPointResult(
                x, False, False, "Giá trị lặp không hữu hạn.", q, mapping_ok,
                rows, last_error, last_relative, math.inf,
                status="numerical_failure", numerical_failure=True,
                invalid_contraction=not contraction_sampled,
            )
        if not a <= new <= b:
            return FixedPointResult(
                new, False, False,
                f"Phép lặp rời khỏi [a,b] tại bước {k}; không có chặn sai số được chứng nhận.",
                q, mapping_ok, rows, None, None,
                abs(float(phi(new)) - new) if math.isfinite(new) else math.inf,
                status="left_domain", left_domain=True,
                invalid_contraction=not contraction_sampled,
            )
        difference = abs(new - x)
        residual = abs(float(phi(new)) - new)
        error = q / (1.0 - q) * difference if contraction_sampled else None
        relative_error = (
            error / (abs(new) - error)
            if error is not None and abs(new) > error
            else None
        )
        rows.append((k, new, difference, error, relative_error))
        x, last_error, last_relative = new, error, relative_error
        meets = (
            relative_error is not None and relative_error <= epsilon
            if relative
            else error is not None and error <= epsilon
        )
        if fixed_iterations is not None:
            if residual == 0.0:
                return FixedPointResult(
                    x, True, True, "Da gap diem bat dong chinh xac.", q, mapping_ok,
                    rows, error, relative_error, residual, status="exact_solution",
                )
            if k >= fixed_iterations:
                reason = (
                    f"Da thuc hien dung k={fixed_iterations} buoc lap don; "
                    "chua dung tieu chuan epsilon de chung nhan."
                )
                return FixedPointResult(
                    x, True, certified_conditions and error is not None and error <= epsilon,
                    reason, q, mapping_ok, rows, error, relative_error, residual,
                    status="fixed_steps", invalid_contraction=not contraction_sampled,
                )
            continue
        if meets:
            if certified_conditions:
                reason = "Đạt chặn sai số Banach; kết quả được chứng nhận."
                certified = True
            else:
                reason = "Kiểm tra số trên lưới chưa phát hiện vi phạm, không thay thế chứng minh giải tích."
                certified = False
            status = "certified" if certified else "numerical_only"
            return FixedPointResult(
                x, True, certified, reason, q, mapping_ok, rows, error,
                relative_error, residual, status=status,
            )
        if not contraction_sampled and residual <= 100.0 * sys.float_info.epsilon:
            return FixedPointResult(
                x, True, False,
                "Đã đạt điểm bất động theo phần dư số; không có chứng nhận Banach.",
                q, mapping_ok, rows, None, None, residual,
                status="numerical_only", invalid_contraction=True,
            )
    residual = abs(float(phi(x)) - x)
    status = "invalid_contraction" if not contraction_sampled else "max_iter_reached"
    return FixedPointResult(
        x, False, False,
        "Điều kiện ánh xạ co không được thỏa mãn; đã đạt max_iter."
        if not contraction_sampled else "Đã đạt max_iter.",
        q, mapping_ok, rows, last_error, last_relative, residual,
        status=status, max_iter_reached=True,
        invalid_contraction=not contraction_sampled,
    )


def fixed_point_priori(
    phi: Callable[[float], float],
    derivative: Callable[[float], float],
    a: float,
    b: float,
    x0: float,
    epsilon: float,
    **kwargs,
) -> FixedPointResult:
    first_difference = abs(float(phi(x0)) - x0)
    probe = fixed_point(
        phi,
        derivative,
        a,
        b,
        x0,
        epsilon,
        fixed_iterations=1,
        **kwargs,
    )
    steps = priori_iteration_count(probe.q, epsilon, first_difference)
    return fixed_point(
        phi,
        derivative,
        a,
        b,
        x0,
        epsilon,
        fixed_iterations=steps,
        **kwargs,
    )


def print_custom_algorithm(
    phi_expr,
    a,
    b,
    x0,
    q,
    cond_type,
    target_val,
    final_n,
    final_xn,
    precision,
    G_name="",
):
    fmt = f".{precision}f"
    print("\n" + "=" * 50)
    print("MÔ TẢ THUẬT TOÁN LẶP ĐƠN ĐÃ SỬ DỤNG")
    print("=" * 50)
    print(f"1. Thiết lập hàm số: x = \u03c6(x) = {phi_expr}")
    print(f"2. Khoảng phân ly ban đầu: [{a}, {b}]")
    print("3. Khảo sát điều kiện hội tụ:")
    print(f"   - Hệ số co q = max|\u03c6'(x)| trên [{a}, {b}] = {q:{fmt}}")
    if q < 1:
        print("   - q < 1 mới là một điều kiện; còn phải kiểm tra phi([a,b]) nằm trong [a,b].")
    else:
        print("   - [Cảnh báo] q \u2265 1, quá trình lặp có thể phân kỳ.")
    print(f"4. Điểm xuất phát: x0 = {x0}")

    if cond_type == "x_abs":
        print("5. Điều kiện dừng: Sai số tuyệt đối của x (Hậu nghiệm):")
        if q < 1:
            print(f"   \u0394x = [q / (1-q)] * |x_n - x_{{n-1}}| \u2264 {target_val}")
        else:
            print("   Chỉ theo dõi |x_n-x_{n-1}|; không coi đây là chặn sai số nghiệm.")
    elif cond_type == "priori":
        print("5. Điều kiện dừng: Sai số tiên nghiệm của x:")
        print("   n = ceil(log_q(\u03b5(1-q)/|\u03c6(x0)-x0|))")
        print(f"   Thực hiện đúng n = {int(target_val)} bước lặp.")
    elif cond_type == "x_rel":
        print("5. Điều kiện dừng: Sai số tương đối của x:")
        print(f"   \u03b4x \u2264 \u0394x / (|x_n|-\u0394x) \u2264 {target_val}, cần |x_n|>\u0394x")
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
    print("   - Sử dụng công thức: xₙ₊₁ = φ(xₙ)")
    print("7. Kết luận:")
    print(f"   - Sau {final_n} bước lặp, tiêu chuẩn số đã chọn được thỏa mãn.")
    print(f"   - Nghiệm gần đúng cuối cùng: x \u2248 {final_xn:{fmt}}")
    print("=" * 50)


def fixed_point_iteration(max_iter=10000):
    print("=== GIẢI PHƯƠNG TRÌNH BẰNG PHƯƠNG PHÁP LẶP ĐƠN (1 BIẾN) ===\n")
    print("Input: φ(x), khoảng xét, x₀ và điều kiện dừng.")
    print("Output: hệ số co q, bảng lặp, nghiệm gần đúng và chặn sai số.")
    print("Công thức: x⁽ᵏ⁺¹⁾ = φ(x⁽ᵏ⁾).\n")
    x, pi_sym, e_sym = sp.symbols("x pi e")

    # 1. Nhập phương trình phi(x)
    print("Để sử dụng lặp đơn, bạn cần biến đổi f(x) = 0 về dạng x = \u03c6(x).")
    phi_input = input("Nhập hàm \u03c6(x) (vd: (x+1)**(1/3) hoặc cbrt(x+1)): ")

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
    combined_input = phi_input + " " + g_input
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

    # 4. Thay số và thiết lập hàm phi(x)
    try:
        phi_expr = parse_math_expression(phi_input, {"x": x, "pi": sp.Float(pi_val), "e": sp.Float(e_val)})
        phi_prime = sp.diff(phi_expr, x)

        phi_prime_func = sp.lambdify(x, phi_prime, "numpy")
    except Exception as e:
        print("Lỗi cú pháp hàm số:", e)
        return

    # 5. Xây dựng công thức truyền sai số cho G(x)
    G_func = None
    delta_G_func = None
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
        input(
            "\nBạn có muốn quét tìm trước các nghiệm f(x) = x - \u03c6(x) = 0 không? (y/n): "
        )
        .strip()
        .lower()
    )
    if preview_choice == "y":
        print("Đang quét không gian [-50, 50]...")
        f_expr = x - phi_expr
        f_func = sp.lambdify(x, f_expr, "numpy")
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

    # --- BƯỚC 1: KHOẢNG CÁCH LY & HỆ SỐ CO ---
    print("\n--- BƯỚC 1: KHOẢNG CÁCH LY VÀ HỆ SỐ CO q ---")
    a = parse_real(input("Nhập cận dưới a = "))
    b = parse_real(input("Nhập cận trên b = "))
    if max_iter <= 0 or not (math.isfinite(a) and math.isfinite(b)) or not a < b:
        print("[X] Cần a < b hữu hạn và max_iter > 0.")
        return

    print(f"\nĐạo hàm \u03c6'(x) = {phi_prime}")

    X_vals = np.linspace(a, b, 2000)
    Y_prime = np.abs(phi_prime_func(X_vals))
    q = float(np.max(Y_prime))
    phi_values = np.array([float(phi_expr.subs(x, point)) for point in X_vals])
    mapping_ok = bool(
        np.all(np.isfinite(phi_values))
        and np.min(phi_values) >= a
        and np.max(phi_values) <= b
    )
    print("Kiểm tra số trên lưới chưa phát hiện vi phạm, không thay thế chứng minh giải tích.")

    print(f"-> Hệ số co q = max|\u03c6'(x)| trên [{a}, {b}] = {q:{fmt}}")

    if q < 1 and mapping_ok:
        print("=> Trên lưới: phi([a,b]) nằm trong [a,b] và q < 1; chưa phải chứng minh giải tích.")
    else:
        print("=> [!] Không đạt đồng thời điều kiện bất biến miền và q < 1; không chứng nhận hội tụ.")

    # --- BƯỚC 2: CHỌN ĐIỂM XUẤT PHÁT ---
    print("\n--- BƯỚC 2: CHỌN ĐIỂM XUẤT PHÁT x0 ---")
    x0 = parse_real(input(f"Nhập điểm xuất phát x0 nằm trong [{a}, {b}]: "))
    if not (a <= x0 <= b):
        print("[X] x0 nằm ngoài miền; dừng vì không có bảo đảm hội tụ.")
        return

    # --- BƯỚC 3: ĐIỀU KIỆN DỪNG ---
    print("\n--- BƯỚC 3: CHỌN ĐIỀU KIỆN DỪNG ---")
    target_val = 0
    cond_type = ""

    print("1. Sai số tiên nghiệm của x")
    print("2. Sai số hậu nghiệm của x")
    print("3. Đúng k bước")
    print("4. Sai số tương đối của x")
    choice = input("Lựa chọn (1/2/3/4): ")
    if choice == "1":
        target_val = parse_real(input("Nhập epsilon tiên nghiệm = "))
        cond_type = "priori"
        if q < 1 and q > 0:
            x1_temp = float(phi_expr.subs(x, x0))
            first_difference = abs(x1_temp - x0)
            if first_difference > 0:
                n_estimate = priori_iteration_count(q, target_val, first_difference)
                print(
                    f"\n[Công thức tiên nghiệm] n = ceil(log_q(epsilon(1-q)/|phi(x0)-x0|)) = {n_estimate}."
                )
                target_val = n_estimate
            else:
                print("\n[Công thức tiên nghiệm] x0 đã là điểm bất động, lấy n = 1.")
                target_val = 1
        else:
            print("\n[Cảnh báo] q không thuộc (0,1), công thức tiên nghiệm chỉ mang tính tham khảo.")
            target_val = max_iter
    elif choice == "2":
        target_val = parse_real(input("Nhập sai số hậu nghiệm epsilon = "))
        cond_type = "x_abs"
    elif choice == "3":
        target_val = int(input("Nhập số lần lặp k = "))
        cond_type = "iter"
    else:
        target_val = parse_real(input("Nhập sai số tương đối epsilon = "))
        cond_type = "x_rel"

    if cond_type != "iter" and (not math.isfinite(target_val) or target_val <= 0):
        print("[X] epsilon phải dương và hữu hạn.")
        return

    # --- BƯỚC 4: QUÁ TRÌNH LẶP ---
    print("\n--- BƯỚC 4: QUÁ TRÌNH LẶP ---")

    if use_G == "y":
        header = f"{'n':<3} | {'xₙ':<15} | {'Eₓ (nếu co)':<18} | {G_name:<15} | {'Bᴳ/XPᴳ':<15} | {'Bᴳ tương đối':<15}"
    else:
        header = f"{'n':<3} | {'xₙ':<15} | {'Tiên nghiệm (Δx)':<18} | {'Hậu nghiệm (Δx)':<18} | {'Tương đối (δx)':<18}"

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    xn = x0
    x_prev = None
    x1_val = float(phi_expr.subs(x, x0))
    diff_x1_x0 = abs(x1_val - x0)
    n = 0

    while True:
        if n >= max_iter:
            print(f"[X] Dừng tại max_iter = {max_iter}; chưa chứng nhận đạt sai số.")
            return
        # Tính sai số cho x
        if n == 0:
            str_dx_pri = "-"
            str_dx_post = "-"
            str_dx_rel = "-"
            delta_x = float("inf")
            delta_x_rel = float("inf")
        else:
            if q < 1 and mapping_ok:
                err_pri = ((q**n) / (1 - q)) * diff_x1_x0
                delta_x = (q / (1 - q)) * abs(xn - x_prev)
                str_dx_pri = f"{err_pri:<18.{precision}e}"
            else:
                str_dx_pri = "N/A (không co)"
                delta_x = float("inf")

            delta_x_rel = (
                delta_x / (abs(xn) - delta_x)
                if q < 1 and mapping_ok and abs(xn) > delta_x
                else float("inf")
            )
            str_dx_post = (
                f"{delta_x:<18.{precision}e}"
                if math.isfinite(delta_x)
                else "N/A (không co)"
            )
            str_dx_rel = f"{delta_x_rel:<18.{precision}e}"

        # In bảng và kiểm tra điều kiện
        if use_G == "y":
            G_val = float(G_func(xn))
            G_bound_certified = False
            if n > 0 and math.isfinite(delta_x):
                interval_left = max(a, xn - delta_x)
                interval_right = min(b, xn + delta_x)
                M_G = symbolic_derivative_bound(
                    G_eval_expr, x, interval_left, interval_right
                )
                if M_G is not None and delta_pi == 0.0 and delta_e == 0.0:
                    delta_G_val = M_G * delta_x
                    G_bound_certified = True
                else:
                    delta_G_val = float(delta_G_func(xn, delta_x))
                delta_G_rel = (
                    delta_G_val / (abs(G_val) - delta_G_val)
                    if abs(G_val) > delta_G_val
                    else float("inf")
                )
                str_dG = f"{delta_G_val:<15.{precision}e}"
                str_dG_rel = f"{delta_G_rel:<15.{precision}e}"
            else:
                delta_G_val = float("inf")
                delta_G_rel = float("inf")
                str_dG = "-"
                str_dG_rel = "-"

            print(
                f"{n:<3} | {xn:<15.{precision}f} | {str_dx_post} | {G_val:<15.{precision}f} | {str_dG} | {str_dG_rel}"
            )

            if n > 0:
                if cond_type == "priori" and n >= target_val:
                    break
                elif cond_type == "iter" and n >= target_val:
                    break
                elif cond_type == "x_abs" and q < 1 and mapping_ok and delta_x <= target_val:
                    break
                elif cond_type == "x_rel" and q < 1 and mapping_ok and delta_x_rel <= target_val:
                    break
        else:
            print(
                f"{n:<3} | {xn:<15.{precision}f} | {str_dx_pri} | {str_dx_post} | {str_dx_rel}"
            )

            if n > 0:
                if cond_type == "priori" and n >= target_val:
                    break
                elif cond_type == "iter" and n >= target_val:
                    break
                elif cond_type == "x_abs" and q < 1 and mapping_ok and delta_x <= target_val:
                    break
                elif cond_type == "x_rel" and q < 1 and mapping_ok and delta_x_rel <= target_val:
                    break

        # Xử lý vòng lặp 0 khi chọn iter=0
        if n == 0 and cond_type == "iter" and n >= target_val:
            break

        # Tính bước tiếp theo
        try:
            x_next = float(phi_expr.subs(x, xn))
        except Exception as e:
            print(
                f"\n[X] Lỗi tính toán ở bước {n + 1}: Hàm \u03c6(x) không xác định tại x = {xn}. ({e})"
            )
            break

        if not math.isfinite(x_next):
            print("[X] Giá trị lặp không hữu hạn; dừng.")
            return
        if not a <= x_next <= b:
            print("[X] Phép lặp rời khỏi [a,b]; không còn bảo đảm hội tụ.")
            return

        x_prev = xn
        xn = x_next
        n += 1

    print("-" * len(header))
    print(f"=> Quá trình lặp hoàn tất tại bước {n}.")

    # In thuật toán
    try:
        print_algo = (
            input("\nIn phần thuật toán để chép bài? [C/k]: ")
            .strip()
            .lower()
        )
    except (EOFError, StopIteration):
        print_algo = "n"
    if print_algo in {"", "c", "co", "có", "y", "yes"}:
        print_custom_algorithm(
            phi_input, a, b, x0, q, cond_type, target_val, n, xn, precision, G_name
        )


if __name__ == "__main__":
    try:
        fixed_point_iteration()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã dừng chương trình; không có dữ liệu đầu vào đầy đủ.")
    except Exception as error:
        print(f"\nKhông thể thực hiện: {error}")
