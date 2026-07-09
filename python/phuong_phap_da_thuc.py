import sys
from exam_format import exam_print as print
import math
from fractions import Fraction
from numbers import Integral, Real
import sympy as sp
from input_utils import MathInputError, parse_exact, parse_real, split_number_row


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# NHẬP DỮ LIỆU
# ============================================================


def input_nonnegative_integer(prompt, default=None):
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


def input_positive_number(prompt, default=None):
    while True:
        token = input(prompt).strip()
        if token == "" and default is not None:
            return default
        try:
            try:
                value = parse_real(token)
            except ValueError:
                value = float(token)

            if not math.isfinite(value) or value <= 0:
                raise ValueError

            return value
        except (ValueError, ZeroDivisionError):
            print("Lỗi: Vui lòng nhập số dương hợp lệ, ví dụ 1e-8, 0.0001 hoặc 1/1000.")


def input_coefficients(degree):
    expected = degree + 1

    print(
        f"\nNhập {expected} hệ số a_0, a_1, ..., a_{degree} "
        "theo thứ tự từ bậc cao xuống bậc thấp."
    )
    print("Có thể nhập số nguyên, số thập phân hoặc phân số.")

    while True:
        try:
            tokens = split_number_row(input("Các hệ số cách nhau bởi khoảng trắng: "), expected)
            return [parse_exact(token) for token in tokens]
        except (MathInputError, ValueError, ZeroDivisionError):
            print("Lỗi: Hệ số không hợp lệ. Ví dụ hợp lệ: 2, -3, 0.25, 1/3.")


def choose_output_mode():
    print("\nChọn cách in quá trình chia đôi:")
    print("1. In gọn, đủ nội dung để chép bài thi")
    print("2. In toàn bộ bảng lặp chia đôi")

    while True:
        choice = input("Chọn [Enter = 1]: ").strip() or "1"
        if choice in {"1", "2"}:
            return choice == "2"
        print("Lỗi: Vui lòng chỉ chọn 1 hoặc 2.")


def minimum_display_decimals(epsilon):
    """Số chữ số thập phân để sai số làm tròn không quá epsilon/2."""
    if epsilon >= 1.0:
        return 0
    return max(0, math.ceil(-math.log10(epsilon)))


# ============================================================
# PHÉP TOÁN ĐA THỨC CHÍNH XÁC BẰNG FRACTION
# Hệ số lưu theo thứ tự số mũ giảm dần.
# ============================================================


def strip_leading_zeros(coefficients):
    coefficients = [normalize_coefficient(value) for value in coefficients]
    while len(coefficients) > 1 and coefficients[0] == 0:
        coefficients.pop(0)
    return coefficients


def normalize_coefficient(value):
    """Chuẩn hóa mọi kiểu hệ số ngay tại biên của các hàm lõi."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Hệ số không được là NaN hoặc vô cùng.")
        return Fraction(str(number))
    expression = sp.sympify(value)
    if expression.is_real is not True or expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise ValueError("Hệ số phải là số thực hữu hạn.")
    if expression.is_Rational:
        return Fraction(int(expression.p), int(expression.q))
    return sp.simplify(expression)


def is_zero_polynomial(coefficients):
    coefficients = strip_leading_zeros(coefficients)
    return len(coefficients) == 1 and coefficients[0] == 0


def polynomial_degree(coefficients):
    return len(strip_leading_zeros(coefficients)) - 1


def polynomial_derivative(coefficients):
    coefficients = strip_leading_zeros(coefficients)
    degree = len(coefficients) - 1

    if degree <= 0:
        return [Fraction(0)]

    return [coefficients[i] * Fraction(degree - i) for i in range(degree)]


def polynomial_monic(coefficients):
    coefficients = strip_leading_zeros(coefficients)

    if is_zero_polynomial(coefficients):
        return [Fraction(0)]

    leading = coefficients[0]
    return [value / leading for value in coefficients]


def polynomial_multiply(left, right):
    left = strip_leading_zeros(left)
    right = strip_leading_zeros(right)

    result = [Fraction(0) for _ in range(len(left) + len(right) - 1)]

    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            result[i + j] += left_value * right_value

    return strip_leading_zeros(result)


def polynomial_power(coefficients, exponent):
    result = [Fraction(1)]

    for _ in range(exponent):
        result = polynomial_multiply(result, coefficients)

    return result


def polynomial_divmod(dividend, divisor):
    dividend = strip_leading_zeros(dividend)
    divisor = strip_leading_zeros(divisor)

    if is_zero_polynomial(divisor):
        raise ZeroDivisionError("Không thể chia cho đa thức 0.")

    dividend_degree = polynomial_degree(dividend)
    divisor_degree = polynomial_degree(divisor)

    if dividend_degree < divisor_degree:
        return [Fraction(0)], dividend

    quotient_degree = dividend_degree - divisor_degree
    quotient = [Fraction(0)] * (quotient_degree + 1)
    remainder = dividend[:]

    while (
        not is_zero_polynomial(remainder)
        and polynomial_degree(remainder) >= divisor_degree
    ):
        current_degree = polynomial_degree(remainder)
        shift = current_degree - divisor_degree
        factor = remainder[0] / divisor[0]

        quotient[quotient_degree - shift] += factor

        subtractor = [factor * value for value in divisor] + [Fraction(0)] * shift

        if len(subtractor) < len(remainder):
            subtractor = [Fraction(0)] * (len(remainder) - len(subtractor)) + subtractor

        remainder = strip_leading_zeros(
            [remainder[i] - subtractor[i] for i in range(len(remainder))]
        )

    return strip_leading_zeros(quotient), strip_leading_zeros(remainder)


def polynomial_exact_divide(dividend, divisor):
    quotient, remainder = polynomial_divmod(dividend, divisor)

    if not is_zero_polynomial(remainder):
        raise ArithmeticError("Phép chia đa thức đáng lẽ phải có dư bằng 0.")

    return quotient


def polynomial_gcd(left, right):
    left = strip_leading_zeros(left)
    right = strip_leading_zeros(right)

    while not is_zero_polynomial(right):
        _, remainder = polynomial_divmod(left, right)
        left, right = right, remainder

    return polynomial_monic(left)


def square_free_decomposition(coefficients):
    """
    Phân tích:
        P(x)/a0 = Q1(x)^1 * Q2(x)^2 * ...
    """
    polynomial = polynomial_monic(coefficients)

    if polynomial_degree(polynomial) <= 0:
        return []

    derivative = polynomial_derivative(polynomial)
    common = polynomial_gcd(polynomial, derivative)
    remaining = polynomial_exact_divide(polynomial, common)

    factors = []
    multiplicity = 1

    while polynomial_degree(remaining) > 0:
        shared = polynomial_gcd(remaining, common)
        factor = polynomial_exact_divide(remaining, shared)

        if polynomial_degree(factor) > 0:
            factors.append((polynomial_monic(factor), multiplicity))

        remaining = shared
        common = polynomial_exact_divide(common, shared)
        multiplicity += 1

    return factors


# ============================================================
# HOOCNE, GIÁ TRỊ VÀ MIỀN CHỨA NGHIỆM
# ============================================================


def horner_value(coefficients, x):
    result = 0.0

    for coefficient in coefficients:
        result = result * x + float(coefficient)

    return result


def horner_value_exact(coefficients, x):
    """Tính P(x) chính xác khi x và hệ số là Fraction."""
    x = sp.sympify(x)
    result = sp.S.Zero

    for coefficient in coefficients:
        result = result * x + coefficient

    return result


def cauchy_radius_exact(coefficients):
    """Bán kính Cauchy dạng Fraction: mọi nghiệm thỏa |z| < R."""
    coefficients = strip_leading_zeros(coefficients)

    if polynomial_degree(coefficients) <= 0:
        return sp.S.Zero

    leading = abs(coefficients[0])
    return sp.S.One + max(abs(value) / leading for value in coefficients[1:])


def sturm_sequence(coefficients):
    """Lập dãy Sturm chính xác cho đa thức khác 0."""
    p0 = strip_leading_zeros(coefficients)

    if is_zero_polynomial(p0):
        raise ValueError("Không lập dãy Sturm cho đa thức 0.")

    p1 = strip_leading_zeros(polynomial_derivative(p0))
    sequence = [p0]

    if is_zero_polynomial(p1):
        return sequence

    sequence.append(p1)

    while not is_zero_polynomial(sequence[-1]):
        _, remainder = polynomial_divmod(sequence[-2], sequence[-1])

        if is_zero_polynomial(remainder):
            break

        sequence.append([-value for value in remainder])

    return sequence


def sturm_sign_variations(sequence, x):
    """Số lần đổi dấu của dãy Sturm tại x; bỏ qua phần tử bằng 0."""
    signs = []

    for polynomial in sequence:
        value = horner_value_exact(polynomial, x)
        if value > 0:
            signs.append(1)
        elif value < 0:
            signs.append(-1)

    return sum(signs[i] != signs[i - 1] for i in range(1, len(signs)))


def sturm_real_root_count(coefficients):
    """Đếm chính xác số nghiệm thực phân biệt của đa thức."""
    coefficients = strip_leading_zeros(coefficients)

    if polynomial_degree(coefficients) <= 0:
        return 0

    radius = cauchy_radius_exact(coefficients)
    sequence = sturm_sequence(coefficients)
    return sturm_sign_variations(sequence, -radius) - sturm_sign_variations(
        sequence, radius
    )


def polynomial_scale_at(coefficients, x):
    degree = polynomial_degree(coefficients)
    absolute_x = max(1.0, abs(x))

    return max(
        1.0,
        math.fsum(
            abs(float(coefficient)) * absolute_x ** (degree - i)
            for i, coefficient in enumerate(coefficients)
        ),
    )


def cauchy_radius(coefficients):
    return float(cauchy_radius_exact(coefficients))


def polynomial_at_negative_x(coefficients):
    degree = polynomial_degree(coefficients)

    return [
        coefficient * (-1 if (degree - i) % 2 else 1)
        for i, coefficient in enumerate(coefficients)
    ]


def lagrange_positive_upper_bound(coefficients):
    coefficients = strip_leading_zeros(coefficients)

    if polynomial_degree(coefficients) <= 0:
        return None

    if coefficients[0] < 0:
        coefficients = [-value for value in coefficients]

    negative_positions = [i for i in range(1, len(coefficients)) if coefficients[i] < 0]

    if not negative_positions:
        return None

    k = negative_positions[0]
    B = max(abs(float(coefficients[i])) for i in negative_positions)

    return 1.0 + (B / float(coefficients[0])) ** (1.0 / k)


# ============================================================
# ĐỊNH DẠNG ĐA THỨC
# ============================================================


def fraction_text(value):
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    return sp.sstr(sp.simplify(value))


def polynomial_text(coefficients, variable="x"):
    coefficients = strip_leading_zeros(coefficients)
    degree = polynomial_degree(coefficients)

    if is_zero_polynomial(coefficients):
        return "0"

    terms = []

    for i, coefficient in enumerate(coefficients):
        if coefficient == 0:
            continue

        exponent = degree - i
        sign = "-" if coefficient < 0 else "+"
        absolute = abs(coefficient)

        if exponent == 0:
            body = fraction_text(absolute)
        else:
            coefficient_part = "" if absolute == 1 else fraction_text(absolute) + "*"

            if exponent == 1:
                body = coefficient_part + variable
            else:
                body = coefficient_part + variable + f"^{exponent}"

        if not terms:
            terms.append(("-" if sign == "-" else "") + body)
        else:
            terms.append(f" {sign} {body}")

    return "".join(terms)


# ============================================================
# CHIA ĐÔI VÀ HIỆU CHỈNH NEWTON
# ============================================================


def newton_polish(coefficients, root, left, right, epsilon):
    """
    Hiệu chỉnh Newton nhưng chỉ chấp nhận khi vẫn giữ được chứng nhận
    |x - x*| <= epsilon từ khoảng phân ly [left, right].
    """
    derivative = polynomial_derivative(coefficients)
    current = root

    for _ in range(8):
        value = horner_value(coefficients, current)
        derivative_value = horner_value(derivative, current)

        if abs(derivative_value) <= 1e-18:
            break

        candidate = current - value / derivative_value

        if not (left <= candidate <= right):
            break

        # Nghiệm thật nằm trong [left,right]. Vì vậy chỉ nhận candidate
        # khi khoảng cách lớn nhất tới hai đầu không vượt quá epsilon.
        if max(candidate - left, right - candidate) > epsilon:
            break

        current = candidate

        if abs(value) <= 4.0 * math.ulp(1.0) * polynomial_scale_at(
            coefficients, current
        ):
            break

    return current


def bisection_core(coefficients, left, right, epsilon):
    left = float(left)
    right = float(right)

    if not (math.isfinite(left) and math.isfinite(right) and left <= right):
        raise ValueError("Khoảng chia đôi không hợp lệ.")

    left_value = horner_value(coefficients, left)
    right_value = horner_value(coefficients, right)

    if left_value == 0.0:
        return left, 0, [], left, left, 0.0, False

    if right_value == 0.0:
        return right, 0, [], right, right, 0.0, False

    if left_value * right_value > 0.0:
        raise ValueError("Khoảng chia đôi không có đổi dấu.")

    history = []
    iteration = 0

    # Số vòng lý thuyết đủ để nửa độ dài khoảng không vượt epsilon.
    initial_length = right - left
    maximum_needed = max(
        0,
        math.ceil(math.log2(initial_length / (2.0 * epsilon)))
        if initial_length > 2.0 * epsilon
        else 0,
    )
    maximum_safe = maximum_needed + 8

    while (right - left) / 2.0 > epsilon:
        iteration += 1
        middle = left + (right - left) / 2.0
        middle_value = horner_value(coefficients, middle)

        history.append((iteration, left, right, middle, middle_value))

        if middle_value == 0.0:
            left = right = middle
            break

        if left_value * middle_value < 0.0:
            right = middle
            right_value = middle_value
        else:
            left = middle
            left_value = middle_value

        if iteration > maximum_safe:
            raise ArithmeticError(
                "Chia đôi không co khoảng đúng như lý thuyết; dữ liệu số có thể đã tràn hoặc mất chính xác."
            )

    midpoint = left + (right - left) / 2.0
    certified_error = (right - left) / 2.0
    polished = newton_polish(coefficients, midpoint, left, right, epsilon)
    used_newton = polished != midpoint

    # Với nghiệm thật x* thuộc [left,right], chặn đúng của điểm được in là:
    # |polished-x*| <= max(polished-left, right-polished).
    certified_error = max(polished - left, right - polished)

    if certified_error > epsilon * (1.0 + 16.0 * math.ulp(1.0)):
        polished = midpoint
        used_newton = False
        certified_error = (right - left) / 2.0

    return polished, iteration, history, left, right, certified_error, used_newton


def bisection(coefficients, left, right, epsilon, print_full_table=False):
    initial_length = right - left
    theoretical_iterations = max(
        0, math.ceil(math.log2(initial_length / (2.0 * epsilon)))
    )

    (
        root,
        iteration,
        history,
        final_left,
        final_right,
        certified_error,
        used_newton,
    ) = bisection_core(coefficients, left, right, epsilon)

    if print_full_table:
        print("\nBảng lặp chia đôi:")
        print(
            " k | a_k                | b_k                | x_k                | P(x_k)"
        )
        print("-" * 102)

        for row in history:
            k, a, b, middle, value = row

            print(f"{k:2d} | {a:18.10f} | {b:18.10f} | {middle:18.10f} | {value: .10e}")

    print(f"Số lần lặp lý thuyết cần thiết: N >= {theoretical_iterations}")
    print(f"Số lần lặp đã thực hiện: {iteration}")
    print(f"Khoảng cuối: [{final_left:.12g}, {final_right:.12g}]")
    print(
        f"Chặn sai số được bảo đảm: |x - x*| <= "
        f"{certified_error:.3e} <= eps = {epsilon:.3e}"
    )
    if used_newton:
        print("Đã hiệu chỉnh Newton và vẫn bảo toàn chặn sai số trên.")
    else:
        print("Dùng trung điểm khoảng cuối để giữ đúng chứng nhận sai số.")

    return root, iteration, history


# ============================================================
# TÌM TOÀN BỘ NGHIỆM THỰC PHÂN BIỆT
# ============================================================


def deduplicate_sorted(values, tolerance):
    values = sorted(values)
    result = []

    for value in values:
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
        else:
            result[-1] = (result[-1] + value) / 2.0

    return result


def all_distinct_real_roots(coefficients, epsilon):
    coefficients = strip_leading_zeros(coefficients)

    if polynomial_degree(coefficients) <= 0:
        return []

    factors = square_free_decomposition(coefficients)
    roots = []

    for factor, _ in factors:
        roots.extend(roots_of_square_free_factor(factor, epsilon))

    return deduplicate_sorted(roots, 1e-12)


def roots_of_square_free_factor(coefficients, epsilon):
    coefficients = strip_leading_zeros(coefficients)
    degree = polynomial_degree(coefficients)

    if degree <= 0:
        return []

    if degree == 1:
        return [-float(coefficients[1]) / float(coefficients[0])]

    radius = cauchy_radius(coefficients)
    derivative = polynomial_derivative(coefficients)

    internal_epsilon = max(1e-14, min(1e-10, epsilon * 0.01))

    critical_points = all_distinct_real_roots(derivative, internal_epsilon)

    critical_points = [value for value in critical_points if -radius < value < radius]

    survey = [-radius] + critical_points + [radius]
    roots = []

    for left, right in zip(survey[:-1], survey[1:]):
        left_value = horner_value(coefficients, left)
        right_value = horner_value(coefficients, right)

        if left_value * right_value < 0:
            root, _, _, _, _, _, _ = bisection_core(coefficients, left, right, epsilon)
            roots.append(root)

    return deduplicate_sorted(roots, 1e-12)


# ============================================================
# IN QUÁ TRÌNH GIẢI MỘT NHÂN TỬ KHÔNG CÓ NGHIỆM BỘI
# ============================================================


def solve_square_free_factor_with_output(
    coefficients, multiplicity, epsilon, decimals, print_full_table
):
    degree = polynomial_degree(coefficients)

    print("\n" + "=" * 106)
    print(f"GIẢI NHÂN TỬ ỨNG VỚI NGHIỆM BỘI {multiplicity}")
    print("=" * 106)

    print(f"\nQ(x) = {polynomial_text(coefficients)}")
    print(f"Bậc của Q: {degree}")
    expected_real_roots = sturm_real_root_count(coefficients)

    if degree == 1:
        exact_root = -coefficients[1] / coefficients[0]
        root = float(exact_root)

        print("\nĐây là phương trình bậc nhất:")
        print(f"x = -a_1/a_0 = {fraction_text(exact_root)}")
        print(f"x ≈ {root:.{decimals}f}")

        return [root]

    radius = cauchy_radius(coefficients)
    derivative = polynomial_derivative(coefficients)

    internal_epsilon = max(1e-14, min(1e-10, epsilon * 0.01))

    critical_points = all_distinct_real_roots(derivative, internal_epsilon)

    critical_points = [value for value in critical_points if -radius < value < radius]

    print("\nĐạo hàm:")
    print(f"Q'(x) = {polynomial_text(derivative)}")

    print("\nCác điểm tới hạn thực của Q trong miền chứa nghiệm:")

    if critical_points:
        for i, value in enumerate(critical_points, start=1):
            print(f"c_{i} ≈ {value:.{decimals}f}")
    else:
        print("Không có điểm tới hạn thực.")

    survey = [-radius] + critical_points + [radius]

    print("\nMảng khảo sát:")
    print("survey = [" + ", ".join(f"{value:.{decimals}f}" for value in survey) + "]")

    print("\nBảng dấu tại các mốc khảo sát:")
    print(" i | x_i                  | Q(x_i)")
    print("-" * 58)

    survey_values = []

    for i, value in enumerate(survey):
        function_value = horner_value(coefficients, value)
        survey_values.append(function_value)

        print(f"{i:2d} | {value:20.{decimals}f} | {function_value: .12e}")

    intervals = []

    for i in range(len(survey) - 1):
        if survey_values[i] * survey_values[i + 1] < 0:
            intervals.append((survey[i], survey[i + 1]))

    print("\nCác khoảng phân ly nghiệm:")

    if not intervals:
        print("Không có khoảng đổi dấu.")
        if expected_real_roots == 0:
            print("Nhân tử này không có nghiệm thực.")
            return []
        raise ArithmeticError(
            "Phân ly bằng các điểm tới hạn gần đúng đã bỏ sót nghiệm thực. "
            "Hãy tăng độ chính xác tính toán hoặc kiểm tra lại dữ liệu."
        )

    for i, (left, right) in enumerate(intervals, start=1):
        print(f"I_{i} = ({left:.{decimals}f}, {right:.{decimals}f})")

    if len(intervals) != expected_real_roots:
        raise ArithmeticError(
            f"Kiểm tra chính xác cho thấy Q có {expected_real_roots} nghiệm thực phân biệt, "
            f"nhưng quá trình phân ly số chỉ tạo được {len(intervals)} khoảng. "
            "Không tiếp tục để tránh bỏ sót hoặc lặp nghiệm."
        )

    print(
        f"Kiểm tra tính đầy đủ: {len(intervals)} khoảng phân ly ứng với đúng "
        f"{expected_real_roots} nghiệm thực phân biệt của Q."
    )

    roots = []

    for i, (left, right) in enumerate(intervals, start=1):
        print("\n" + "-" * 106)
        print(f"CHIA ĐÔI TRÊN KHOẢNG I_{i}")
        print(f"a = {left:.{decimals}f}, b = {right:.{decimals}f}")
        print(f"Q(a) = {horner_value(coefficients, left):.12e}")
        print(f"Q(b) = {horner_value(coefficients, right):.12e}")
        print("Q(a)*Q(b) < 0 nên khoảng này chứa đúng một nghiệm.")

        root, _, _ = bisection(coefficients, left, right, epsilon, print_full_table)

        print(f"Nghiệm gần đúng được chọn: x ≈ {root:.{decimals}f}")
        print(f"Q(x) ≈ {horner_value(coefficients, root):.12e}")

        roots.append(root)

    return roots


# ============================================================
# KIỂM TRA PHÂN TÍCH BÌNH PHƯƠNG TỰ DO
# ============================================================


def reconstruct_from_square_free(leading_coefficient, factors):
    result = [Fraction(1)]

    for factor, multiplicity in factors:
        result = polynomial_multiply(result, polynomial_power(factor, multiplicity))

    return [leading_coefficient * value for value in result]


# ============================================================
# TRÌNH BÀY LÝ THUYẾT
# ============================================================


def print_theory():
    print("\n" + "=" * 106)
    print("PHƯƠNG PHÁP GIẢI PHƯƠNG TRÌNH ĐA THỨC")
    print("=" * 106)

    print("\nInput: hệ số của P(x), sai số eps.")
    print("Output: toàn bộ nghiệm thực, bội của nghiệm và khoảng sai số chứng nhận.")
    print("\nCho:")
    print("P(x) = a₀xⁿ + a₁xⁿ⁻¹ + … + aₙ, a₀ ≠ 0.")

    print("\n1. Sơ đồ Hoocne")
    print("  b₀ = a₀")
    print("  bᵢ = bᵢ₋₁x + aᵢ, i = 1,…,n")
    print("  P(x) = bₙ")
    print("Sơ đồ này được dùng để tính nhanh P(x) trong khảo sát và chia đôi.")

    print("\n2. Miền chứa nghiệm")
    print("  R = 1 + max{|aᵢ/a₀|, i = 1,…,n}")
    print(
        "Mọi nghiệm phức z của P đều thỏa |z| < R; "
        "do đó mọi nghiệm thực nằm trong (-R,R)."
    )

    print("\n3. Phân ly nghiệm")
    print("Các nghiệm thực của P'(x) chia (-R,R) thành những khoảng mà P đơn điệu.")
    print(
        "Nếu P(a)*P(b) < 0 trên hai mốc liên tiếp "
        "thì (a,b) chứa đúng một nghiệm thực đơn."
    )

    print("\n4. Nghiệm bội")
    print("Để không bỏ sót nghiệm bội chẵn, chương trình phân tích:")
    print("  P(x)/a₀ = Q₁(x)Q₂(x)²…")
    print("Mọi nghiệm của Q_j có bội đúng bằng j trong P.")

    print("\n5. Phương pháp chia đôi")
    print("  xₖ = (aₖ + bₖ)/2")
    print("  Giữ lại nửa khoảng có đổi dấu.")
    print("  Dừng khi (bₖ − aₖ)/2 ≤ ε.")
    print(
        "  Nghiệm in ra phải nằm trong khoảng cuối và có chặn "
        "|x-x*| <= eps; hiệu chỉnh Newton chỉ được nhận nếu không làm mất chặn này."
    )

    print("\n6. Phạm vi kết quả")
    print(
        "  Chương trình tìm toàn bộ nghiệm thực và bội của chúng. "
        "Các nghiệm phức không thực chỉ được đếm, không được tính giá trị."
    )


# ============================================================
# GIẢI PHƯƠNG TRÌNH ĐA THỨC
# ============================================================


def solve_polynomial(original_coefficients, epsilon, decimals, print_full_table):
    print_theory()

    requested_epsilon = epsilon
    minimum_decimals = minimum_display_decimals(requested_epsilon)
    entered_decimals = decimals
    decimals = max(decimals, minimum_decimals)

    # Dành một nửa ngân sách sai số cho phép tính và một nửa cho việc
    # làm tròn số được in. Nhờ đó con số người làm bài chép ra vẫn đạt eps.
    epsilon = requested_epsilon / 2.0

    coefficients = strip_leading_zeros(original_coefficients)

    entered_degree = len(original_coefficients) - 1
    actual_degree = polynomial_degree(coefficients)

    print("\n" + "=" * 106)
    print("DỮ LIỆU VÀ XỬ LÝ BAN ĐẦU")
    print("=" * 106)

    print(f"\nSai số yêu cầu của nghiệm được chép ra: eps = {requested_epsilon:.3e}")
    if decimals > entered_decimals:
        print(
            f"Số chữ số hiển thị đã tự tăng từ {entered_decimals} lên {decimals} "
            "để sai số làm tròn không vượt quá eps/2."
        )
    print(f"Sai số dùng trong chia đôi: eps_tính = eps/2 = {epsilon:.3e}.")

    print(f"\nĐa thức đã nhập theo bậc khai báo {entered_degree}:")
    print(f"P(x) = {polynomial_text(original_coefficients)}")

    if is_zero_polynomial(coefficients):
        print("\nTất cả hệ số đều bằng 0.")
        print("KẾT LUẬN: Phương trình 0 = 0 có vô số nghiệm.")
        return []

    if actual_degree < entered_degree:
        print(
            f"\nCác hệ số đầu bằng 0 nên bậc thực tế giảm từ "
            f"{entered_degree} xuống {actual_degree}."
        )
        print(f"P(x) = {polynomial_text(coefficients)}")

    if actual_degree == 0:
        print("\nP(x) là một hằng số khác 0.")
        print("KẾT LUẬN: Phương trình không có nghiệm.")
        return []

    if actual_degree == 1:
        root_exact = -coefficients[1] / coefficients[0]
        root = float(root_exact)

        print("\nĐây là phương trình bậc nhất.")
        print(f"x = -a_1/a_0 = {fraction_text(root_exact)}")

        print("\n" + "=" * 106)
        print("KẾT QUẢ CUỐI CÙNG")
        print("=" * 106)
        print(f"\nx_1 ≈ {root:.{decimals}f}")
        print("Bội của nghiệm: 1")
        print(f"P(x_1) ≈ {horner_value(coefficients, root):.12e}")
        print("\nĐã tìm đủ toàn bộ nghiệm thực của đa thức.")

        return [(root, 1)]

    derivative = polynomial_derivative(coefficients)

    print("\nĐạo hàm của đa thức:")
    print(f"P'(x) = {polynomial_text(derivative)}")

    radius = cauchy_radius(coefficients)
    numerical_floor = 32.0 * math.ulp(max(1.0, radius))
    if epsilon < numerical_floor:
        raise ValueError(
            f"eps quá nhỏ so với độ phân giải số thực ở miền |x| <= {radius:.6g}. "
            f"Với cách tính hiện tại, eps_tính nên không nhỏ hơn khoảng {numerical_floor:.3e}."
        )

    print("\nBán kính nghiệm Cauchy:")
    print("R = 1 + max|aᵢ/a₀|")
    print(f"max|aᵢ/a₀| = {radius - 1.0:.{decimals}f}")
    print(f"R = {radius:.{decimals}f}")
    print(
        f"Mọi nghiệm thực của P nằm trong "
        f"(-{radius:.{decimals}f}, "
        f"{radius:.{decimals}f})."
    )

    positive_bound = lagrange_positive_upper_bound(coefficients)

    negative_bound = lagrange_positive_upper_bound(
        polynomial_at_negative_x(coefficients)
    )

    print("\nCận Lagrange theo dấu hệ số:")

    if positive_bound is None:
        print("Sau khi chuẩn hóa a₀ > 0 không có hệ số âm; không có nghiệm dương.")
    else:
        print(f"Mọi nghiệm dương thỏa 0 < x < {positive_bound:.{decimals}f}.")

    if negative_bound is None:
        print("P(-x) sau khi chuẩn hóa không có hệ số âm; không có nghiệm âm.")
    else:
        print(f"Mọi nghiệm âm thỏa -{negative_bound:.{decimals}f} < x < 0.")

    print("\n" + "=" * 106)
    print("PHÂN TÍCH NGHIỆM BỘI")
    print("=" * 106)

    factors = square_free_decomposition(coefficients)
    reconstructed = reconstruct_from_square_free(coefficients[0], factors)

    decomposition_terms = [
        f"({polynomial_text(factor)})^{multiplicity}"
        for factor, multiplicity in factors
    ]

    print("\nPhân tích bình phương tự do:")
    print(
        f"P(x) = {fraction_text(coefficients[0])}"
        + (" * " if decomposition_terms else "")
        + " * ".join(decomposition_terms)
    )

    print(
        "\nKiểm tra nhân lại các nhân tử: "
        + ("ĐÚNG." if reconstructed == coefficients else "KHÔNG KHỚP.")
    )

    all_roots = []

    for factor, multiplicity in factors:
        factor_roots = solve_square_free_factor_with_output(
            factor, multiplicity, epsilon, decimals, print_full_table
        )

        for root in factor_roots:
            all_roots.append((root, multiplicity))

    all_roots.sort(key=lambda item: item[0])

    distinct_expected = sturm_real_root_count(coefficients)
    if len(all_roots) != distinct_expected:
        raise ArithmeticError(
            f"Kiểm tra cuối cho thấy phải có {distinct_expected} nghiệm thực phân biệt, "
            f"nhưng chương trình đang có {len(all_roots)} nghiệm. "
            "Dừng để không in kết luận sai."
        )

    print("\n" + "=" * 106)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 106)
    print(
        f"Sai số tính toán không quá {epsilon:.3e}; sai số làm tròn của số in "
        f"không quá {0.5 * 10.0 ** (-decimals):.3e}."
    )
    print(
        f"Vì vậy sai số tuyệt đối của mỗi nghiệm được chép ra không vượt quá "
        f"eps = {requested_epsilon:.3e}."
    )

    if not all_roots:
        print("\nPhương trình không có nghiệm thực.")
        print(f"Đa thức bậc {actual_degree} có các nghiệm còn lại là nghiệm phức.")
        return []

    total_real_multiplicity = sum(multiplicity for _, multiplicity in all_roots)

    for index, (root, multiplicity) in enumerate(all_roots, start=1):
        residual = horner_value(coefficients, root)

        print(f"\nx_{index} ≈ {root:.{decimals}f}")
        print(f"Bội của nghiệm: {multiplicity}")
        print(f"P(x_{index}) ≈ {residual:.12e}")

        if multiplicity > 1:
            print("Kiểm tra các đạo hàm tại nghiệm:")
            current = coefficients[:]

            for order in range(multiplicity + 1):
                value = horner_value(current, root)
                print(f"P^({order})(x_{index}) ≈ {value:.12e}")
                current = polynomial_derivative(current)

    print(f"\nTổng bội của các nghiệm thực tìm được: {total_real_multiplicity}")

    complex_count = actual_degree - total_real_multiplicity

    if complex_count > 0:
        print(f"Số nghiệm phức không thực còn lại (tính cả bội): {complex_count}")
    else:
        print("Đã tìm đủ toàn bộ nghiệm của đa thức và tất cả đều là nghiệm thực.")

    return all_roots


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================


def main():
    print("=" * 106)
    print("GIẢI PHƯƠNG TRÌNH ĐA THỨC")
    print("=" * 106)

    declared_degree = input_nonnegative_integer("Nhập bậc n của đa thức: ")

    coefficients = input_coefficients(declared_degree)

    epsilon = input_positive_number("\nNhập sai số eps [Enter = 1e-7]: ", 1e-7)

    decimals = input_nonnegative_integer(
        "Số chữ số sau dấu phẩy [Enter = 7]: ", default=7
    )

    print_full_table = choose_output_mode()

    solve_polynomial(coefficients, epsilon, decimals, print_full_table)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except Exception as error:
        print(f"\nLỗi trong quá trình tính toán: {error}")
