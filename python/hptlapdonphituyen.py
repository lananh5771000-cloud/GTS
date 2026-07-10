"""Giai he phuong trinh phi tuyen bang phep lap don dong thoi.

Moi nghiem do chuong trinh tao ra deu co dang x^(k+1) = Phi(x^(k)).
SymPy chi duoc dung de bien doi dai so; khong dung Newton/nsolve/Seidel.
"""

from __future__ import annotations

import itertools
from exam_format import exam_print as print
import math
import sys
import time
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Sequence

import mpmath as mp
import sympy as sp
from input_utils import MathInputError, parse_math_expression, parse_real

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


ALLOWED_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
}
BAD_NAMES = {"nan", "NaN", "inf", "Inf", "Infinity", "zoo", "oo"}
MAX_CANDIDATES = 5000
MAX_ISOLATED_CANDIDATES = 100
MAX_SUBREGION_CANDIDATES = 1000
MAX_ADAPTIVE_BOXES = 4096
DEFAULT_SEARCH_SECONDS = 10.0
TOL = 1e-12
EXAM_OUTPUT = True
DEBUG_OUTPUT = False
TYPE_PRIORITY = {
    "direct": 0,
    "equivalent": 1,
    "isolated": 2,
    "relaxation": 3,
    "unknown": 4,
}


class InputError(ValueError):
    pass


def parse_expression(text: str, symbols: Sequence[sp.Symbol]) -> sp.Expr:
    text = text.strip().replace("^", "**")
    if not text or any(word in text for word in BAD_NAMES):
        raise InputError("Biểu thức rỗng hoặc chứa NaN/inf.")
    local = dict(ALLOWED_FUNCTIONS)
    local.update({str(x): x for x in symbols})
    try:
        expr = parse_math_expression(text, {str(x): x for x in symbols})
    except (MathInputError, sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        raise InputError(f"Biểu thức không hợp lệ: {text}") from exc
    unknown = expr.free_symbols - set(symbols)
    if unknown:
        raise InputError("Biến không hợp lệ: " + ", ".join(map(str, unknown)))
    if expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or not all(
        f.func in set(ALLOWED_FUNCTIONS.values()) for f in expr.atoms(sp.Function)
    ):
        raise InputError("Biểu thức chứa hàm hoặc ký hiệu không được hỗ trợ.")
    return expr


def parse_number(text: str) -> float:
    expr = parse_expression(text, ())
    if expr.free_symbols or expr.is_real is False:
        raise InputError("Cần một số thực.")
    try:
        value = parse_real(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError("Không tính được số thực.") from exc
    if not math.isfinite(value):
        raise InputError("Số phải hữu hạn.")
    return value


def parse_equation(text: str, symbols: Sequence[sp.Symbol]) -> sp.Expr:
    if text.count("=") > 1:
        raise InputError("Mỗi phương trình chỉ có tối đa một dấu '='.")
    parts = text.split("=")
    lhs = parse_expression(parts[0], symbols)
    rhs = parse_expression(parts[1], symbols) if len(parts) == 2 else sp.S.Zero
    return sp.simplify(lhs - rhs)


def read_int(prompt: str, minimum: int = 0) -> int:
    while True:
        try:
            value = int(input(prompt).strip())
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            print(f"[!] Hãy nhập số nguyên ≥ {minimum}.")


def read_number(prompt: str, positive: bool = False) -> float:
    while True:
        try:
            value = parse_number(input(prompt))
            if positive and value <= 0:
                raise InputError("Giá trị phải dương.")
            return value
        except InputError as exc:
            print(f"[!] {exc}")


def read_vector(n: int, name: str) -> list[float]:
    return [read_number(f"{name}{i + 1} = ") for i in range(n)]


def read_bounds(n: int) -> list[tuple[float, float]]:
    result = []
    for i in range(n):
        while True:
            a = read_number(f"a{i + 1} = ")
            b = read_number(f"b{i + 1} = ")
            if a < b:
                result.append((a, b))
                break
            print("[!] Cần a_i < b_i.")
    return result


def read_system(n: int, symbols: Sequence[sp.Symbol]) -> list[sp.Expr]:
    equations = []
    for i in range(n):
        while True:
            try:
                prompt = f"F{i + 1}(x)=0, nhập biểu thức F{i + 1}(x): "
                equations.append(parse_equation(input(prompt), symbols))
                break
            except InputError as exc:
                print(f"[!] {exc}")
    return equations


# ---------- So hoc khoang co huong lam tron cua mpmath.iv ----------
def _ends(v) -> tuple[float, float]:
    try:
        return float(v.a), float(v.b)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Khoảng không hữu hạn hoặc không thực") from exc


def _interval_pow(base, exponent: sp.Expr):
    """Lũy thừa khoảng thực, xử lý riêng mẫu chẵn/lẻ và số mũ âm."""
    if exponent.is_Integer:
        lo, hi = _ends(base)
        if int(exponent) < 0 and lo <= 0 <= hi:
            raise ValueError("lũy thừa âm có cơ số chứa 0")
        return base ** int(exponent)
    if exponent.is_Rational:
        p, q = int(exponent.p), int(exponent.q)
        lo, hi = _ends(base)
        if q % 2 == 0 and lo < 0:
            raise ValueError("lũy thừa hữu tỉ mẫu chẵn cần cơ số không âm")
        if p < 0 and lo <= 0 <= hi:
            raise ValueError("lũy thừa âm có cơ số chứa 0")

        def real_power(value):
            if q % 2:
                root = math.copysign(abs(value) ** (1.0 / q), value)
                return root**p
            return value ** (p / q)

        values = [real_power(lo), real_power(hi)]
        if p > 0 and p % 2 == 0 and lo <= 0 <= hi:
            values.append(0.0)
        lower, upper = min(values), max(values)
        return mp.iv.mpf([str(lower), str(upper)])
    raise NotImplementedError("số mũ không được hỗ trợ")


def interval_evaluate(expr: sp.Expr, symbols, bounds):
    """Tra ve lower, upper, valid, reason (bao ngoai bang so hoc khoang)."""
    env = {s: mp.iv.mpf([a, b]) for s, (a, b) in zip(symbols, bounds)}

    def walk(e):
        if e == sp.pi or e == sp.E or (e.is_number and not e.free_symbols):
            return mp.iv.mpf([str(e.evalf(40)), str(e.evalf(40))])
        if e.is_Symbol:
            if e not in env:
                raise ValueError(f"bien la {e}")
            return env[e]
        args = [walk(a) for a in e.args]
        if e.func is sp.Add:
            out = mp.iv.mpf([0, 0])
            for a in args:
                out += a
            return out
        if e.func is sp.Mul:
            out = mp.iv.mpf([1, 1])
            for a in args:
                out *= a
            return out
        if e.func is sp.Pow:
            return _interval_pow(args[0], e.args[1])
        if e.func is sp.sin:
            return mp.iv.sin(args[0])
        if e.func is sp.cos:
            return mp.iv.cos(args[0])
        if e.func is sp.tan:
            lo, hi = _ends(args[0])
            k0 = math.ceil((lo - math.pi / 2) / math.pi)
            k1 = math.floor((hi - math.pi / 2) / math.pi)
            if k0 <= k1:
                raise ValueError("tan gap diem cuc")
            return mp.iv.tan(args[0])
        if e.func is sp.exp:
            return mp.iv.exp(args[0])
        if e.func is sp.log:
            lo, _ = _ends(args[0])
            if lo <= 0:
                raise ValueError("đối số log có thể không dương")
            return mp.iv.log(args[0])
        if e.func is sp.Abs:
            return abs(args[0])
        raise NotImplementedError(f"chưa hỗ trợ {e.func}")

    try:
        lo, hi = _ends(walk(expr))
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError("ket qua NaN/inf")
        return lo, hi, True, ""
    except (ValueError, ZeroDivisionError, NotImplementedError, TypeError) as exc:
        return math.nan, math.nan, False, str(exc)


def adaptive_interval_evaluate_with_meta(
    expr: sp.Expr,
    symbols,
    bounds,
    max_depth=8,
    max_boxes=MAX_ADAPTIVE_BOXES,
    target_width=None,
    deadline=None,
):
    """Đánh giá chia hộp và trả cả metadata của chứng minh."""
    queue = [(list(bounds), 0)]
    accepted = []
    boxes = 0
    max_depth_used = 0
    terminal_reason = ""

    def result(lo, hi, valid, reason):
        return (
            lo,
            hi,
            valid,
            reason,
            {
                "boxes_used": boxes,
                "max_depth_used": max_depth_used,
                "method": "adaptive_subdivision",
            },
        )

    while queue and boxes < max_boxes:
        if deadline is not None and time.perf_counter() >= deadline:
            return result(
                math.nan, math.nan, False, "đã đạt giới hạn thời gian chia miền"
            )
        box, depth = queue.pop(0)
        boxes += 1
        max_depth_used = max(max_depth_used, depth)
        lo, hi, valid, reason = interval_evaluate(expr, symbols, box)
        narrow = valid and (target_width is None or hi - lo <= target_width)
        if narrow or (valid and depth >= max_depth):
            accepted.append((lo, hi))
            continue
        if depth >= max_depth:
            terminal_reason = reason or "khoảng chưa đủ hẹp sau giới hạn chia hộp"
            return result(math.nan, math.nan, False, terminal_reason)
        widths = [b - a for a, b in box]
        j = max(range(len(widths)), key=lambda i: (widths[i], -i))
        if widths[j] <= 0:
            return result(math.nan, math.nan, False, reason or "không thể chia hộp")
        mid = (box[j][0] + box[j][1]) / 2
        left, right = list(box), list(box)
        left[j] = (box[j][0], mid)
        right[j] = (mid, box[j][1])
        queue.extend([(left, depth + 1), (right, depth + 1)])
    if queue:
        return result(math.nan, math.nan, False, "vượt giới hạn số hộp")
    if not accepted:
        return result(
            math.nan, math.nan, False, terminal_reason or "không đánh giá được"
        )
    return result(min(v[0] for v in accepted), max(v[1] for v in accepted), True, "")


def adaptive_interval_evaluate(
    expr: sp.Expr,
    symbols,
    bounds,
    max_depth=8,
    max_boxes=MAX_ADAPTIVE_BOXES,
    target_width=None,
    deadline=None,
):
    """API tương thích: trả lower, upper, valid, reason."""
    return adaptive_interval_evaluate_with_meta(
        expr, symbols, bounds, max_depth, max_boxes, target_width, deadline
    )[:4]


def check_domain(phi, symbols, bounds, deadline=None):
    details, ok = [], True
    for i, expr in enumerate(phi):
        lo, hi, valid, reason = interval_evaluate(expr, symbols, bounds)
        if not valid:
            lo, hi, valid, reason = adaptive_interval_evaluate(
                expr,
                symbols,
                bounds,
                max_depth=8,
                max_boxes=MAX_ADAPTIVE_BOXES,
                deadline=deadline,
            )
        details.append((lo, hi, valid, reason))
        ok &= valid
    return bool(ok), details


def check_mapping_subset_with_meta(phi, symbols, bounds, deadline=None):
    """Kiểm tra Phi(D) subset D và lưu đúng khoảng đã dùng chứng minh."""
    domain_ok, ranges = check_domain(phi, symbols, bounds, deadline)
    refined = []
    adaptive_used = False
    boxes_used = 0
    max_depth_used = 0
    for expr, current, (a, b) in zip(phi, ranges, bounds):
        lo, hi, valid, reason = current
        if valid and not (a - TOL <= lo and hi <= b + TOL):
            lo, hi, valid, reason, meta = adaptive_interval_evaluate_with_meta(
                expr,
                symbols,
                bounds,
                max_depth=8,
                max_boxes=4096,
                target_width=max((b - a) / 256, TOL),
                deadline=deadline,
            )
            adaptive_used = True
            boxes_used += meta["boxes_used"]
            max_depth_used = max(max_depth_used, meta["max_depth_used"])
        refined.append((lo, hi, valid, reason))
    ranges = refined
    ok = domain_ok and all(
        valid and a - TOL <= lo and hi <= b + TOL
        for (lo, hi, valid, _), (a, b) in zip(ranges, bounds)
    )
    meta = {
        "method": "adaptive_subdivision" if adaptive_used else "direct_interval",
        "boxes_used": boxes_used,
        "max_depth_used": max_depth_used,
    }
    return ok, ranges, meta


def check_mapping_subset(phi, symbols, bounds, deadline=None):
    """API tương thích cho các chỗ chỉ cần kết luận và khoảng."""
    ok, ranges, _ = check_mapping_subset_with_meta(phi, symbols, bounds, deadline)
    return ok, ranges


def bound_jacobian(phi, symbols, bounds, deadline=None):
    derivs, intervals, matrix = [], [], []
    valid = True
    for expr in phi:
        drow, irow, mrow = [], [], []
        for x in symbols:
            derivative = sp.simplify(sp.diff(expr, x))
            lo, hi, good, reason = interval_evaluate(derivative, symbols, bounds)
            drow.append(derivative)
            irow.append((lo, hi, good, reason))
            mrow.append(max(abs(lo), abs(hi)) if good else math.inf)
            valid &= good
        derivs.append(drow)
        intervals.append(irow)
        matrix.append(mrow)
    q_inf = max((sum(row) for row in matrix), default=0.0)
    q_one = max(
        (sum(matrix[i][j] for i in range(len(phi))) for j in range(len(symbols))),
        default=0.0,
    )
    if valid and min(q_inf, q_one) >= 1:
        for i, row in enumerate(derivs):
            for j, derivative in enumerate(row):
                old = intervals[i][j]
                target = max((old[1] - old[0]) / 32, TOL) if old[2] else TOL
                lo, hi, good, reason = adaptive_interval_evaluate(
                    derivative,
                    symbols,
                    bounds,
                    max_depth=8,
                    max_boxes=MAX_ADAPTIVE_BOXES,
                    target_width=target,
                    deadline=deadline,
                )
                intervals[i][j] = (lo, hi, good, reason)
                matrix[i][j] = max(abs(lo), abs(hi)) if good else math.inf
                valid &= good
        q_inf = max((sum(row) for row in matrix), default=0.0)
        q_one = max(
            (sum(matrix[i][j] for i in range(len(phi))) for j in range(len(symbols))),
            default=0.0,
        )
    return derivs, intervals, matrix, q_inf, q_one, bool(valid)


@dataclass
class Candidate:
    phi: list[sp.Expr]
    source: str
    branches: list[str] = field(default_factory=list)
    candidate_type: str = "unknown"
    domain_proven: bool = False
    mapping_proven: bool = False
    q_inf: float = math.inf
    q_one: float = math.inf
    q: float = math.inf
    selected_norm: str = "khong xac dinh"
    complexity: int = 0
    notes: str = ""
    jacobian_data: tuple | None = None
    mapping_ranges: list | None = None
    domain_ranges: list | None = None
    derivative_ranges: list | None = None
    proof_bounds: list | None = None
    mapping_method: str = ""
    mapping_meta: dict = field(default_factory=dict)

    @property
    def banach_proven(self):
        return (
            self.domain_proven
            and self.mapping_proven
            and math.isfinite(self.q)
            and self.q < 1
        )


def generate_isolation_candidates(F, symbols, bounds=None, deadline=None):
    choices = {x: [] for x in symbols}
    for r, equation in enumerate(F):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        for x in symbols:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            try:
                roots = sp.solve(sp.Eq(equation, 0), x, check=True)
            except (NotImplementedError, ValueError, TypeError):
                roots = []
            for root in roots:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                root = sp.simplify(root)
                if any(
                    func.func not in set(ALLOWED_FUNCTIONS.values())
                    for func in root.atoms(sp.Function)
                ):
                    continue
                if bounds is not None:
                    valid, _ = check_domain([root], symbols, bounds, deadline)
                    if not valid:
                        continue
                    lo, hi, good, _ = adaptive_interval_evaluate(
                        root,
                        symbols,
                        bounds,
                        max_depth=5,
                        max_boxes=256,
                        deadline=deadline,
                    )
                    target = bounds[list(symbols).index(x)]
                    if good and (hi < target[0] - TOL or lo > target[1] + TOL):
                        continue
                # Khong chap nhan nhanh neu SymPy khong chung minh duoc rang
                # the nguoc vao phuong trinh goc cho dong nhat 0.
                verification = equation.subs(x, root)
                forms = []
                for transform in (
                    sp.simplify,
                    sp.cancel,
                    sp.factor,
                    sp.trigsimp,
                    lambda e: sp.powsimp(e, force=False),
                ):
                    try:
                        forms.append(transform(verification))
                    except (
                        TypeError,
                        ValueError,
                        NotImplementedError,
                        sp.PolynomialError,
                    ):
                        continue
                verified = any(v == 0 for v in forms)
                if not verified:
                    for value in forms:
                        try:
                            if value.equals(0) is True:
                                verified = True
                                break
                        except (TypeError, ValueError, NotImplementedError):
                            continue
                if (
                    verified
                    and x not in root.free_symbols
                    and root.free_symbols <= set(symbols)
                    and root.is_real is not False
                ):
                    item = (r, root)
                    if item not in choices[x]:
                        choices[x].append(item)
    candidates = []
    if any(not choices[x] for x in symbols):
        return candidates

    # Backtracking dong thoi ghep song anh phuong trinh-bien.
    def visit(j, used, selected):
        if len(candidates) >= MAX_ISOLATED_CANDIDATES:
            return
        if deadline is not None and time.perf_counter() >= deadline:
            return
        if j == len(symbols):
            candidates.append(
                Candidate(
                    [e for _, e in selected],
                    "Rút biến đại số",
                    [f"F{r + 1}->{symbols[i]}" for i, (r, _) in enumerate(selected)],
                    candidate_type="isolated",
                )
            )
            return
        for r, expr in choices[symbols[j]]:
            if r not in used:
                visit(j + 1, used | {r}, selected + [(r, expr)])

    visit(0, set(), [])
    return candidates


def generate_relaxation_candidates(F, symbols, bounds, x0):
    point = x0 if x0 else [(a + b) / 2 for a, b in bounds]
    bases = []
    for i, (f, x) in enumerate(zip(F, symbols)):
        try:
            d = float(sp.N(sp.diff(f, x).subs(dict(zip(symbols, point)))))
        except (TypeError, ValueError, ZeroDivisionError):
            return []
        if not math.isfinite(d) or abs(d) < TOL:
            return []
        bases.append(1.0 / d)
    result = []
    for factor in (0.25, 0.5, 1.0, 1.5):
        phi = [
            sp.simplify(x - factor * lam * f) for x, lam, f in zip(symbols, bases, F)
        ]
        result.append(
            Candidate(
                phi,
                "Lặp đơn thư giãn",
                candidate_type="relaxation",
                notes=f"hệ số tỷ lệ lambda: {factor:g}",
            )
        )
    return result


def generate_canonical_equivalent_candidates(F, symbols, deadline=None):
    """Sinh và xác minh x_j=x_j-F_r trước mọi phép solve."""
    result = []
    if len(symbols) <= 6:
        assignments = itertools.permutations(range(len(F)), len(symbols))
    else:
        assignments = [tuple(range(min(len(F), len(symbols))))]
    for assignment in itertools.islice(assignments, MAX_CANDIDATES):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if len(assignment) != len(symbols):
            continue
        phi, verified = [], True
        for x, r in zip(symbols, assignment):
            component = sp.simplify(x - F[r])
            identity = x - component - F[r]
            forms = []
            for transform in (sp.simplify, sp.cancel, sp.factor, sp.trigsimp):
                try:
                    forms.append(transform(identity))
                except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
                    continue
            if not any(value == 0 or value.equals(0) is True for value in forms):
                verified = False
                break
            phi.append(component)
        if not verified:
            continue
        result.append(
            Candidate(
                phi,
                "Biến đổi tương đương",
                [f"F{r + 1}->{symbols[i]}" for i, r in enumerate(assignment)],
                candidate_type="equivalent",
            )
        )
    return result


def generate_unit_rearrangement_candidates(F, symbols):
    """Alias tương thích cho mã gọi cũ."""
    return generate_canonical_equivalent_candidates(F, symbols)


def evaluate_phi_candidate(c, symbols, bounds, deadline=None):
    c.domain_proven, c.domain_ranges = check_domain(c.phi, symbols, bounds, deadline)
    c.mapping_proven, c.mapping_ranges, meta = check_mapping_subset_with_meta(
        c.phi, symbols, bounds, deadline
    )
    data = bound_jacobian(c.phi, symbols, bounds, deadline)
    c.jacobian_data = data
    c.derivative_ranges = data[1]
    c.proof_bounds = [tuple(v) for v in bounds]
    c.mapping_method = meta["method"]
    c.mapping_meta = meta
    c.q_inf, c.q_one = data[3], data[4]
    c.q = min(c.q_inf, c.q_one)
    c.selected_norm = "vo cung" if c.q_inf <= c.q_one else "1"
    c.complexity = sum(sp.count_ops(e) for e in c.phi)
    return c


def candidate_search_key(candidate):
    """Khóa tìm kiếm ưu tiên dạng đơn giản, tránh nhánh nghịch đảo phức tạp."""
    inverse_functions = {sp.asin, sp.acos, sp.atan, sp.asinh, sp.acosh, sp.atanh}
    inverse_count = sum(
        expr.count(func) for expr in candidate.phi for func in inverse_functions
    )
    root_count = sum(
        1
        for expr in candidate.phi
        for power in expr.atoms(sp.Pow)
        if power.exp.is_Rational and power.exp.q > 1
    )
    division_count = sum(
        1
        for expr in candidate.phi
        for power in expr.atoms(sp.Pow)
        if power.exp.is_number and power.exp.is_negative
    )
    return (
        TYPE_PRIORITY.get(candidate.candidate_type, 4),
        candidate.complexity,
        inverse_count,
        root_count,
        division_count,
        len(candidate.branches),
    )


def rank_candidates(candidates):
    return sorted(
        candidates,
        key=lambda c: (
            not c.domain_proven,
            not c.mapping_proven,
            not (c.q < 1),
            candidate_search_key(c),
            c.q,
        ),
    )


def build_phi_candidates(F, symbols, bounds, x0, deadline=None):
    """Tìm Phi theo tầng equivalent -> isolated -> relaxation."""
    attempted = []

    def try_stage(raw_candidates, subregion_limit):
        ordered = []
        for candidate in raw_candidates:
            candidate.complexity = sum(sp.count_ops(e) for e in candidate.phi)
            ordered.append(candidate)
        ordered.sort(key=candidate_search_key)
        for candidate in ordered:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            evaluate_phi_candidate(candidate, symbols, bounds, deadline)
            attempted.append(candidate)
            if candidate.banach_proven:
                return [candidate]
        for candidate in ordered[:subregion_limit]:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            region = search_subregion(
                candidate.phi, symbols, bounds, x0, deadline=deadline
            )
            if region is None:
                continue
            evaluate_phi_candidate(candidate, symbols, region, deadline)
            if candidate.banach_proven:
                return [candidate]
        return []

    equivalent = generate_canonical_equivalent_candidates(F, symbols, deadline)
    found = try_stage(equivalent, 5)
    if found:
        return found
    if deadline is not None and time.perf_counter() >= deadline:
        return rank_candidates(attempted)

    isolated = generate_isolation_candidates(F, symbols, bounds, deadline)
    found = try_stage(isolated, 5)
    if found:
        return found
    if deadline is not None and time.perf_counter() >= deadline:
        return rank_candidates(attempted)

    relaxation = generate_relaxation_candidates(F, symbols, bounds, x0)[:4]
    found = try_stage(relaxation, 4)
    if found:
        return found
    return rank_candidates(attempted)


def search_subregion(phi, symbols, D, x0, deadline=None):
    """Tìm miền con có giới hạn thời gian và số hộp xác định."""
    found = []
    tried = 0

    def expired():
        return tried >= MAX_SUBREGION_CANDIDATES or (
            deadline is not None and time.perf_counter() >= deadline
        )

    def test_box(box):
        nonlocal tried
        if expired():
            return None
        tried += 1
        mapping, _ = check_mapping_subset(phi, symbols, box, deadline)
        data = bound_jacobian(phi, symbols, box, deadline)
        if mapping and data[5] and min(data[3], data[4]) < 1:
            return min(data[3], data[4])
        return None

    # Hộp tự nhiên: lấy biên độ giáo trình của từng thành phần, trong đó
    # sin/cos được chặn bởi [-1,1], rồi giao với D. Không nhận diện đề cụ thể.
    natural = []
    for expr, (a, b) in zip(phi, D):
        if expired():
            return None
        lo, hi, good = textbook_interval(expr, symbols, D)
        if not good or max(a, lo) >= min(b, hi):
            natural = []
            break
        natural.append((max(a, lo), min(b, hi)))
    if natural:
        q_natural = test_box(natural)
        if q_natural is not None:
            return natural
        # Co miền tự nhiên bằng D^(m+1)=D^(m) giao Phi(D^(m)).
        current = natural
        for _ in range(8):
            if expired():
                break
            next_box = []
            for expr, (a, b) in zip(phi, current):
                lo, hi, valid, _ = adaptive_interval_evaluate(
                    expr,
                    symbols,
                    current,
                    max_depth=7,
                    max_boxes=2048,
                    target_width=max((b - a) / 128, TOL),
                    deadline=deadline,
                )
                if not valid or max(a, lo) >= min(b, hi):
                    next_box = []
                    break
                next_box.append((max(a, lo), min(b, hi)))
            if not next_box:
                break
            q_next = test_box(next_box)
            if q_next is not None:
                return next_box
            if all(
                abs(a - c) + abs(b - d) <= TOL
                for (a, b), (c, d) in zip(current, next_box)
            ):
                break
            current = next_box
    # Các hộp neo tại biên với tỉ lệ đơn giản thường cho lời giải dễ chép.
    simple_scales = (1 / 2, 1 / 3, 1 / 4, 2 / 3, 3 / 4)
    if len(D) <= 4:
        for factors in itertools.product(simple_scales, repeat=len(D)):
            if expired():
                break
            for B in (
                [(a, a + s * (b - a)) for (a, b), s in zip(D, factors)],
                [(b - s * (b - a), b) for (a, b), s in zip(D, factors)],
            ):
                q_box = test_box(B)
                if q_box is not None:
                    volume = math.prod(b - a for a, b in B)
                    if q_box <= 0.5:
                        return B
                    found.append((volume, q_box, B))
    scales = (0.9, 0.75, 0.6, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05)
    factor_sets = itertools.product(scales, repeat=len(D))
    if len(D) > 4:
        factor_sets = itertools.islice(factor_sets, 5000)
    for factors in factor_sets:
        if expired():
            break
        B = [
            (max(a, x - s * (x - a)), min(b, x + s * (b - x)))
            for (a, b), x, s in zip(D, x0, factors)
        ]
        q_box = test_box(B)
        if q_box is not None:
            volume = math.prod(b - a for a, b in B)
            if q_box <= 0.5:
                return B
            found.append((volume, q_box, B))
    return max(found, key=lambda z: (z[0], -z[1]))[2] if found else None


def textbook_interval(expr, symbols, bounds):
    """Khoảng bảo thủ dễ trình bày; mọi kết quả vẫn được kiểm chứng lại."""
    env = {s: interval for s, interval in zip(symbols, bounds)}

    def mul(u, v):
        values = (u[0] * v[0], u[0] * v[1], u[1] * v[0], u[1] * v[1])
        return min(values), max(values)

    def walk(node):
        if node.is_Number:
            value = float(node)
            return value, value
        if node.is_Symbol:
            if node not in env:
                raise ValueError
            return env[node]
        if node.func is sp.Add:
            parts = [walk(a) for a in node.args]
            return sum(p[0] for p in parts), sum(p[1] for p in parts)
        if node.func is sp.Mul:
            out = (1.0, 1.0)
            for child in node.args:
                out = mul(out, walk(child))
            return out
        if node.func in (sp.sin, sp.cos):
            return -1.0, 1.0
        if node.func is sp.Pow and node.args[1].is_Integer:
            base = walk(node.args[0])
            exponent = int(node.args[1])
            if exponent >= 0:
                values = [base[0] ** exponent, base[1] ** exponent]
                if exponent % 2 == 0 and base[0] <= 0 <= base[1]:
                    values.append(0.0)
                return min(values), max(values)
        lo, hi, valid, _ = interval_evaluate(node, symbols, bounds)
        if not valid:
            raise ValueError
        return lo, hi

    try:
        lo, hi = walk(sp.sympify(expr))
        return lo, hi, math.isfinite(lo) and math.isfinite(hi)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return math.nan, math.nan, False


def format_number(value, precision=7):
    if value is None:
        return "–"
    if abs(value) < 0.5 * 10 ** (-max(precision, 1)):
        value = 0.0
    if value != 0 and (abs(value) < 10 ** (-precision) or abs(value) >= 10**precision):
        return f"{value:.{max(1, precision - 1)}e}"
    return f"{value:.{precision}f}".rstrip("0").rstrip(".") or "0"


SUBSCRIPT = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋")
SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def format_exact_or_decimal(value, precision=7, max_denominator=1000):
    del max_denominator
    return format_number(float(value), precision)


def format_exact_with_decimal(value, precision=7):
    exact = format_exact_or_decimal(value, precision)
    decimal = format_number(value, precision)
    return f"{exact} ≈ {decimal}" if "/" in exact else decimal


def _symbol_text(symbol):
    name = str(symbol)
    digits = "".join(ch for ch in name if ch.isdigit())
    prefix = name[: len(name) - len(digits)] if digits else name
    return prefix + digits.translate(SUBSCRIPT)


def needs_parentheses(expr) -> bool:
    """Một tổng cần ngoặc khi nằm trong tích hoặc phân thức."""
    return sp.sympify(expr).func in (sp.Add, sp.Mul)


def parenthesize_if_needed(expr, text) -> str:
    return f"({text})" if needs_parentheses(expr) else text


def format_fraction(numerator, denominator, symbol_map=None) -> str:
    """Định dạng phân thức như một khối, không ghép nghịch đảo vào tích."""
    symbol_map = symbol_map or {}
    numerator, denominator = sp.sympify(numerator), sp.factor(denominator)
    if denominator.could_extract_minus_sign():
        numerator, denominator = -numerator, -denominator
    # Với mẫu nguyên và tử một hạng, dùng thống nhất hệ số (p/q)f.
    if (
        denominator.is_Integer
        and abs(int(denominator)) != 1
        and numerator.func is not sp.Add
        and not numerator.is_Number
    ):
        coefficient, rest = numerator.as_coeff_Mul()
        if coefficient.is_Rational:
            combined = sp.cancel(coefficient / denominator)
            sign = "−" if combined < 0 else ""
            coefficient_text = format_math_expr(
                abs(combined), symbol_map, allow_fraction=False
            )
            rest_text = format_math_expr(rest, symbol_map, allow_fraction=False)
            return sign + f"({coefficient_text})" + rest_text
    numerator_text = format_math_expr(numerator, symbol_map, allow_fraction=False)
    denominator_text = format_math_expr(denominator, symbol_map, allow_fraction=False)
    numerator_text = (
        f"({numerator_text})" if numerator.func is sp.Add else numerator_text
    )
    denominator_text = parenthesize_if_needed(denominator, denominator_text)
    return f"{numerator_text}/{denominator_text}"


def format_product(expr, symbol_map=None) -> str:
    """Định dạng tích bằng phép nhân ngầm nhưng không gây dính `1/`."""
    symbol_map = symbol_map or {}
    coefficient, factors = sp.sympify(expr).as_coeff_mul()
    sign = "−" if coefficient < 0 else ""
    coefficient = abs(coefficient)
    pieces = []
    if coefficient != 1:
        text = format_math_expr(coefficient, symbol_map, allow_fraction=False)
        pieces.append(
            f"({text})" if coefficient.is_Rational and coefficient.q != 1 else text
        )
    for factor in factors:
        text = format_math_expr(factor, symbol_map, allow_fraction=False)
        pieces.append(parenthesize_if_needed(factor, text))
    return sign + "".join(pieces)


def format_math_expr(expr, symbol_map=None, allow_fraction=True):
    """Định dạng SymPy một dòng, tách phân thức trước khi ghép chuỗi."""
    symbol_map = symbol_map or {}
    expr = sp.sympify(expr)
    if expr.is_Symbol:
        return symbol_map.get(expr, _symbol_text(expr))
    if expr == sp.pi:
        return "π"
    if expr == sp.E:
        return "e"
    if expr.is_Integer:
        return str(int(expr)).replace("-", "−")
    if expr.is_Rational:
        sign = "−" if expr < 0 else ""
        return f"{sign}{abs(int(expr.p))}/{int(expr.q)}"
    if expr.is_Float:
        return format_number(float(expr), 8)
    if allow_fraction and expr.func in (sp.Add, sp.Mul, sp.Pow):
        try:
            numerator, denominator = sp.fraction(sp.cancel(expr))
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            numerator, denominator = sp.together(expr).as_numer_denom()
        if denominator != 1:
            return format_fraction(numerator, denominator, symbol_map)
    if expr.func is sp.Add:
        parts = []
        for term in expr.as_ordered_terms():
            negative = term.could_extract_minus_sign()
            text = format_math_expr(
                -term if negative else term, symbol_map, allow_fraction=False
            )
            parts.append(("−" if negative else "+", text))
        out = ("−" if parts[0][0] == "−" else "") + parts[0][1]
        for sign, text in parts[1:]:
            out += sign + text
        return out
    if expr.func is sp.Mul:
        return format_product(expr, symbol_map)
    if expr.func is sp.Pow:
        base, exponent = expr.args
        b = format_math_expr(base, symbol_map, allow_fraction=False)
        if exponent == sp.Rational(1, 2):
            return f"√({b})"
        if base.func is sp.Add:
            b = f"({b})"
        if exponent.is_Integer:
            if exponent < 0:
                positive = base ** abs(int(exponent))
                return format_fraction(sp.S.One, positive, symbol_map)
            return b + str(int(exponent)).translate(SUPERSCRIPT)
        return f"({b})^({format_math_expr(exponent, symbol_map, False)})"
    if expr.func in (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.Abs):
        names = {
            sp.sin: "sin",
            sp.cos: "cos",
            sp.tan: "tan",
            sp.exp: "exp",
            sp.log: "ln",
            sp.Abs: "|",
        }
        arg = format_math_expr(expr.args[0], symbol_map)
        return f"|{arg}|" if expr.func is sp.Abs else f"{names[expr.func]}({arg})"
    return sp.pretty(expr, use_unicode=True).replace("**", "^").replace("*", "")


def format_vector(vector, precision=7):
    return "(" + ", ".join(format_number(v, precision) for v in vector) + ")ᵀ"


def format_interval(interval, precision=7):
    return (
        f"[{format_exact_or_decimal(interval[0], precision)}, "
        f"{format_exact_or_decimal(interval[1], precision)}]"
    )


def format_equation_system(expressions, left_prefix="x"):
    lines = [
        "⎧ "
        + f"{left_prefix}{str(i + 1).translate(SUBSCRIPT)} = {format_math_expr(expr)}"
        for i, expr in enumerate(expressions)
    ]
    if lines:
        lines[-1] = "⎩ " + lines[-1][2:]
    return "\n".join(lines)


def print_math_table(headers, rows):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(border)
    print("|" + "|".join(f" {str(v):<{w}} " for v, w in zip(headers, widths)) + "|")
    print(border)
    for row in rows:
        print("|" + "|".join(f" {str(v):<{w}} " for v, w in zip(row, widths)) + "|")
    print(border)


def explain_interval_bound(
    expr, symbols, bounds, precision=7, certified_range=None, adaptive_used=False
):
    """Phân tích cây SymPy và sinh các bước chặn khoảng có kiểm chứng."""
    seen, steps = set(), []

    def visit(node):
        key = sp.srepr(node)
        lo, hi, valid, reason = interval_evaluate(node, symbols, bounds)
        if not valid:
            return lo, hi, False, reason
        if node.is_Symbol:
            if key not in seen:
                steps.append(
                    f"{format_math_expr(node)} ∈ {format_interval((lo, hi), precision)}."
                )
        elif node.is_number and not node.free_symbols:
            pass
        elif not node.is_Number:
            child_ranges = []
            children = node.args[:1] if node.func is sp.Pow else node.args
            for child in children:
                clo, chi, good, why = visit(child)
                if not good:
                    return clo, chi, False, why
                child_ranges.append((clo, chi))
            if key not in seen:
                text = format_math_expr(node)
                if node.func in (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.Abs):
                    source = format_interval(child_ranges[0], precision)
                    steps.append(
                        f"{format_math_expr(node.args[0])} ∈ {source} ⇒ {text} ∈ {format_interval((lo, hi), precision)}."
                    )
                elif node.func is sp.Pow and node.args[1] == sp.Rational(1, 2):
                    steps.append(
                        f"{format_math_expr(node.args[0])} ∈ {format_interval(child_ranges[0], precision)} ⇒ {text} ∈ {format_interval((lo, hi), precision)}."
                    )
                else:
                    sources = ", ".join(
                        format_interval(r, precision) for r in child_ranges
                    )
                    steps.append(
                        f"Từ các khoảng {sources}, suy ra {text} ∈ {format_interval((lo, hi), precision)}."
                    )
        seen.add(key)
        return lo, hi, True, ""

    lo, hi, valid, reason = visit(sp.sympify(expr))
    final_range = certified_range if certified_range is not None else (lo, hi)
    if certified_range is not None and adaptive_used:
        steps.append("Sau khi chia miền thích nghi, thu được khoảng bao chặt hơn:")
        steps.append(
            f"{format_math_expr(expr)}(D) ⊂ {format_interval(certified_range, precision)}."
        )
    return steps, final_range, valid, reason


def explain_absolute_bound(expr, symbols, bounds, precision=7):
    """Sinh chặn giáo trình |expr|<=M; kèm chặn khoảng nghiêm ngặt."""
    lo, hi, valid, reason = interval_evaluate(expr, symbols, bounds)
    if not valid:
        return [], math.inf, math.inf, False, reason
    rigorous = max(abs(lo), abs(hi))
    details = []

    def simple(node):
        if node.is_Number:
            return abs(float(node))
        if node.is_Symbol:
            i = list(symbols).index(node)
            value = max(abs(bounds[i][0]), abs(bounds[i][1]))
            details.append(
                f"|{format_math_expr(node)}| ≤ {format_exact_or_decimal(value, precision)}"
            )
            return value
        if node.func is sp.Mul:
            values = [simple(a) for a in node.args]
            return math.prod(values)
        if node.func is sp.Add:
            return sum(simple(a) for a in node.args)
        if node.func in (sp.sin, sp.cos):
            details.append(f"|{format_math_expr(node)}| ≤ 1")
            return 1.0
        if node.func is sp.Pow and node.args[1].is_Integer:
            exponent = int(node.args[1])
            if exponent >= 0:
                return simple(node.args[0]) ** exponent
        if node.func is sp.Abs:
            return simple(node.args[0])
        nlo, nhi, good, _ = interval_evaluate(node, symbols, bounds)
        return max(abs(nlo), abs(nhi)) if good else math.inf

    textbook = simple(sp.sympify(expr))
    steps = list(dict.fromkeys(details))
    steps.append(
        f"Do đó |{format_math_expr(expr)}| ≤ {format_exact_or_decimal(textbook, precision)}."
    )
    return steps, rigorous, textbook, math.isfinite(textbook), ""


def prepare_textbook_bounds(candidate, symbols, bounds, precision=7):
    """Tự tạo M từ đạo hàm; ưu tiên chặn dễ chép nếu vẫn chứng minh co."""
    derivs, intervals, rigorous_M, _, _, valid = candidate.jacobian_data
    simple_M, explanations = [], []
    for i, row in enumerate(derivs):
        mrow, erow = [], []
        for j, derivative in enumerate(row):
            steps, _, simple_bound, good, _ = explain_absolute_bound(
                derivative, symbols, bounds, precision
            )
            lo, hi, rigorous_good, _ = intervals[i][j]
            rigorous_bound = max(abs(lo), abs(hi)) if rigorous_good else math.inf
            simple_good = (
                good
                and math.isfinite(simple_bound)
                and simple_bound + TOL >= rigorous_bound
            )
            mrow.append(simple_bound if simple_good else math.inf)
            erow.append(steps if simple_good else [])
        simple_M.append(mrow)
        explanations.append(erow)
    qi = max((sum(row) for row in simple_M), default=0.0)
    q1 = max(
        (
            sum(simple_M[i][j] for i in range(len(simple_M)))
            for j in range(len(simple_M))
        ),
        default=0.0,
    )
    if qi < 1:  # chuẩn hàng thường thuận tiện nhất để trình bày
        chosen_M, q, norm = simple_M, qi, "vo cung"
    elif q1 < 1:
        chosen_M, q, norm = simple_M, q1, "1"
    else:
        rqi = max((sum(row) for row in rigorous_M), default=0.0)
        rq1 = max(
            (
                sum(rigorous_M[i][j] for i in range(len(rigorous_M)))
                for j in range(len(rigorous_M))
            ),
            default=0.0,
        )
        chosen_M, q, norm = (
            rigorous_M,
            min(rqi, rq1),
            ("vo cung" if rqi <= rq1 else "1"),
        )
        explanations = []
        for i, row in enumerate(derivs):
            erow = []
            for j, derivative in enumerate(row):
                lo, hi, good, _ = intervals[i][j]
                erow.append(
                    [
                        f"{format_math_expr(derivative)}(D) ⊂ {format_interval((lo, hi), precision)}, "
                        f"nên |{format_math_expr(derivative)}| ≤ {format_exact_or_decimal(rigorous_M[i][j], precision)}."
                    ]
                    if good
                    else []
                )
            explanations.append(erow)
        qi, q1 = rqi, rq1
    candidate.q_inf, candidate.q_one, candidate.q = qi, q1, q
    candidate.selected_norm = norm
    candidate.jacobian_data = (derivs, intervals, chosen_M, qi, q1, valid)
    candidate.derivative_ranges = intervals
    candidate.bound_explanations = explanations
    return candidate


def vector_norm(v, norm):
    return sum(abs(x) for x in v) if norm == "1" else max(map(abs, v), default=0.0)


def bound_system_lipschitz(F, symbols, bounds):
    """Chặn ||J_F|| vô cùng và thang đo F bằng số học khoảng thích nghi."""
    if F is None:
        return None, None
    row_sums, scale = [], 0.0
    for equation in F:
        row = 0.0
        lo, hi, good, _ = adaptive_interval_evaluate(
            equation, symbols, bounds, max_depth=7, max_boxes=2048, target_width=None
        )
        if not good:
            return math.inf, math.inf
        scale = max(scale, abs(lo), abs(hi))
        for symbol in symbols:
            derivative = sp.diff(equation, symbol)
            dlo, dhi, valid, _ = adaptive_interval_evaluate(
                derivative,
                symbols,
                bounds,
                max_depth=7,
                max_boxes=2048,
                target_width=None,
            )
            if not valid:
                return math.inf, math.inf
            row += max(abs(dlo), abs(dhi))
        row_sums.append(row)
    return max(row_sums, default=0.0), scale


def evaluate_real_vector(exprs, symbols, point):
    values = []
    for i, expr in enumerate(exprs):
        try:
            z = complex(sp.N(expr.subs(dict(zip(symbols, point))), 17))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ArithmeticError(
                f"φ{str(i + 1).translate(SUBSCRIPT)} không tính được: {exc}"
            ) from exc
        if abs(z.imag) > 1e-12 or not math.isfinite(z.real):
            raise ArithmeticError(
                f"φ{str(i + 1).translate(SUBSCRIPT)} không phải số thực hữu hạn"
            )
        values.append(z.real)
    return values


def compute_error_bounds(q, norm, x_new, x_old, phi_at_new):
    if not math.isfinite(q) or q >= 1:
        return None, None, None
    e1 = q / (1 - q) * vector_norm([a - b for a, b in zip(x_new, x_old)], norm)
    e2 = 1 / (1 - q) * vector_norm([a - b for a, b in zip(phi_at_new, x_new)], norm)
    return e1, e2, min(e1, e2)


def priori_iteration_count(q, epsilon, first_difference):
    """So buoc tien nghiem: n=ceil(log_q(eps*(1-q)/||Phi(X0)-X0||))."""
    if not (math.isfinite(q) and math.isfinite(epsilon) and math.isfinite(first_difference)):
        raise ValueError("q, epsilon va first_difference phai huu han.")
    if epsilon <= 0 or first_difference < 0 or not (0 <= q < 1):
        raise ValueError("Can epsilon>0, first_difference>=0 va 0<=q<1.")
    if first_difference == 0 or q == 0:
        return 1
    ratio = epsilon * (1 - q) / first_difference
    if ratio >= 1:
        return 1
    return max(1, math.ceil(math.log(ratio) / math.log(q)))


def fixed_point_iteration_priori(
    phi,
    symbols,
    x0,
    bounds,
    q,
    norm,
    epsilon,
    **kwargs,
):
    first = evaluate_real_vector(phi, symbols, x0)
    first_difference = vector_norm([a - b for a, b in zip(first, x0)], norm)
    steps = priori_iteration_count(q, epsilon, first_difference)
    result = fixed_point_iteration(
        phi,
        symbols,
        x0,
        bounds,
        q,
        norm,
        epsilon=epsilon,
        exact_steps=steps,
        **kwargs,
    )
    result["priori_steps"] = steps
    result["priori_first_difference"] = first_difference
    return result


def fixed_point_iteration(
    phi,
    symbols,
    x0,
    bounds,
    q,
    norm,
    F=None,
    epsilon=None,
    relative=False,
    exact_steps=None,
    max_iter=1000,
    banach_proven=False,
    equation_residual_tolerance=None,
    F_lipschitz=None,
    F_scale=None,
):
    """Lap don dong thoi; khong dung thanh phan moi tinh trong cung buoc."""
    x = list(x0)
    history = [tuple(x)]
    residual_tol = equation_residual_tolerance
    if residual_tol is None and epsilon is not None:
        residual_tol = epsilon

    def residual_at(point):
        if F is None:
            return None
        return vector_norm(evaluate_real_vector(F, symbols, point), "vo cung")

    round_tolerance = 100 * sys.float_info.epsilon * (1 + (F_scale or 0.0))

    def residual_consistent(residual, error_bound):
        if F is None:
            return True
        if residual is None or not math.isfinite(residual):
            return False
        if error_bound is None:
            return residual <= round_tolerance
        if F_lipschitz is None or not math.isfinite(F_lipschitz):
            return False
        return residual <= F_lipschitz * error_bound + round_tolerance

    # Hang k=0 va nhan biet ngay diem dau da la nghiem.
    try:
        phi0 = evaluate_real_vector(phi, symbols, x)
        fixed0 = vector_norm([a - b for a, b in zip(phi0, x)], norm)
        residual0 = residual_at(x)
    except ArithmeticError as exc:
        return {"x": x, "rows": [], "status": "invalid_value", "reason": str(exc)}
    rows = [(0, list(x), None, residual0, None, None, None)]
    # Khong dung epsilon cua nghiem nhu mot dung sai phan du diem bat dong:
    # voi q gan 1, phan du nho van co the tuong ung sai so nghiem lon.
    residual_ok0 = residual_consistent(residual0, 0.0)
    if exact_steps is None and fixed0 <= TOL and residual_ok0:
        status = "certified" if banach_proven else "numerical_only"
        return {
            "x": x,
            "rows": rows,
            "status": status,
            "reason": "Điểm ban đầu đã là nghiệm.",
            "steps": 0,
        }

    limit = exact_steps if exact_steps is not None else max_iter
    status, reason = "max_iter", "Đã đạt giới hạn số bước lặp."
    for k in range(1, limit + 1):
        try:
            # Tat ca thanh phan new deu duoc tinh tu cung vector x.
            new = evaluate_real_vector(phi, symbols, x)
            phi_at_new = evaluate_real_vector(phi, symbols, new)
            residual = residual_at(new)
        except ArithmeticError as exc:
            status, reason = "invalid_value", str(exc)
            break
        if any(not (a - TOL <= y <= b + TOL) for y, (a, b) in zip(new, bounds)):
            status = "left_domain"
            reason = "Dãy lặp đã ra ngoài miền dùng để chứng minh hội tụ."
            break
        diff = vector_norm([a - b for a, b in zip(new, x)], norm)
        fixed_res = vector_norm([a - b for a, b in zip(phi_at_new, new)], norm)
        e1 = e2 = error = rel = None
        if banach_proven:
            e1, e2, _ = compute_error_bounds(q, norm, new, x, phi_at_new)
            error = e1  # Công thức hậu nghiệm theo giáo trình.
            new_norm = vector_norm(new, norm)
            rel = (
                error / (new_norm - error)
                if error is not None and new_norm > error
                else None
            )
            if q == 0 and fixed_res <= TOL:
                e1 = e2 = error = rel = 0.0
        rows.append((k, list(new), error, residual, e1, e2, rel))
        x = new
        if exact_steps is None and epsilon is not None:
            residual_ok = residual_consistent(residual, error)
            error_ok = (
                (rel is not None and rel <= epsilon)
                if relative
                else (error is not None and error <= epsilon)
            )
            if q == 0 and fixed_res <= TOL:
                error_ok = True
            if banach_proven and error_ok and residual_ok:
                status, reason = "certified", "Đạt sai số hậu nghiệm và phần dư hệ."
                break
            if (
                not banach_proven
                and diff <= epsilon
                and fixed_res <= epsilon
                and residual_ok
            ):
                status = "numerical_only"
                reason = "Chi dat tieu chuan so; khong co chung minh Banach."
                break
        if exact_steps is None and tuple(new) in history[-4:]:
            status, reason = "cycle", "Phát hiện chu kỳ hoặc đình trệ số học."
            break
        history.append(tuple(new))
    else:
        if exact_steps is not None:
            status, reason = "fixed_steps", f"Đã thực hiện đúng {exact_steps} bước."
    return {
        "x": x,
        "rows": rows,
        "status": status,
        "reason": reason,
        "steps": rows[-1][0] if rows else 0,
    }


def validate_printed_proof(candidate, bounds):
    """Chặn mọi lời in nếu dữ liệu chứng minh nội bộ không đồng nhất."""
    if not candidate.domain_proven or not candidate.mapping_proven:
        return False
    if candidate.mapping_ranges is None or candidate.proof_bounds is None:
        return False
    if len(candidate.mapping_ranges) != len(bounds):
        return False
    if len(candidate.proof_bounds) != len(bounds):
        return False
    if any(
        abs(a - c) > TOL or abs(b - d) > TOL
        for (a, b), (c, d) in zip(candidate.proof_bounds, bounds)
    ):
        return False
    for (lo, hi, valid, _), (a, b) in zip(candidate.mapping_ranges, bounds):
        if not valid or not (math.isfinite(lo) and math.isfinite(hi)):
            return False
        if not (a - TOL <= lo and hi <= b + TOL):
            return False
    if candidate.jacobian_data is None or candidate.derivative_ranges is None:
        return False
    for row in candidate.derivative_ranges:
        for lo, hi, valid, _ in row:
            if not valid or not (math.isfinite(lo) and math.isfinite(hi)):
                return False
    M = candidate.jacobian_data[2]
    if any(not math.isfinite(value) for row in M for value in row):
        return False
    q_inf = max((sum(row) for row in M), default=0.0)
    q_one = max(
        (sum(M[i][j] for i in range(len(M))) for j in range(len(M[0]) if M else 0)),
        default=0.0,
    )
    expected = q_inf if candidate.selected_norm == "vo cung" else q_one
    return (
        candidate.selected_norm in ("vo cung", "1")
        and math.isfinite(candidate.q)
        and candidate.q < 1
        and abs(candidate.q - expected) <= 100 * TOL * max(1, abs(expected))
    )


def print_exam_proof(
    candidate,
    symbols,
    bounds,
    F=None,
    direct=False,
    original_bounds=None,
    precision=7,
    stop_kind="absolute",
):
    if not validate_printed_proof(candidate, bounds):
        print(
            "Dữ liệu chứng minh nội bộ không nhất quán, không thể xuất lời giải bảo đảm."
        )
        return False
    print("\nPHẦN 1. INPUT – OUTPUT – THUẬT TOÁN")
    print("Input:")
    print("  • Hệ phương trình hoặc dạng lặp đã cho")
    if stop_kind == "fixed":
        print("  • Miền D, số bước k và giá trị ban đầu t₀")
    else:
        print("  • Miền D, sai số δ và giá trị ban đầu t₀")
    print("Output:")
    if stop_kind == "relative":
        print("  • Nghiệm gần đúng với sai số tương đối không vượt quá δ.")
    elif stop_kind == "absolute":
        print("  • Nghiệm gần đúng với sai số tuyệt đối không vượt quá δ.")
    else:
        print("  • Giá trị gần đúng sau đúng k bước lặp.")
    print("Thuật toán:")
    print("B1. Kiểm tra Φ(D) ⊂ D.  B2. Tính JΦ và hệ số co q.")
    print("B3. Chọn t₀ ∈ D.  B4. Lặp tₖ = Φ(tₖ₋₁).")
    if stop_kind == "relative":
        if candidate.q == 0:
            print("B5. Vì q=0, dừng khi Φ(t₁)=t₁.  B6. Trả ra nghiệm gần đúng.")
        else:
            print(
                "B5. Dừng khi ||tₖ−tₖ₋₁||/||tₖ|| ≤ (1−q)δ/q.  B6. Trả ra nghiệm gần đúng."
            )
    elif stop_kind == "absolute":
        print("B5. Dừng theo sai số hậu nghiệm q||tₖ−tₖ₋₁||/(1−q) ≤ δ.  B6. Trả ra nghiệm gần đúng.")
    elif stop_kind == "priori":
        print(
            "B5. Tính n = ceil(log_q(δ(1−q)/||t₁−t₀||)).  "
            "B6. Lặp đúng n bước rồi trả ra nghiệm gần đúng."
        )
    else:
        print("B5. Thực hiện đúng k bước lặp.  B6. Trả ra giá trị gần đúng.")

    print("\nPHẦN 2. ĐƯA HỆ VỀ DẠNG LẶP")
    if F is not None:
        print("Đưa hệ phương trình về dạng lặp:")
        for i, equation in enumerate(F):
            print(f"  F{i + 1}(x) = {format_math_expr(equation)} = 0")
    else:
        print("Xét dạng lặp đã cho:")
    for i, expr in enumerate(candidate.phi):
        print(f"  {_symbol_text(symbols[i])} = {format_math_expr(expr)}")
    if not direct:
        print("Trong các cách biến đổi hợp lệ, chọn dạng trên vì có hệ số co nhỏ nhất")
        print("và thuận tiện cho quá trình lặp.")

    print("\nPHẦN 3. KIỂM TRA Φ(D) ⊂ D")
    if original_bounds is not None and original_bounds != bounds:
        print(
            f"Trên miền ban đầu D = {' × '.join(format_interval(v, precision) for v in original_bounds)},"
        )
        old_ok, old_ranges, old_meta = check_mapping_subset_with_meta(
            candidate.phi, symbols, original_bounds
        )
        if old_meta["method"] == "adaptive_subdivision":
            print("Dùng chia miền thích nghi để đánh giá Φ trên miền ban đầu.")
        violated = []
        for i, ((lo, hi, valid, _), target) in enumerate(
            zip(old_ranges, original_bounds)
        ):
            if valid:
                old_steps, _, _, _ = explain_interval_bound(
                    candidate.phi[i],
                    symbols,
                    original_bounds,
                    precision,
                    (lo, hi),
                    old_meta["method"] == "adaptive_subdivision",
                )
                for step in old_steps:
                    print("  " + step)
            contained = valid and target[0] - TOL <= lo and hi <= target[1] + TOL
            relation = "⊂" if contained else "⊄"
            connector = "và" if contained else "nhưng"
            print(
                f"  φ{i + 1}(D) ⊂ {format_interval((lo, hi), precision)}, "
                f"{connector} {format_interval((lo, hi), precision)} {relation} {format_interval(target, precision)}."
            )
            if not contained:
                violated.append(i + 1)
        if not old_ok:
            print(
                "Các thành phần vi phạm: " + ", ".join(f"φ{i}" for i in violated) + "."
            )
            print("Vì vậy Φ(D) không nằm trong D.")
        print(
            f"Chọn miền con D′ = {' × '.join(format_interval(v, precision) for v in bounds)} ⊂ D."
        )
        print("Kiểm tra điều kiện hội tụ trên miền D′.")
    ranges = candidate.mapping_ranges
    if candidate.mapping_method == "adaptive_subdivision":
        boxes = candidate.mapping_meta.get("boxes_used", 0)
        print("Chia miền đang xét thành các hộp con và đánh giá Φ trên từng hộp.")
        print(
            (
                f"Dùng chia miền thích nghi ({boxes} hộp), hợp các khoảng thu được:"
                if boxes
                else "Dùng chia miền thích nghi, hợp các khoảng thu được:"
            )
        )
    for i, ((lo, hi, valid, reason), (a, b)) in enumerate(zip(ranges, bounds)):
        if valid:
            steps, _, _, _ = explain_interval_bound(
                candidate.phi[i],
                symbols,
                bounds,
                precision,
                (lo, hi),
                candidate.mapping_method == "adaptive_subdivision",
            )
            for step in steps:
                print("  " + step)
            print(
                f"  φ{i + 1} = {format_math_expr(candidate.phi[i])} ∈ {format_interval((lo, hi), precision)}"
            )
            contained = a - TOL <= lo and hi <= b + TOL
            relation = "⊂" if contained else "⊄"
            print(
                f"  và {format_interval((lo, hi), precision)} {relation} {format_interval((a, b), precision)}."
            )
        else:
            print(f"Không xác nhận được φ{i + 1} trên miền: {reason}.")
    if candidate.mapping_proven:
        print("Suy ra Φ(D′) ⊂ D′." if original_bounds != bounds else "Suy ra Φ(D) ⊂ D.")

    print("\nPHẦN 4. ĐẠO HÀM RIÊNG VÀ HỆ SỐ CO")
    derivs, _, M, qi, q1, _ = candidate.jacobian_data
    print("Ma trận Jacobi JΦ(x) =")
    for row in derivs:
        print("  [" + ", ".join(format_math_expr(v) for v in row) + "]")
    for i, row in enumerate(derivs):
        for j, derivative in enumerate(row):
            print(
                f"  ∂φ{i + 1}/∂{_symbol_text(symbols[j])} = {format_math_expr(derivative)}."
            )
            explanations = getattr(candidate, "bound_explanations", None)
            steps = (
                explanations[i][j]
                if explanations is not None
                else explain_absolute_bound(derivative, symbols, bounds, precision)[0]
            )
            for step in steps:
                print("    " + step)
    if candidate.selected_norm == "vo cung":
        sums = [sum(row) for row in M]
        print("Theo chuẩn hàng:")
        print(
            "  q = ||JΦ||∞ ≤ max{"
            + ", ".join(format_number(v, precision) for v in sums)
            + f"}} = {format_exact_with_decimal(candidate.q, precision)} < 1."
        )
    else:
        sums = [sum(M[i][j] for i in range(len(M))) for j in range(len(M))]
        print("Theo chuẩn cột:")
        print(
            "  q = ||JΦ||₁ ≤ max{"
            + ", ".join(format_number(v, precision) for v in sums)
            + f"}} = {format_exact_with_decimal(candidate.q, precision)} < 1."
        )
    print(
        f"Vậy Φ là ánh xạ co với hệ số co q = {format_exact_with_decimal(candidate.q, precision)}."
    )

    print("\nPHẦN 5. KẾT LUẬN HỘI TỤ")
    if candidate.banach_proven:
        print(
            "Theo định lý ánh xạ co Banach, hệ có duy nhất một nghiệm trong miền đã chọn"
        )
        print("và dãy lặp đơn hội tụ tới nghiệm đó với mọi t₀ thuộc miền này.")
    else:
        print("Chưa đủ điều kiện để áp dụng định lý ánh xạ co Banach.")
    return True


def print_iteration_formula(
    candidate, symbols, x0, precision=7, bounds=None, midpoint=False
):
    print("\nPHẦN 6. CHỌN GIÁ TRỊ BAN ĐẦU")
    if midpoint and bounds is not None:
        terms = [
            f"({format_exact_or_decimal(a, precision)}+{format_exact_or_decimal(b, precision)})/2"
            for a, b in bounds
        ]
        print(f"t₀ = ({', '.join(terms)})ᵀ = {format_vector(x0, precision)}.")
    else:
        print(f"t₀ = {format_vector(x0, precision)} thuộc miền đã chọn.")
    print("\nPHẦN 7. CÔNG THỨC LẶP")
    print("tₖ = Φ(tₖ₋₁), cụ thể:")
    previous = {symbol: sp.Symbol(f"__prev_{i}") for i, symbol in enumerate(symbols)}
    symbol_map = {previous[symbol]: f"{_symbol_text(symbol)},ₖ₋₁" for symbol in symbols}
    for i, expr in enumerate(candidate.phi):
        rhs = expr.xreplace(previous)
        print(f"  {_symbol_text(symbols[i])},ₖ = {format_math_expr(rhs, symbol_map)}")
    if candidate.banach_proven:
        print("Chỉ dùng đánh giá hậu nghiệm:")
        if candidate.q == 0:
            print("  Vì q=0 nên Φ là ánh xạ hằng.")
            print("  Sau một bước lặp, nếu Φ(t₁)=t₁ thì sai số hậu nghiệm bằng 0.")
        else:
            print("  Eₖ = q/(1-q) · ||tₖ-tₖ₋₁||.")
            print("  ||tₖ-t*|| ≤ Eₖ.")
            print("  Nếu ||tₖ||>Eₖ thì ||tₖ-t*||/||t*|| ≤ Eₖ/(||tₖ||-Eₖ).")


def iteration_report_rows(result, norm):
    """Đặt tên đúng bốn đại lượng dùng trong bảng để tránh nhầm nhãn."""
    report = []
    previous = None
    for k, x, error, residual, _e1, _e2, relative_bound in result["rows"]:
        step_difference = (
            vector_norm([a - b for a, b in zip(x, previous)], norm)
            if previous is not None
            else None
        )
        report.append({
            "k": k,
            "x": x,
            "step_difference": step_difference,
            "absolute_error_bound": error,
            "relative_error_bound": relative_bound,
            "residual_norm": residual,
        })
        previous = x
    return report


def print_iteration_table(result, precision, stop_kind, norm):
    print("\nPHẦN 8. BẢNG LẶP")
    headers = [
        "k",
        "tₖ",
        "step_difference",
        "absolute_error_bound Eₖ",
        "relative_error_bound",
        "residual_norm",
    ]
    rows = []
    for item in iteration_report_rows(result, norm):
        rows.append(
            [
                str(item["k"]),
                format_vector(item["x"], precision),
                format_number(item["step_difference"], precision),
                format_number(item["absolute_error_bound"], precision),
                format_number(item["relative_error_bound"], precision),
                format_number(item["residual_norm"], precision),
            ]
        )
    print_math_table(headers, rows)


def looks_like_direct_phi(expressions, symbols, bounds, seed, deadline=None):
    """Kiểm tra ngầm khả năng người dùng đã nhập Phi thay cho F."""
    candidate = evaluate_phi_candidate(
        Candidate(list(expressions), "Nhập trực tiếp", candidate_type="direct"),
        symbols,
        bounds,
        deadline,
    )
    if candidate.banach_proven:
        return True
    region = search_subregion(candidate.phi, symbols, bounds, seed, deadline)
    if region is None:
        return False
    return evaluate_phi_candidate(candidate, symbols, region, deadline).banach_proven


def exam_main():
    print("=== GIẢI HỆ PHƯƠNG TRÌNH BẰNG PHƯƠNG PHÁP LẶP ĐƠN ĐỒNG THỜI ===")
    n = read_int("Số ẩn n = ", 1)
    symbols = sp.symbols(f"x1:{n + 1}")
    print("1. Nhập hệ F(x)=0")
    print("   Ví dụ: F1 = x1 - (1/3)sin(x1*x2)")
    print("           F2 = x2 - (1/4)cos(x1^2+x2^2)")
    print("2. Nhập trực tiếp dạng lặp x=Phi(x)")
    print("   Ví dụ: phi_1 = (1/3)sin(x1*x2)")
    print("           phi_2 = (1/4)cos(x1^2+x2^2)")
    mode = read_int("Lựa chọn = ", 1)
    while mode not in (1, 2):
        mode = read_int("Chỉ chọn 1 hoặc 2: ", 1)
    F = read_system(n, symbols) if mode == 1 else None
    phi_direct = None
    if mode == 2:
        phi_direct = []
        for i in range(n):
            while True:
                try:
                    phi_direct.append(parse_expression(input(f"φ_{i + 1} = "), symbols))
                    break
                except InputError as exc:
                    print(f"Lỗi: {exc}")
    print("Nhập miền hộp D:")
    bounds = read_bounds(n)
    precision = read_int("Số chữ số hiển thị = ", 0)
    seed = [(a + b) / 2 for a, b in bounds]
    search_deadline = time.perf_counter() + DEFAULT_SEARCH_SECONDS
    candidates = (
        [
            evaluate_phi_candidate(
                Candidate(phi_direct, "Nhập trực tiếp", candidate_type="direct"),
                symbols,
                bounds,
                search_deadline,
            )
        ]
        if phi_direct is not None
        else build_phi_candidates(F, symbols, bounds, seed, search_deadline)
    )
    if (
        mode == 1
        and not any(c.banach_proven for c in candidates)
        and looks_like_direct_phi(F, symbols, bounds, seed, search_deadline)
    ):
        print("Các biểu thức vừa nhập có vẻ là các thành phần của Phi,")
        print("không phải các vế trái F_i(x)=0.\n")
        answer = (
            input(
                "Có hiểu chúng là x1 = biểu thức 1, ..., xn = biểu thức n không? [C/k]: "
            )
            .strip()
            .lower()
        )
        if answer in ("c", "co", "có", "y", "yes"):
            phi_direct, F, mode = list(F), None, 2
            candidates = [
                evaluate_phi_candidate(
                    Candidate(phi_direct, "Nhập trực tiếp", candidate_type="direct"),
                    symbols,
                    bounds,
                    search_deadline,
                )
            ]
        else:
            print("Nếu phương trình là x1 = phi_1 thì cần nhập:")
            print("F1 = x1 - phi_1.")
            return
    choices = []
    proven_on_D = [c for c in candidates if c.banach_proven]
    nonproven = [c for c in candidates if not c.banach_proven]
    nonproven.sort(key=lambda c: (TYPE_PRIORITY.get(c.candidate_type, 3), c.complexity))
    scan = [(c, c.proof_bounds or bounds) for c in proven_on_D[:20]]
    for candidate in nonproven[:10]:
        if time.perf_counter() >= search_deadline:
            break
        B = search_subregion(candidate.phi, symbols, bounds, seed, search_deadline)
        if B is not None:
            scan.append((candidate, B))
            if candidate.candidate_type == "equivalent":
                break
    for candidate, region in scan:
        regions = [region]
        for region in regions:
            evaluated = evaluate_phi_candidate(
                candidate, symbols, region, search_deadline
            )
            if not evaluated.banach_proven:
                continue
            volume = math.prod(b - a for a, b in region)
            endpoint_cost = sum(
                Fraction(v).limit_denominator(100).denominator
                for interval in region
                for v in interval
            )
            choices.append(
                (
                    TYPE_PRIORITY.get(evaluated.candidate_type, 4),
                    round(evaluated.q, 8),
                    -volume,
                    evaluated.complexity,
                    endpoint_cost,
                    evaluated,
                    region,
                )
            )
    if not choices:
        if time.perf_counter() >= search_deadline:
            print("Quá trình tự tìm dạng lặp đã đạt giới hạn thời gian.")
            print("Hãy nhập trực tiếp dạng lặp Phi nếu đã biến đổi được hệ.")
            return
        print("Không tìm được dạng lặp đơn thỏa điều kiện hội tụ trên miền đã cho.")
        print("Chưa chứng minh được điều kiện trên toàn miền bằng số học khoảng.")
        print("Chương trình chưa tìm được miền con thích hợp trong giới hạn tìm kiếm.")
        print("Không thể tiếp tục lời giải có bảo đảm bằng phương pháp lặp đơn.")
        return
    _, _, _, _, _, chosen, active_bounds = min(choices, key=lambda z: z[:5])

    chosen = prepare_textbook_bounds(chosen, symbols, active_bounds, precision)
    print(
        "\nChọn giá trị ban đầu:\n1. Nhập t₀\n2. Tự chọn trung điểm miền cuối cùng (mặc định)"
    )
    initial_choice = read_int("Lựa chọn = ", 1)
    if initial_choice == 1:
        print("Nhập t₀:")
        x0 = read_vector(n, "x")
        if any(not (a <= x <= b) for x, (a, b) in zip(x0, active_bounds)):
            raise InputError("t₀ phải thuộc miền cuối cùng.")
    else:
        x0 = [(a + b) / 2 for a, b in active_bounds]
    print(
        "\nYêu cầu dừng:\n1. Sai số tiên nghiệm\n2. Sai số hậu nghiệm\n3. Đúng k bước\n4. Sai số tương đối nếu đang hỗ trợ"
    )
    stop = read_int("Lựa chọn = ", 1)
    epsilon = None
    relative = False
    steps = None
    max_iter = 1000
    if stop in (1, 2, 4):
        epsilon = read_number("δ = ", positive=True)
        relative = stop == 4
        max_iter = read_int("max_iter = ", 1)
    elif stop == 3:
        steps = read_int("k = ", 0)
    else:
        raise InputError("Lựa chọn điều kiện dừng không hợp lệ.")

    stop_kind = "priori" if stop == 1 else ("relative" if relative else ("absolute" if epsilon is not None else "fixed"))
    if not print_exam_proof(
        chosen, symbols, active_bounds, F, mode == 2, bounds, precision, stop_kind
    ):
        return
    print_iteration_formula(
        chosen, symbols, x0, precision, active_bounds, midpoint=initial_choice != 1
    )
    if relative and chosen.q != 0:
        threshold = (1 - chosen.q) / chosen.q * epsilon
        print(
            f"Ngưỡng dừng tương đối: (1-q)/q · δ = {format_number(threshold, precision)}."
        )
    F_lipschitz, F_scale = bound_system_lipschitz(F, symbols, active_bounds)
    if stop == 1:
        result = fixed_point_iteration_priori(
            chosen.phi,
            symbols,
            x0,
            active_bounds,
            chosen.q,
            chosen.selected_norm,
            epsilon,
            F=F,
            banach_proven=True,
            equation_residual_tolerance=epsilon,
            F_lipschitz=F_lipschitz,
            F_scale=F_scale,
        )
    else:
        result = fixed_point_iteration(
            chosen.phi,
            symbols,
            x0,
            active_bounds,
            chosen.q,
            chosen.selected_norm,
            F,
            epsilon,
            relative,
            steps,
            max_iter,
            banach_proven=True,
            equation_residual_tolerance=epsilon if epsilon is not None else None,
            F_lipschitz=F_lipschitz,
            F_scale=F_scale,
        )
    print_iteration_table(result, precision, stop_kind, chosen.selected_norm)
    print("\nPHẦN 9. KIỂM TRA ĐIỀU KIỆN DỪNG")
    last = result["rows"][-1] if result["rows"] else None
    if epsilon is not None and last is not None:
        if result.get("steps") == 0 and result.get("status") == "certified":
            print("t₀ = Φ(t₀), do đó t₀ đã là điểm bất động.")
            print(f"E₀ = 0 ≤ δ = {format_number(epsilon, precision)}.")
            if relative:
                print("Sai số tương đối bằng 0.")
        elif chosen.q == 0:
            print("Vì q=0 nên Φ là ánh xạ hằng.")
            print("Sau một bước lặp thu được điểm cố định chính xác của ánh xạ.")
            print(
                f"Sai số hậu nghiệm E₁ = 0 ≤ δ = {format_number(epsilon, precision)}."
            )
            if relative:
                print("Sai số tương đối bằng 0.")
        elif relative:
            threshold = (1 - chosen.q) / chosen.q * epsilon
            print(
                f"||tₖ-tₖ₋₁||/||tₖ|| = {format_number(last[6], precision)} "
                f"≤ (1-q)/q · δ = {format_number(threshold, precision)}."
            )
        else:
            print(
                f"Eₖ = q/(1-q)||tₖ-tₖ₋₁|| = {format_number(last[2], precision)} "
                f"≤ δ = {format_number(epsilon, precision)}."
            )
    print("\nPHẦN 10. KẾT QUẢ")
    status = result["status"]
    if status == "certified":
        print(
            f"Nghiệm gần đúng:\nt* ≈ t_{result['steps']} = {format_vector(result['x'], precision)}."
        )
        print(
            f"Sai số {'tương đối' if relative else 'tuyệt đối'} không vượt quá δ = {format_number(epsilon, precision)}."
        )
        print(f"Dãy lặp dừng sau {result['steps']} bước.")
    elif status == "fixed_steps":
        print(f"Giá trị gần đúng sau đúng {result['steps']} bước:")
        print(f"t_{result['steps']} = {format_vector(result['x'], precision)}.")
    else:
        print("Quá trình chưa thỏa điều kiện để xác nhận nghiệm.")
        print(
            f"Vector cuối cùng trước khi dừng: {format_vector(result['x'], precision)}."
        )


if __name__ == "__main__":
    try:
        exam_main()
    except (InputError, KeyboardInterrupt, EOFError) as exc:
        print(f"\nLỗi: {exc}")
