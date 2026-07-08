"""Chặn truyền sai số một biến qua định lý giá trị trung bình."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import sympy as sp


@dataclass
class PropagatedBound:
    value: float
    absolute_bound: float
    relative_bound: float | None
    derivative_bound: float
    interval: tuple[float, float]
    rigorous: bool
    description: str


def symbolic_derivative_bound(
    expression: sp.Expr, symbol: sp.Symbol, left: float, right: float
) -> float | None:
    try:
        derivative = sp.diff(expression, symbol)
        domain = sp.Interval(left, right)
        lower = sp.minimum(derivative, symbol, domain)
        upper = sp.maximum(derivative, symbol, domain)
        value = max(abs(float(sp.N(lower, 17))), abs(float(sp.N(upper, 17))))
    except (ValueError, TypeError, NotImplementedError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def propagate_bound(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    x_approx: float,
    error_bound_x: float,
    domain: tuple[float, float],
    *,
    derivative_bound: float | None = None,
    derivative_bound_verified: bool = False,
    samples: int = 2001,
) -> PropagatedBound:
    a, b = domain
    if not all(math.isfinite(v) for v in (a, b, x_approx, error_bound_x)):
        raise ValueError("Dữ liệu truyền sai số phải hữu hạn.")
    if not a <= x_approx <= b or error_bound_x < 0 or samples < 2:
        raise ValueError("Miền hoặc chặn sai số không hợp lệ.")
    left, right = max(a, x_approx - error_bound_x), min(b, x_approx + error_bound_x)
    value = float(function(x_approx))
    if not math.isfinite(value):
        raise ArithmeticError("G(xấp xỉ) không hữu hạn.")
    if derivative_bound is None:
        slopes = [
            abs(float(derivative(left + (right - left) * i / (samples - 1))))
            for i in range(samples)
        ]
        if not all(math.isfinite(item) for item in slopes):
            raise ArithmeticError("G' không hữu hạn trên khoảng sai số.")
        M = max(slopes, default=0.0)
        rigorous = False
        description = "M_G ước lượng bằng lưới; không thay thế chứng minh giải tích."
    else:
        M = float(derivative_bound)
        if not math.isfinite(M) or M < 0:
            raise ValueError("M_G phải không âm và hữu hạn.")
        rigorous = derivative_bound_verified
        description = (
            "M_G là chặn đã xác nhận trên toàn khoảng sai số."
            if rigorous
            else "M_G chưa được xác nhận; kết quả chỉ tham khảo."
        )
    absolute = M * error_bound_x
    relative = absolute / (abs(value) - absolute) if abs(value) > absolute else None
    return PropagatedBound(
        value, absolute, relative, M, (left, right), rigorous, description
    )
