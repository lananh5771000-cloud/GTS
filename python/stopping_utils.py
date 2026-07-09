"""Shared stopping-mode helpers for iterative numerical methods."""

from __future__ import annotations

from dataclasses import dataclass
import math


FIXED_ITERATIONS = "fixed_iterations"
ABSOLUTE_STEP = "absolute_step"
RELATIVE_STEP = "relative_step"
RESIDUAL = "residual"
STEP_AND_RESIDUAL = "step_and_residual"
THEORETICAL_BOUND = "theoretical_bound"
SIGNIFICANT_DIGITS = "significant_digits"
MAX_ITER = "max_iter"


STOPPING_MODE_LABELS: dict[str, str] = {
    FIXED_ITERATIONS: "Thực hiện đúng k bước",
    ABSOLUTE_STEP: "Dừng theo sai khác tuyệt đối",
    RELATIVE_STEP: "Dừng theo sai khác tương đối",
    RESIDUAL: "Dừng theo residual",
    STEP_AND_RESIDUAL: "Dừng khi đồng thời đạt sai khác và residual",
    THEORETICAL_BOUND: "Dừng theo chặn sai số lý thuyết",
    SIGNIFICANT_DIGITS: "Dừng theo số chữ số chắc",
    MAX_ITER: "Giới hạn số bước tối đa",
}


@dataclass(frozen=True)
class StopCheck:
    mode: str
    reached: bool
    reason: str
    value: float | None = None
    residual: float | None = None
    epsilon: float | None = None


def safe_relative(value: float, scale: float) -> float:
    """Return value / max(1, abs(scale)) without a near-zero denominator."""
    if not math.isfinite(value) or not math.isfinite(scale):
        return math.inf
    return abs(value) / max(1.0, abs(scale))


def scaled_tolerance(
    user_epsilon: float,
    dimension: int,
    scale: float = 1.0,
    *,
    multiplier: float = 10000.0,
) -> float:
    """Tolerance floor scaled by problem size, floating precision and data size."""
    if user_epsilon <= 0 or not math.isfinite(user_epsilon):
        raise ValueError("user_epsilon must be positive and finite")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    safe_scale = max(1.0, abs(float(scale))) if math.isfinite(scale) else 1.0
    return max(
        user_epsilon * safe_scale,
        multiplier * dimension * math.ulp(1.0) * safe_scale,
    )


def check_stop(
    mode: str,
    *,
    iteration: int,
    epsilon: float,
    fixed_iterations: int | None = None,
    step: float | None = None,
    current_norm: float | None = None,
    residual: float | None = None,
    residual_epsilon: float | None = None,
    theoretical_bound: float | None = None,
    significant_digits: int | None = None,
) -> StopCheck:
    """Evaluate one standard stopping criterion and return an explicit reason."""
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")

    if mode == FIXED_ITERATIONS:
        if fixed_iterations is None or fixed_iterations < 0:
            raise ValueError("fixed_iterations must be nonnegative")
        reached = iteration >= fixed_iterations
        return StopCheck(
            mode,
            reached,
            (
                f"Dừng sau đúng k={fixed_iterations} bước theo yêu cầu đề bài; "
                "chưa dùng tiêu chuẩn epsilon để chứng nhận."
            ),
            float(iteration),
            residual,
            epsilon,
        )

    if mode == ABSOLUTE_STEP:
        value = math.inf if step is None else abs(float(step))
        return StopCheck(
            mode,
            value <= epsilon,
            f"Hội tụ theo sai khác tuyệt đối với epsilon={epsilon:.3e}.",
            value,
            residual,
            epsilon,
        )

    if mode == RELATIVE_STEP:
        value = safe_relative(math.inf if step is None else float(step), current_norm or 0.0)
        return StopCheck(
            mode,
            value <= epsilon,
            f"Hội tụ theo sai khác tương đối với epsilon={epsilon:.3e}.",
            value,
            residual,
            epsilon,
        )

    if mode == RESIDUAL:
        value = math.inf if residual is None else abs(float(residual))
        tol = epsilon if residual_epsilon is None else residual_epsilon
        return StopCheck(
            mode,
            value <= tol,
            f"Hội tụ theo residual với epsilon={tol:.3e}.",
            value,
            value,
            tol,
        )

    if mode == STEP_AND_RESIDUAL:
        step_value = math.inf if step is None else abs(float(step))
        residual_value = math.inf if residual is None else abs(float(residual))
        tol = epsilon if residual_epsilon is None else residual_epsilon
        reached = step_value <= epsilon and residual_value <= tol
        return StopCheck(
            mode,
            reached,
            (
                f"Hội tụ khi đồng thời sai khác <= {epsilon:.3e} "
                f"và residual <= {tol:.3e}."
            ),
            step_value,
            residual_value,
            epsilon,
        )

    if mode == THEORETICAL_BOUND:
        value = math.inf if theoretical_bound is None else abs(float(theoretical_bound))
        return StopCheck(
            mode,
            value <= epsilon,
            f"Hội tụ theo chặn sai số lý thuyết với epsilon={epsilon:.3e}.",
            value,
            residual,
            epsilon,
        )

    if mode == SIGNIFICANT_DIGITS:
        if significant_digits is None or significant_digits < 0:
            raise ValueError("significant_digits must be nonnegative")
        target = 0.5 * 10.0 ** (-significant_digits)
        value = math.inf if step is None else abs(float(step))
        return StopCheck(
            mode,
            value <= target,
            f"Hội tụ theo {significant_digits} chữ số chắc.",
            value,
            residual,
            target,
        )

    if mode == MAX_ITER:
        return StopCheck(
            mode,
            True,
            f"Dừng tại max_iter={iteration}; chưa thỏa tiêu chuẩn hội tụ đã chọn.",
            float(iteration),
            residual,
            epsilon,
        )

    raise ValueError(f"Unsupported stopping mode: {mode}")


def print_stop_menu(modes: list[str]) -> None:
    for index, mode in enumerate(modes, start=1):
        print(f"{index}. {STOPPING_MODE_LABELS[mode]}")
