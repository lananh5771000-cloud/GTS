import numpy as np

import chiadoi_pt
import daycung
import khai_trien_ky_di_svd as svd
import lapdon_pt
import lapdon_tuyentinh
import lapjacobituyentinh
import seideltuyentinh
import tieptuyen
from stopping_utils import (
    RELATIVE_STEP,
    STEP_AND_RESIDUAL,
    check_stop,
    safe_relative,
)


def _assert_final_svd_ok(A, **kwargs):
    result = svd.svd_power(A, epsilon=kwargs.pop("epsilon", 1e-12), max_iter=10000, **kwargs)
    assert result.final_accepted
    assert result.converged
    assert result.singular_triplet_residual_ok
    assert result.orthogonality_ok
    assert result.reconstruction_ok
    assert result.relative_reconstruction_error <= result.final_relative_tolerance
    assert result.reconstruction_error <= result.final_absolute_tolerance
    assert all(len(power.history) < 10000 for power in (result.power_results or []))
    return result


def test_svd_rectangular_2x4_final_accepts_scaled_checks():
    A = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    result = _assert_final_svd_ok(A, full_matrices=False)
    assert result.rank == 2
    assert result.relative_reconstruction_error < 1e-14
    assert result.final_relative_tolerance >= 1e-11


def test_svd_required_edge_matrices():
    rng = np.random.default_rng(20260709)
    cases = [
        np.array([[3.0, 3.0, 7.0], [3.0, 9.0, 10.0], [8.0, 4.0, 6.0], [2.0, 10.0, 9.0]]),
        rng.normal(size=(5, 2)),
        np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]),
        np.diag([1.0, 1.0 + 1e-10]),
        1e150 * np.array([[1.0, 2.0], [3.0, 4.0]]),
        1e-100 * np.array([[1.0, 2.0], [3.0, 4.0]]),
    ]
    for A in cases:
        _assert_final_svd_ok(A, full_matrices=False)

    zero = svd.svd_power(np.zeros((3, 2)), epsilon=1e-12, full_matrices=False)
    assert zero.final_accepted
    assert zero.rank == 0
    assert zero.reconstruction_error == 0.0


def test_fixed_iterations_for_scalar_algorithms_do_not_stop_on_small_error():
    def f(x):
        return x * x - 2.0

    bisection = chiadoi_pt.bisection(
        f, 1.0, 2.0, 1e-30, function_tolerance=0.0, fixed_iterations=5
    )
    assert len(bisection.iterations) == 5
    assert bisection.bracket == (1.40625, 1.4375)
    assert "k=5" in bisection.reason

    chord = daycung.chord_method(
        f, 1.0, 2.0, 1e-30, m1=2.0, fixed_iterations=5
    )
    assert len(chord.iterations) == 5
    assert "k=5" in chord.reason

    newton = tieptuyen.safeguarded_newton(
        f, lambda x: 2.0 * x, 1.0, 2.0, 1e-30, x0=2.0, fixed_iterations=4
    )
    assert len(newton.iterations) == 5
    assert "k=4" in newton.reason

    fixed = lapdon_pt.fixed_point(
        lambda x: 0.5 * x + 0.5,
        lambda _x: 0.5,
        0.0,
        2.0,
        2.0,
        1e-30,
        fixed_iterations=4,
    )
    assert fixed.status == "fixed_steps"
    assert len(fixed.iterations) == 5


def test_direct_linear_fixed_iterations_for_B_d_forms():
    B = np.array([[0.0, 0.25], [0.5, 0.0]])
    d = np.array([1.0, -1.0])
    x0 = np.zeros(2)
    A = np.eye(2) - B

    richardson = lapdon_tuyentinh.simple_iteration(
        A, d, B, d, x0, "inf", 1e-12, 20, 3
    )
    assert len(richardson.records) == 3

    jacobi = lapjacobituyentinh.jacobi_fixed_point(
        B, d, x0, stop_mode="fixed", fixed_steps=3, norm_kind="inf"
    )
    assert jacobi["status"] == "fixed_steps"
    assert len(jacobi["history"]) == 4

    seidel_x, seidel_info = seideltuyentinh.seidel_fixed_point(
        B, d, x0=x0, fixed_iterations=3, show=False
    )
    assert seidel_info["iterations"] == 3
    assert seidel_info["fixed_iterations_completed"]
    assert np.all(np.isfinite(seidel_x))


def test_standard_relative_and_step_residual_checks():
    assert safe_relative(1e-9, 0.0) == 1e-9
    rel = check_stop(RELATIVE_STEP, iteration=1, epsilon=1e-8, step=1e-9, current_norm=0.0)
    assert rel.reached

    step_only = check_stop(
        STEP_AND_RESIDUAL,
        iteration=2,
        epsilon=1e-6,
        step=1e-7,
        residual=1e-3,
    )
    assert not step_only.reached

    both = check_stop(
        STEP_AND_RESIDUAL,
        iteration=3,
        epsilon=1e-6,
        step=1e-7,
        residual=1e-7,
    )
    assert both.reached
