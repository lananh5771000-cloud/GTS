import contextlib
import importlib.util
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sympy as sp

import khai_trien_ky_di_svd as svd
import lapdon_pt
import hptlapdonphituyen as nonlinear_fixed
import gaussrank
import danilevski_tri_rieng as danilevsky


ROOT = Path(__file__).resolve().parents[1]


def load_gauss_two_module():
    path = ROOT / "gauss (2).py"
    spec = importlib.util.spec_from_file_location("gauss_two_pdf_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_svd_menu_cond_number_and_low_rank_approximation():
    A = np.array([[3.0, 0.0], [0.0, 1.0]])
    assert svd.condition_number_pdf(A) == 3.0

    approximation, kept, tail, result = svd.svd_low_rank_approximation(A, 0.4)
    assert kept == 1
    assert tail <= 0.4
    np.testing.assert_allclose(approximation, np.array([[3.0, 0.0], [0.0, 0.0]]), atol=1e-8)
    assert result.rank == 2

    inputs = iter([
        "2",
        "2",
        "3 0",
        "0 1",
        "7",
        "2",
        "1e-10",
        "50",
    ])
    output = io.StringIO()
    with patch("builtins.input", lambda prompt="": next(inputs)), contextlib.redirect_stdout(output):
        svd.main()
    text = output.getvalue()
    assert "TÍNH SỐ ĐIỀU KIỆN cond(A)" in text
    assert "cond(A) = sqrt(λmax/λmin)" in text


def test_scalar_fixed_point_priori_menu_runs():
    inputs = iter([
        "(x+1)/2",
        "n",
        "6",
        "n",
        "0",
        "2",
        "1",
        "1",
        "0.01",
        "n",
    ])
    output = io.StringIO()
    with patch("builtins.input", lambda prompt="": next(inputs)), contextlib.redirect_stdout(output):
        lapdon_pt.fixed_point_iteration()
    text = output.getvalue()
    assert "Sai số tiên nghiệm của x" in text
    assert "[Công thức tiên nghiệm]" in text


def test_nonlinear_fixed_point_priori_is_invoked():
    x1 = sp.symbols("x1")
    result = nonlinear_fixed.fixed_point_iteration_priori(
        [x1 / 2],
        (x1,),
        [1.0],
        [(0.0, 2.0)],
        0.5,
        "vo cung",
        0.01,
    )
    assert result["priori_steps"] >= 1
    assert result["steps"] == result["priori_steps"]


def test_gaussrank_idx_and_param_solution():
    A = sp.Matrix([[0, 1]])
    B = sp.Matrix([[1]])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = gaussrank.solve_from_matrices(A, B, 8)
    text = output.getvalue()
    assert "idx = [2]" in text
    assert "Hệ có vô số nghiệm" in text
    assert result == sp.Matrix([[0], [1]])


def test_gauss_jordan_pdf_pivot_order_and_idx_output():
    module = load_gauss_two_module()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        augmented, pivots, last_step = module.gauss_jordan(
            [[0.0, 1.0], [1.0, 1.0]], [[1.0], [2.0]], 8
        )
    text = output.getvalue()
    assert "theo quy tắc PDF" in text
    assert "idx = [1, 2]" in text
    assert last_step == 2
    assert pivots == {0: 0, 1: 1}
    np.testing.assert_allclose([row[-1] for row in augmented], [1.0, 1.0])


def test_danilevsky_frobenius_vector_note_and_convention():
    A = sp.Matrix([[2]])
    F = sp.Matrix([[2]])
    Q = sp.eye(1)
    root_data = [{"exact": True, "value": sp.Integer(2), "multiplicity": 1}]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        danilevsky.print_eigenvectors(A, F, Q, [(0, 1)], root_data, 7)
    text = output.getvalue()
    assert "quy ước trình bày khác" in text
    np.testing.assert_array_equal(
        np.array(danilevsky.canonical_frobenius_vector(2, 3), dtype=object).reshape(-1),
        np.array([4, 2, 1], dtype=object),
    )