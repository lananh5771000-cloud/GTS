import contextlib
import io

import numpy as np
import pytest
import sympy as sp

import danilevski_tri_rieng as danilevsky
import tri_rieng_troi_xuong_thang as eigen


def test_power_positive_dominant():
    result = eigen.dominant_eigenpair([[2.0, 1.0], [1.0, 2.0]], [1.0, 1.0])
    assert result.result.eigenvalue == pytest.approx(3.0, abs=1e-10)
    assert result.result.relative_residual < 1e-10


def test_power_negative_dominant_allows_sign_flip():
    result = eigen.dominant_eigenpair([[-3.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
    assert result.result.eigenvalue == pytest.approx(-3.0, abs=1e-10)
    assert result.result.relative_residual < 1e-10
    assert "phan ky" not in result.warning.lower()


def test_equal_modulus_warns_not_unique():
    result = eigen.dominant_eigenpair([[2.0, 0.0], [0.0, -2.0]], [1.0, 1.0])
    assert not result.dominant_certified
    assert "du dieu kien tach tri rieng troi duy nhat theo modun" in result.warning


def test_bad_initial_vector_is_reported_and_retried():
    result = eigen.dominant_eigenpair([[5.0, 0.0], [0.0, 2.0]], [0.0, 1.0])
    assert result.result.eigenvalue == pytest.approx(5.0, abs=1e-10)
    assert "Vector dau ban dau khong phu hop" in result.warning


def test_nonsymmetric_deflation_uses_left_and_right_vectors():
    A = [[2.0, 1.0], [0.0, 1.0]]
    search = eigen.dominant_eigenpair(A, [1.0, 1.0])
    result = eigen.deflate(A, search.result.eigenvalue, search.result.eigenvector)
    assert "w^T*v" in result.method
    assert result.left_vector is not None
    assert result.left_right_dot is not None
    assert abs(result.left_right_dot) > 1e-12
    symmetric_wrong = np.array(A) - search.result.eigenvalue * np.outer(
        search.result.eigenvector, search.result.eigenvector
    )
    assert not np.allclose(np.array(result.matrix), symmetric_wrong)


def test_symmetric_multiple_deflation_diagonal():
    result = eigen.eigenpairs_with_deflation(
        [[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
        3,
        [1.0, 1.0, 1.0],
    )
    assert result.success
    assert sorted(result.eigenvalues) == pytest.approx([1.0, 3.0, 5.0], abs=1e-8)
    assert all(value < 1e-8 for value in result.relative_residuals)


def test_complex_dominant_rejected_clearly():
    with pytest.raises(ArithmeticError, match="cap tri rieng phuc troi"):
        eigen.dominant_eigenpair([[0.0, -1.0], [1.0, 0.0]], [1.0, 0.0], max_iter=30)


def test_fixed_iterations_do_not_stop_early():
    result = eigen.power_method([[2.0, 1.0], [1.0, 2.0]], [1.0, 0.0], 4, 1e-1, 100)
    assert len(result.iterations) == 4
    assert "k_max" not in result.stop_reason


def test_repeated_eigenvalue_warns_not_unique():
    result = eigen.dominant_eigenpair([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    assert result.result.residual < 1e-10
    assert not result.dominant_certified
    assert "du dieu kien tach tri rieng troi duy nhat theo modun" in result.warning


def test_danilevsky_3x3_characteristic_matches_sympy():
    A = sp.Matrix([[4, 1, 0], [0, 3, 1], [2, 0, 1]])
    F, _Q, blocks = danilevsky.danilevsky_transform(A, show_steps=False)
    variable = sp.Symbol("lambda")
    _parts, polynomial = danilevsky.characteristic_polynomial_from_blocks(F, blocks, variable)
    assert sp.expand(polynomial) == sp.expand(A.charpoly(variable).as_expr())


def test_danilevsky_zero_pivot_swaps_or_warns():
    A = sp.Matrix([[1, 2, 3], [4, 5, 6], [1, 0, 0]])
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        F, _Q, blocks = danilevsky.danilevsky_transform(A, show_steps=True)
    text = stream.getvalue().lower()
    assert blocks
    assert "pivot" in text and ("ho" in text or "tach" in text)
    assert sp.expand(F.charpoly().as_expr()) == sp.expand(A.charpoly().as_expr())


def test_power_exam_report_has_clean_symbols():
    result = eigen.dominant_eigenpair([[2.0, 1.0], [1.0, 2.0]], [1.0, 1.0]).result
    report = eigen.render_power_exam_report(result, stage=1)
    forbidden = [
        "lambda_1",
        "v_i",
        "x_k",
        "final_accepted",
        "residual_ok",
        "np.float64",
        "debug",
    ]
    for token in forbidden:
        assert token not in report
    assert "λ₁" in report
    assert "v₁" in report
    assert "‖Av₁ − λ₁v₁‖₂" in report
    assert "Kết luận" in report
