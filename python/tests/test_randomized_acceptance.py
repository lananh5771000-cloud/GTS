"""1.550 ca ngẫu nhiên tái lập được theo đúng bảng nghiệm thu của project."""

from fractions import Fraction
import contextlib

import numpy as np
import sympy as sp

import cholesky
import danilevski_tri_rieng as danilevsky
import khai_trien_ky_di_svd as svd
import lapjacobituyentinh as jacobi
import nghichdao_newton as newton_inverse
import nghichdao_vienquanh as bordering
import phantach_lu as lu
import phuong_phap_da_thuc as polynomial
import seideltuyentinh as seidel
import tri_rieng_troi_xuong_thang as eigen
from newton_he_phi_tuyen import newton_system


SEED = 20260708


class _Discard:
    def write(self, text):
        return len(text)

    def flush(self):
        return None


DISCARD = _Discard()


def _fraction_matrix(array):
    return [[Fraction(int(value)) for value in row] for row in np.asarray(array)]


def test_random_200_plu():
    rng = np.random.default_rng(SEED + 1)
    for case in range(200):
        n = 1 + case % 4
        A = rng.integers(-3, 4, size=(n, n))
        A += np.diag(np.sum(np.abs(A), axis=1) + 1)
        B = rng.integers(-5, 6, size=(n, 1 + case % 2))
        A_exact, B_exact = _fraction_matrix(A), _fraction_matrix(B)
        X, *_ = lu.plu_solve(A_exact, B_exact, show_steps=False)
        assert lu.multiply_matrices(A_exact, X) == B_exact


def test_random_100_cholesky():
    rng = np.random.default_rng(SEED + 2)
    for case in range(100):
        n = 1 + case % 4
        L = np.tril(rng.integers(-2, 3, size=(n, n)))
        np.fill_diagonal(L, rng.integers(1, 5, size=n))
        A = _fraction_matrix(L @ L.T)
        with contextlib.redirect_stdout(DISCARD):
            U, reason = cholesky.cholesky_decomposition(A, 8, show_steps=False)
        assert reason is None
        assert cholesky.multiply_matrices(cholesky.transpose_matrix(U), U) == A


def test_random_150_bordering():
    rng = np.random.default_rng(SEED + 3)
    with contextlib.redirect_stdout(DISCARD):
        for case in range(150):
            n = 1 + case % 4
            L = np.tril(rng.integers(-2, 3, size=(n, n)))
            np.fill_diagonal(L, rng.integers(1, 4, size=n))
            A = _fraction_matrix(L @ L.T)
            inverse, _determinant, error = bordering.bordering_inverse_recursive(A)
            assert error is None
            assert bordering.multiply_matrices(A, inverse) == bordering.identity_matrix(n)


def test_random_100_newton_schulz():
    rng = np.random.default_rng(SEED + 4)
    with contextlib.redirect_stdout(DISCARD):
        for case in range(100):
            n = 1 + case % 3
            diagonal = rng.uniform(0.5, 3.0, size=n)
            A = np.diag(diagonal).tolist()
            X0, *_ = newton_inverse.automatic_initial_approximation(A)
            result = newton_inverse.newton_inverse(A, X0, 1e-8, 30, 10)
            assert result is not None and result["converged"]
            X = np.asarray(result["raw_matrix"], dtype=float)
            assert np.linalg.norm(np.asarray(A) @ X - np.eye(n), np.inf) < 1e-7


def test_random_150_jacobi():
    rng = np.random.default_rng(SEED + 5)
    for case in range(150):
        n = 2 + case % 4
        off = rng.uniform(-0.2, 0.2, size=(n, n))
        np.fill_diagonal(off, 0.0)
        A = off + np.diag(np.sum(np.abs(off), axis=1) + rng.uniform(1.0, 2.0, size=n))
        B = rng.normal(size=(n, 1 + case % 2))
        result = jacobi.jacobi_solve(A, B, np.zeros_like(B), epsilon=1e-9, max_iter=500)
        assert result["status"] in {"converged", "converged_exact"}
        assert np.linalg.norm(A @ result["X"] - B, np.inf) < 1e-7


def test_random_150_seidel_Ax_b():
    rng = np.random.default_rng(SEED + 6)
    for case in range(150):
        n = 2 + case % 4
        off = rng.uniform(-0.2, 0.2, size=(n, n))
        np.fill_diagonal(off, 0.0)
        A = off + np.diag(np.sum(np.abs(off), axis=1) + rng.uniform(1.0, 2.0, size=n))
        b = rng.normal(size=n)
        x, info = seidel.gauss_seidel(A, b, epsilon=1e-9, max_iter=500, show=False)
        assert info["converged"]
        assert np.linalg.norm(A @ x - b, np.inf) < 1e-7


def test_random_150_seidel_direct_x_Bx_d():
    rng = np.random.default_rng(SEED + 7)
    for case in range(150):
        n = 2 + case % 4
        B = rng.uniform(-1.0, 1.0, size=(n, n))
        B *= 0.35 / max(np.linalg.norm(B, np.inf), np.finfo(float).tiny)
        d = rng.normal(size=n)
        x, info = seidel.seidel_fixed_point(B, d, epsilon=1e-10, max_iter=500, show=False)
        assert info["converged"]
        assert np.linalg.norm(x - B @ x - d, np.inf) < 1e-8


def test_random_100_danilevsky():
    rng = np.random.default_rng(SEED + 8)
    for case in range(100):
        n = 2 + case % 2
        A = sp.Matrix(rng.integers(-4, 5, size=(n, n)).tolist())
        F, _Q, _blocks = danilevsky.danilevsky_transform(A, show_steps=False)
        assert sp.expand(A.charpoly().as_expr()) == sp.expand(F.charpoly().as_expr())


def test_random_100_eigenpairs():
    rng = np.random.default_rng(SEED + 9)
    for _case in range(100):
        angle = rng.uniform(-np.pi, np.pi)
        Q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        values = np.sort(rng.uniform(0.5, 5.0, size=2))[::-1]
        if values[0] - values[1] < 0.2:
            values[0] = values[1] + 0.2
        A = Q @ np.diag(values) @ Q.T
        result = eigen.eigenpairs_with_deflation(A.tolist(), 2, epsilon=1e-9, max_iter=2000)
        assert result.success
        assert max(result.relative_residuals) < 1e-8


def test_random_150_svd_sizes():
    rng = np.random.default_rng(SEED + 10)
    for case in range(150):
        m, n = 1 + case % 4, 1 + (case // 4) % 4
        p = min(m, n)
        diagonal = np.sort(rng.uniform(0.5, 5.0, size=p))[::-1]
        A = np.zeros((m, n))
        A[np.arange(p), np.arange(p)] = diagonal
        result = svd.svd_power(A, epsilon=1e-10, max_iter=2000, full_matrices=False)
        assert result.converged
        np.testing.assert_allclose(result.singular_values, diagonal, rtol=1e-9, atol=1e-10)
        assert result.relative_reconstruction_error < 1e-9


def test_random_100_polynomials():
    rng = np.random.default_rng(SEED + 11)
    for case in range(100):
        left = int(rng.integers(-5, 1))
        right = int(rng.integers(1, 6))
        m1, m2 = 1 + case % 3, 1 + (case // 3) % 3
        coefficients = polynomial.polynomial_multiply(
            polynomial.polynomial_power([1, -left], m1),
            polynomial.polynomial_power([1, -right], m2),
        )
        factors = polynomial.square_free_decomposition(coefficients)
        multiplicity_left = next(
            multiplicity
            for factor, multiplicity in factors
            if polynomial.horner_value_exact(factor, left) == 0
        )
        multiplicity_right = next(
            multiplicity
            for factor, multiplicity in factors
            if polynomial.horner_value_exact(factor, right) == 0
        )
        assert (multiplicity_left, multiplicity_right) == (m1, m2)
        assert polynomial.sturm_real_root_count(coefficients) == 2


def test_random_100_nonlinear_equations():
    rng = np.random.default_rng(SEED + 12)
    x = sp.Symbol("x")
    for _case in range(100):
        root = float(rng.uniform(0.5, 5.0))
        result = newton_system(
            [x**2 - root**2], [x], np.array([max(0.25, root * 0.7)]), 1e-10, 50
        )
        assert result.converged
        assert abs(result.x[0] - root) < 1e-8
        assert result.residual_norm < 1e-9
