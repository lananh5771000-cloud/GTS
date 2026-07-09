import contextlib
import importlib.util
import io
import math
import pathlib
import sys
import unittest
from fractions import Fraction

import numpy as np
import sympy as sp

from chiadoi_pt import bisection
from daycung import chord_method
from hptlapdonphituyen import fixed_point_iteration
from khai_trien_ky_di_svd import svd_power
from lapdon_pt import fixed_point
from phantach_lu import multiply_matrices, plu_decomposition, plu_solve
from tieptuyen import safeguarded_newton
from tri_rieng_troi_xuong_thang import (
    dominant_eigenpair,
    power_method,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestBisection(unittest.TestCase):
    def test_endpoint_a_and_b(self):
        self.assertEqual(bisection(lambda x: x, 0, 2, 1e-6).root, 0)
        self.assertEqual(bisection(lambda x: x - 2, 0, 2, 1e-6).root, 2)

    def test_sign_change_and_step_formula(self):
        result = bisection(lambda x: x * x - 2, 0, 2, 1e-8, continuity_verified=True)
        self.assertTrue(result.converged)
        self.assertTrue(result.certified)
        self.assertLessEqual(result.error_bound, 1e-8)
        self.assertEqual(result.required_steps, math.ceil(math.log2(2 / (2e-8))))
        self.assertAlmostEqual(result.root, math.sqrt(2), places=7)

    def test_bad_interval_and_epsilon(self):
        with self.assertRaises(ValueError):
            bisection(lambda x: x * x + 1, -1, 1, 1e-6)
        with self.assertRaises(ValueError):
            bisection(lambda x: x, -1, 1, 0)

    def test_nonfinite_midpoint_is_rejected(self):
        def f(x):
            return math.nan if x == 0 else x
        with self.assertRaises(ArithmeticError):
            bisection(f, -1, 1, 1e-6, function_tolerance=0)


class TestFixedPoint(unittest.TestCase):
    def test_relative_bound_counterexample(self):
        result = fixed_point(
            lambda x: 0.5 * x + 0.5,
            lambda _x: 0.5,
            0,
            2,
            2,
            0.2,
            relative=True,
            analytic_conditions_verified=True,
        )
        self.assertTrue(result.certified)
        self.assertGreaterEqual(result.iterations[-1][0], 3)
        self.assertNotEqual(result.root, 1.25)
        self.assertLessEqual(result.relative_error_bound, 0.2)

    def test_mapping_and_contraction_are_both_required(self):
        result = fixed_point(lambda x: 2 * x, lambda _x: 2, -1, 1, 0.2, 1e-6, max_iter=8)
        self.assertFalse(result.certified)

    def test_leaving_domain_stops(self):
        result = fixed_point(lambda _x: 2.0, lambda _x: 0.0, 0, 1, 0.5, 1e-6)
        self.assertFalse(result.converged)
        self.assertIn("rời", result.reason)

    def test_nonlinear_system_relative_bound(self):
        x = sp.Symbol("x")
        result = fixed_point_iteration(
            [sp.Rational(1, 2) * x + sp.Rational(1, 2)],
            [x],
            [2.0],
            [(0.0, 2.0)],
            0.5,
            "vo cung",
            F=[x - 1],
            epsilon=0.2,
            relative=True,
            max_iter=20,
            banach_proven=True,
            F_lipschitz=1.0,
            F_scale=1.0,
        )
        self.assertEqual(result["status"], "certified")
        self.assertGreaterEqual(result["steps"], 3)


class TestNewtonAndChord(unittest.TestCase):
    def test_newton_converges(self):
        result = safeguarded_newton(
            lambda x: x * x - 2,
            lambda x: 2 * x,
            1,
            2,
            1e-10,
            x0=2,
            assumptions_verified=True,
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.root, math.sqrt(2), places=9)

    def test_zero_derivative_uses_guard(self):
        result = safeguarded_newton(
            lambda x: x**3 - 1,
            lambda x: 3 * x * x,
            0,
            2,
            1e-10,
            x0=0,
            assumptions_verified=True,
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.root, 1.0)
        self.assertIn("chia đôi bảo vệ", {row[4] for row in result.iterations})

    def test_newton_max_iter(self):
        result = safeguarded_newton(lambda x: x * x - 2, lambda x: 2 * x, 1, 2, 1e-30, max_iter=1)
        self.assertFalse(result.converged)

    def test_regula_falsi_and_fixed_endpoint_are_explicit(self):
        regula = chord_method(
            lambda x: x * x - 2, 0, 2, 1e-8,
            variant="regula_falsi", m1=1, assumptions_verified=True,
        )
        fixed = chord_method(
            lambda x: x * x - 2, 0, 2, 1e-8,
            variant="fixed_endpoint", second_derivative=lambda _x: 2,
            m1=1, assumptions_verified=True,
        )
        self.assertTrue(regula.converged and fixed.converged)
        self.assertEqual(regula.variant, "regula_falsi")
        self.assertEqual(fixed.variant, "fixed_endpoint")

    def test_chord_endpoint_and_bad_bracket(self):
        self.assertEqual(chord_method(lambda x: x, 0, 1, 1e-6).root, 0)
        with self.assertRaises(ValueError):
            chord_method(lambda x: x * x + 1, -1, 1, 1e-6)


class TestPLU(unittest.TestCase):
    def test_no_swap(self):
        A = [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(2)]]
        P, L, U, swaps, _ = plu_decomposition(A)
        self.assertEqual(swaps, [])
        self.assertEqual(multiply_matrices(P, A), multiply_matrices(L, U))

    def test_required_swap_and_solution(self):
        A = [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(1)]]
        B = [[Fraction(1)], [Fraction(2)]]
        X, P, L, U, swaps, _ = plu_solve(A, B)
        self.assertTrue(swaps)
        self.assertEqual(X, [[Fraction(1)], [Fraction(1)]])
        self.assertEqual(multiply_matrices(P, A), multiply_matrices(L, U))

    def test_singular(self):
        A = [[Fraction(1), Fraction(2)], [Fraction(2), Fraction(4)]]
        with self.assertRaises(ArithmeticError):
            plu_decomposition(A)

    def test_near_singular_warning(self):
        A = [[1.0, 0.0], [0.0, 1e-15]]
        _P, _L, _U, _swaps, near = plu_decomposition(A, pivot_tolerance=1e-12)
        self.assertTrue(near)


class TestEigenvalues(unittest.TestCase):
    def test_bad_initial_vector_does_not_hide_dominant(self):
        result = dominant_eigenpair([[5.0, 0.0], [0.0, 2.0]], [0.0, 1.0])
        self.assertAlmostEqual(result.result.eigenvalue, 5.0)
        self.assertFalse(result.dominant_certified)

    def test_negative_largest_modulus(self):
        result = dominant_eigenpair([[-5.0, 0.0], [0.0, 2.0]], [1.0, 1.0])
        self.assertAlmostEqual(result.result.eigenvalue, -5.0)
        self.assertLess(result.result.relative_residual, 1e-10)

    def test_fixed_steps_are_scale_invariant(self):
        A = [[5.0, 0.0], [0.0, 2.0]]
        small = power_method(A, [1.0, 1.0], 5, 1e-8, 100)
        large = power_method(A, [100.0, 100.0], 5, 1e-8, 100)
        self.assertAlmostEqual(small.eigenvalue, large.eigenvalue, places=13)

    def test_repeated_eigenvalue_residual(self):
        result = dominant_eigenpair([[3.0, 0.0], [0.0, 3.0]], [1.0, 0.0])
        self.assertLess(result.result.residual, 1e-12)


class TestSVD(unittest.TestCase):
    def assert_svd(self, A, initial=None, expected_rank=None):
        result = svd_power(A, initial=initial, epsilon=1e-10, full_matrices=True)
        self.assertLess(result.relative_reconstruction_error, 1e-8)
        self.assertLess(result.left_orthogonality_error, 1e-8)
        self.assertLess(result.right_orthogonality_error, 1e-8)
        self.assertTrue(np.all(result.singular_values[:-1] >= result.singular_values[1:]))
        if expected_rank is not None:
            self.assertEqual(result.rank, expected_rank)
        return result

    def test_zero_matrix(self):
        self.assert_svd(np.zeros((2, 3)), expected_rank=0)

    def test_rectangular_and_repeated_values(self):
        result = self.assert_svd(np.array([[1.0, 0, 0], [0, 1.0, 0]]), expected_rank=2)
        np.testing.assert_allclose(result.singular_values, [1, 1], atol=1e-9)

    def test_bad_start_does_not_hide_sigma_max(self):
        result = self.assert_svd(np.diag([3.0, 2.0]), initial=np.array([0.0, 1.0]), expected_rank=2)
        np.testing.assert_allclose(result.singular_values, [3, 2], atol=1e-8)

    def test_rank_deficient(self):
        self.assert_svd(np.ones((2, 2)), expected_rank=1)
        self.assert_svd(np.array([[1.0, 2, 3], [2, 4, 6.0]]), expected_rank=1)

    def test_random_rectangular_rank_deficient(self):
        rng = np.random.default_rng(20260708)
        left = rng.normal(size=(5, 2))
        right = rng.normal(size=(2, 4))
        self.assert_svd(left @ right, expected_rank=2)

    def test_invalid_data(self):
        with self.assertRaises(ValueError):
            svd_power(np.array([[math.nan]]))
        with self.assertRaises(ValueError):
            svd_power(np.ones((2, 2)), max_iter=0)


class TestPreviouslyStrongFiles(unittest.TestCase):
    def test_jacobi_multiple_rhs_and_reorder(self):
        import lapjacobituyentinh as jacobi
        A = np.array([[1.0, 3.0], [4.0, 1.0]])
        B = np.array([[4.0, 7.0], [5.0, 6.0]])
        result = jacobi.jacobi_solve(A, B, np.zeros_like(B), epsilon=1e-9, max_iter=1000)
        self.assertEqual(result["status"], "converged")
        np.testing.assert_allclose(A @ result["X"], B, atol=1e-7)

    def test_seidel_known_solution(self):
        import seideltuyentinh as seidel
        A = np.array([[4.0, 1.0], [2.0, 3.0]])
        b = np.array([1.0, 2.0])
        x, info = seidel.gauss_seidel(A, b, show=False, epsilon=1e-9, max_iter=1000)
        self.assertTrue(info["converged"])
        np.testing.assert_allclose(A @ x, b, atol=1e-7)

    def test_bordering_inverse_order_one_and_permutation(self):
        import nghichdao_vienquanh as bordering
        cases = [
            [[Fraction(2)]],
            [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(1)]],
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            inverses = [bordering.find_inverse_by_bordering(A, 6) for A in cases]
        for A, inverse in zip(cases, inverses):
            self.assertEqual(bordering.multiply_matrices(A, inverse), bordering.identity_matrix(len(A)))
        with contextlib.redirect_stdout(io.StringIO()):
            singular = bordering.find_inverse_by_bordering(
                [[Fraction(1), Fraction(2)], [Fraction(2), Fraction(4)]], 6
            )
        self.assertIsNone(singular)

    def test_danilevsky_characteristic_polynomial(self):
        import danilevski_tri_rieng as danilevsky
        matrices = [
            sp.diag(1, 2),
            sp.Matrix([[0, 1], [-2, 3]]),
            sp.Matrix([[0, -1], [1, 0]]),
        ]
        for A in matrices:
            F, _Q, _blocks = danilevsky.danilevsky_transform(A, show_steps=False)
            self.assertEqual(sp.expand(A.charpoly().as_expr()), sp.expand(F.charpoly().as_expr()))

    def test_cholesky_spd_singular_and_order_one(self):
        import cholesky
        cases = [
            ([[Fraction(4), Fraction(2)], [Fraction(2), Fraction(3)]], True),
            ([[Fraction(1), Fraction(1)], [Fraction(1), Fraction(1)]], False),
            ([[Fraction(1)]], True),
        ]
        for A, success in cases:
            with contextlib.redirect_stdout(io.StringIO()):
                U, reason = cholesky.cholesky_decomposition(A, 6, show_steps=False)
            self.assertEqual(U is not None, success)
            if success:
                self.assertEqual(cholesky.multiply_matrices(cholesky.transpose_matrix(U), U), A)
            else:
                self.assertEqual(reason, "not_positive_definite")

    def test_newton_schulz_inverse_and_residual(self):
        import nghichdao_newton as inverse_newton
        A = [[0.0, 1.0], [1.0, 1.0]]
        X0 = inverse_newton.automatic_initial_approximation(A)[0]
        with contextlib.redirect_stdout(io.StringIO()):
            result = inverse_newton.newton_inverse(A, X0, 1e-8, 100, 7)
        self.assertTrue(result["converged"])
        np.testing.assert_allclose(np.array(A) @ np.array(result["matrix"]), np.eye(2), atol=1e-8)

    def test_gauss_jordan_swap_and_rank(self):
        path = ROOT / "gauss (2).py"
        spec = importlib.util.spec_from_file_location("gauss_two_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["gauss_two_test"] = module
        spec.loader.exec_module(module)
        with contextlib.redirect_stdout(io.StringIO()):
            augmented, pivots, rank = module.gauss_jordan(
                [[0.0, 1.0], [1.0, 1.0]], [[1.0], [2.0]], 8
            )
        self.assertEqual(rank, 2)
        self.assertEqual(pivots, {0: 0, 1: 1})
        np.testing.assert_allclose([row[-1] for row in augmented], [1, 1])

    def test_gauss_rank_exact_solution(self):
        import gaussrank
        A = sp.Matrix([[0, 1], [1, 1]])
        B = sp.Matrix([[1], [2]])
        with contextlib.redirect_stdout(io.StringIO()):
            result = gaussrank.solve_from_matrices(A, B, 8)
        self.assertEqual(result, sp.Matrix([[1], [1]]))

    def test_polynomial_repeated_and_complex_roots(self):
        import phuong_phap_da_thuc as polynomial
        self.assertEqual(polynomial.sturm_real_root_count([1, -2, 1]), 1)
        self.assertEqual(polynomial.all_distinct_real_roots([1, -2, 1], 1e-9), [1.0])
        self.assertEqual(polynomial.sturm_real_root_count([1, 0, 1]), 0)
        factors = polynomial.square_free_decomposition([1, -2, 1])
        self.assertEqual(factors[0][1], 2)

    def test_error_propagation_labels_and_zero_value(self):
        path = ROOT / "saiso (1).py"
        spec = importlib.util.spec_from_file_location("saiso_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["saiso_test"] = module
        spec.loader.exec_module(module)
        approximation = module.propagate_first_order(0.0, [2.0], [0.1])
        rigorous = module.propagate_mean_value_bound(10.0, [3.0], [0.1])
        self.assertFalse(approximation.rigorous)
        self.assertIsNone(approximation.relative_error)
        self.assertTrue(rigorous.rigorous)
        self.assertAlmostEqual(rigorous.absolute_error, 0.3)
        with self.assertRaises(ValueError):
            module.propagate_first_order(1.0, [1.0], [-0.1])

    def test_every_script_imports(self):
        for path in ROOT.glob("*.py"):
            name = "safety_import_" + path.stem.replace(" ", "_").replace("(", "").replace(")", "")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            with contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(module)


if __name__ == "__main__":
    unittest.main()
