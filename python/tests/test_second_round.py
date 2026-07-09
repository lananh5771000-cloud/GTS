import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import sympy as sp

import hptlapdonphituyen as nonlinear_fixed
import khai_trien_ky_di_svd as svd_module
import lapdon_pt as scalar_fixed
import lapdon_tuyentinh as linear_fixed
import newton_he_phi_tuyen as system_newton
import tri_rieng_troi_xuong_thang as eigen_module
from truyen_sai_so import propagate_bound, symbolic_derivative_bound


class TestScaledSVD(unittest.TestCase):
    def check_diagonal(self, diagonal):
        A = np.diag(diagonal)
        result = svd_module.svd_power(A, epsilon=1e-11, full_matrices=True)
        reference = np.linalg.svd(A, compute_uv=False)
        np.testing.assert_allclose(result.singular_values, reference, rtol=1e-12, atol=0.0)
        self.assertEqual(result.rank, 2)
        self.assertLess(result.relative_reconstruction_error, 1e-12)
        self.assertTrue(result.converged)
        self.assertTrue(all(max(pair) <= result.iteration_tolerance for pair in result.relation_residuals))
        return result

    def test_very_small_scale(self):
        self.check_diagonal([1e-100, 2e-100])

    def test_resolvable_condition_ratio(self):
        self.check_diagonal([1e12, 1.0])

    def test_very_large_scale(self):
        self.check_diagonal([1e150, 2e150])

    def test_not_converged_when_required_triplets_fail(self):
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        result = svd_module.svd_power(A, epsilon=1e-16, max_iter=1, full_matrices=True)
        self.assertFalse(result.converged)


class TestEigenpairCertification(unittest.TestCase):
    def test_real_power_rejects_rotation(self):
        A = [[0.0, -1.0], [1.0, 0.0]]
        with self.assertRaises(ArithmeticError):
            eigen_module.dominant_eigenpair(A, [1.0, 0.0], epsilon=1e-10, max_iter=30)
        collection = eigen_module.eigenpairs_with_deflation(A, 1, [1.0, 0.0], max_iter=30)
        self.assertFalse(collection.success)
        self.assertEqual(collection.eigenvalues, [])

    def test_nonsymmetric_vectors_are_checked_on_original(self):
        A = [[4.0, 1.0], [2.0, 3.0]]
        result = eigen_module.eigenpairs_with_deflation(A, 2, [1.0, 1.0], epsilon=1e-10)
        self.assertTrue(result.success)
        np.testing.assert_allclose(sorted(result.eigenvalues), [2.0, 5.0], atol=1e-9)
        self.assertTrue(all(value <= 1e-10 for value in result.relative_residuals))
        for value, vector in zip(result.eigenvalues, result.eigenvectors):
            residual = np.array(A) @ np.array(vector) - value * np.array(vector)
            self.assertLess(np.linalg.norm(residual), 1e-9)

    def test_repeated_identity_near_repeated_and_zero_start(self):
        for A in (
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0 + 1e-8]],
        ):
            result = eigen_module.eigenpairs_with_deflation(A, 2, [0.0, 0.0], epsilon=1e-10)
            self.assertTrue(result.success)
            self.assertTrue(all(value <= 1e-10 for value in result.relative_residuals))


class TestScalarFixedPointStatesAndG(unittest.TestCase):
    def test_noncontraction_leaves_domain_without_convergence(self):
        result = scalar_fixed.fixed_point(
            lambda x: 2.0 * x,
            lambda _x: 2.0,
            -1.0,
            1.0,
            0.2,
            0.5,
            max_iter=10,
        )
        self.assertFalse(result.converged)
        self.assertFalse(result.certified)
        self.assertTrue(result.left_domain)
        self.assertTrue(result.invalid_contraction)
        self.assertEqual(result.status, "left_domain")
        self.assertIsNone(result.error_bound)

    def test_derivative_at_point_is_not_global_bound(self):
        sampled = propagate_bound(
            lambda x: x * x,
            lambda x: 2.0 * x,
            0.0,
            0.1,
            (-1.0, 1.0),
        )
        self.assertGreater(sampled.absolute_bound, 0.0)
        self.assertAlmostEqual(sampled.absolute_bound, 0.02, places=12)
        self.assertFalse(sampled.rigorous)

        x = sp.Symbol("x")
        M = symbolic_derivative_bound(x**2, x, -0.1, 0.1)
        rigorous = scalar_fixed.bound_function_error(
            lambda t: t * t,
            lambda t: 2.0 * t,
            0.0,
            0.1,
            (-1.0, 1.0),
            derivative_bound=M,
            derivative_bound_verified=True,
        )
        self.assertTrue(rigorous.rigorous)
        self.assertAlmostEqual(rigorous.bound, 0.02, places=12)


class TestNonlinearFixedPointReport(unittest.TestCase):
    def test_report_columns_match_values(self):
        result = {
            "rows": [
                (0, [2.0], None, 1.0, None, None, None),
                (1, [1.5], 0.5, 0.5, 0.5, 0.5, 0.5),
            ]
        }
        rows = nonlinear_fixed.iteration_report_rows(result, "vo cung")
        self.assertEqual(rows[1]["step_difference"], 0.5)
        self.assertEqual(rows[1]["absolute_error_bound"], 0.5)
        self.assertEqual(rows[1]["relative_error_bound"], 0.5)
        self.assertEqual(rows[1]["residual_norm"], 0.5)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            nonlinear_fixed.print_iteration_table(result, 5, "relative", "vo cung")
        text = output.getvalue()
        for label in (
            "step_difference",
            "absolute_error_bound",
            "relative_error_bound",
            "residual_norm",
        ):
            self.assertIn(label, text)


class TestNewtonSystemOrder(unittest.TestCase):
    def setUp(self):
        self.x = sp.Symbol("x")

    def solve(self, expression, initial, **kwargs):
        return system_newton.newton_system(
            [expression], [self.x], np.array([initial], dtype=float),
            kwargs.pop("epsilon", 1e-10), kwargs.pop("max_iter", 50),
            **kwargs,
        )

    def test_exact_multiple_root_before_singular_jacobian(self):
        result = self.solve(self.x**2, 0.0)
        self.assertTrue(result.converged)
        self.assertEqual(result.status, "converged")
        self.assertEqual(result.records, [])
        self.assertEqual(result.residual_norm, 0.0)
        self.assertIn("ban đầu", result.reason)

    def test_linear_and_nonlinear_convergence(self):
        linear = self.solve(self.x - 2, 0.0)
        nonlinear = self.solve(self.x**2 - 2, 1.0)
        self.assertTrue(linear.converged and nonlinear.converged)
        self.assertLess(linear.residual_norm, 1e-10)
        self.assertLess(nonlinear.residual_norm, 1e-10)

    def test_singular_not_root_max_iter_and_large_step(self):
        singular = self.solve(self.x**2 + 1, 0.0)
        self.assertFalse(singular.converged)
        self.assertEqual(singular.status, "singular_jacobian")
        limited = self.solve(self.x**2 - 2, 1.0, epsilon=1e-30, max_iter=1)
        self.assertFalse(limited.converged)
        self.assertEqual(limited.status, "max_iter_reached")
        huge = self.solve(self.x - 100, 0.0, maximum_step_norm=10.0)
        self.assertEqual(huge.status, "step_too_large")

    def test_nonfinite_function(self):
        result = self.solve(1 / (self.x - 1), 1.0)
        self.assertFalse(result.converged)
        self.assertEqual(result.status, "numerical_failure")

    def test_fixed_steps_do_not_stop_at_an_early_exact_root(self):
        result = self.solve(self.x - 2, 0.0, fixed_iterations=4)
        self.assertEqual(result.status, "fixed_steps")
        self.assertEqual(len(result.records), 4)
        self.assertEqual(result.x[0], 2.0)
        self.assertIn("đúng k=4", result.reason)


class TestLinearFixedPoint(unittest.TestCase):
    def test_convergence_initial_solution_relative_residual_and_max_iter(self):
        A = np.eye(2)
        b = np.array([1.0, -2.0])
        B = 0.5 * np.eye(2)
        d = 0.5 * b
        result = linear_fixed.simple_iteration(A, b, B, d, np.zeros(2), "inf", 1e-9, 100, 0)
        self.assertTrue(result.converged)
        self.assertLess(result.records[-1].residual, 1e-9)
        self.assertLess(result.records[-1].relative_residual, 1e-9)

        exact = linear_fixed.simple_iteration(A, b, B, d, b.copy(), "inf", 1e-9, 100, 0)
        self.assertTrue(exact.converged)
        self.assertEqual(exact.records, [])

        limited = linear_fixed.simple_iteration(A, b, B, d, np.zeros(2), "inf", 1e-30, 1, 0)
        self.assertFalse(limited.converged)

    def test_noncontraction_singular_and_bad_dimensions(self):
        with self.assertRaises(ValueError):
            linear_fixed.simple_iteration(
                np.eye(2), np.ones(2), 2 * np.eye(2), np.ones(2),
                np.zeros(2), "inf", 1e-8, 10, 0,
            )
        singular = np.array([[1.0, 0.0], [0.0, 0.0]])
        with self.assertRaises(ValueError):
            linear_fixed.simple_iteration(
                singular, np.ones(2), np.eye(2) - singular, np.ones(2),
                np.zeros(2), "inf", 1e-8, 10, 0,
            )
        with self.assertRaises(ValueError):
            linear_fixed.simple_iteration(
                np.eye(2), np.ones((2, 1)), 0.5 * np.eye(2), np.ones(2),
                np.zeros(2), "inf", 1e-8, 10, 0,
            )

    def test_linear_fixed_steps_do_not_stop_at_initial_solution(self):
        A = np.eye(2)
        b = np.array([1.0, -2.0])
        result = linear_fixed.simple_iteration(
            A, b, np.zeros((2, 2)), b.copy(), b.copy(), "inf", 1e-9, 20, 5
        )
        self.assertEqual(len(result.records), 5)
        np.testing.assert_array_equal(result.x, b)

        import lapjacobituyentinh as jacobi

        fixed = jacobi.jacobi_solve(
            A, b, b.copy(), stop_mode="fixed", fixed_steps=5, max_iter=20
        )
        self.assertEqual(fixed["status"], "fixed_steps")
        self.assertEqual(len(fixed["history"]), 6)

    def test_richardson_direct_menu_does_not_request_A_or_b(self):
        output = io.StringIO()
        with patch(
            "builtins.input",
            side_effect=[
                "2", "2",       # nhập trực tiếp B,d; n
                "0, 0", "0, 0",  # B
                "1, -2",          # d
                "1",              # chuẩn vô cùng
                "",               # x^(0)=0
                "1", "2",        # đúng k bước, k=2
                "5",              # chữ số hiển thị
            ],
        ), contextlib.redirect_stdout(output):
            linear_fixed.main()
        text = output.getvalue()
        self.assertIn("Không yêu cầu nhập A hoặc b", text)
        self.assertIn("x-Bx-d", text)
        self.assertNotIn("Nhập ma trận A", text)
        self.assertNotIn("Nhập vector b", text)

    def test_jacobi_direct_fixed_point_formula_and_history(self):
        import lapjacobituyentinh as jacobi

        B = np.array([[0.0, 0.5], [0.25, 0.0]])
        d = np.array([1.0, -1.0])
        result = jacobi.jacobi_fixed_point(
            B,
            d,
            np.zeros(2),
            stop_mode="fixed",
            fixed_steps=3,
            norm_kind="inf",
        )
        self.assertEqual(result["status"], "fixed_steps")
        self.assertEqual(len(result["history"]), 4)
        expected = np.zeros(2)
        for _ in range(3):
            expected = B @ expected + d
        np.testing.assert_allclose(result["x"], expected)


class TestDirectMainPaths(unittest.TestCase):
    def run_main(self, function, answers):
        output = io.StringIO()
        with patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(output):
            function()
        return output.getvalue()

    def test_svd_main_uses_scaled_core(self):
        text = self.run_main(
            svd_module.main,
            ["2", "2", "1e-100 0", "0 2e-100", "", "", "", "", "100", "7"],
        )
        self.assertIn("2.000000000000e-100", text)
        self.assertIn("SVD đạt đồng thời", text)

    def test_eigen_main_does_not_deflate_failed_rotation(self):
        text = self.run_main(
            eigen_module.main,
            ["2", "0 -1", "1 0", "", "", "", "", ""],
        )
        self.assertIn("Không tìm được trị riêng thực trội", text)
        self.assertIn("Không thực hiện xuống thang", text)
        self.assertIn("Không tìm được cặp trị riêng", text)

    def test_scalar_fixed_main_reports_left_domain(self):
        text = self.run_main(
            scalar_fixed.fixed_point_iteration,
            ["2*x", "y", "G", "x", "6", "n", "-1", "1", "0.2", "1", "0.5"],
        )
        self.assertIn("không chứng nhận hội tụ", text.lower())
        self.assertIn("rời khỏi [a,b]", text)
        self.assertNotIn("Quá trình lặp hoàn tất tại bước 1", text)

    def test_newton_main_accepts_exact_singular_root(self):
        text = self.run_main(
            system_newton.main,
            ["1", "x", "x**2", "0", "", "", "", ""],
        )
        self.assertIn("Nghiệm ban đầu đã thỏa hệ", text)
        self.assertIn("nghiệm gần đúng thỏa", text)


if __name__ == "__main__":
    unittest.main()
