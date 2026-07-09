import contextlib
import io
import math
import unittest
from fractions import Fraction

import numpy as np

import cholesky
import lapdon_tuyentinh
import nghichdao_newton
from chiadoi_pt import bisection
from daycung import chord_method
from exam_format import (
    DisplayDigits,
    format_error,
    format_matrix,
    format_matrix_with_digits,
    format_number,
    format_scalar,
    format_vector_with_digits,
    indexed,
    inverse,
    iteration,
    matrix_entry,
    norm,
    prettify_math,
    safe_unicode,
    to_subscript,
    to_superscript,
)
from phantach_lu import plu_decomposition
from tieptuyen import safeguarded_newton


class TestExamFormatting(unittest.TestCase):
    def test_required_unicode_notation(self):
        self.assertEqual(to_subscript("0123456789+-=()ijkn"), "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ᵢⱼₖₙ")
        self.assertEqual(to_superscript("0123456789+-=()ijkn"), "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱʲᵏⁿ")
        self.assertEqual(indexed("λ", 1), "λ₁")
        self.assertEqual(indexed("σ", 1), "σ₁")
        self.assertEqual(matrix_entry("a", "i", "j"), "aᵢⱼ")
        self.assertEqual(iteration("x", "k+1", 1), "x₁⁽ᵏ⁺¹⁾")
        self.assertEqual(inverse("A"), "A⁻¹")
        self.assertEqual(norm("x", "inf"), "‖x‖∞")

    def test_matrix_alignment_fraction_and_negative_zero(self):
        rendered = format_matrix("A⁻¹", [[Fraction(1, 3), -0.0], [-12, 2.5]], 6)
        self.assertIn("A⁻¹ = ⎡ 1/3", rendered)
        self.assertIn("⎣ -12", rendered)
        self.assertNotIn("-0", rendered)
        self.assertNotIn("np.float64", rendered)
        self.assertEqual(format_number(np.float64(-0.0)), "0")
        self.assertEqual(format_number(1e-100), "1.000000e-100")

    def test_separate_display_digits_for_matrix_vector_scalar_and_error(self):
        digits = DisplayDigits(matrix=4, vector=7, scalar=7, error=7)
        matrix = format_matrix_with_digits("A", [[1 / 3]], digits)
        vector = format_vector_with_digits("x", [1 / 3], digits, column=False)
        scalar = format_scalar(1 / 3, digits)
        error = format_error(1 / 3, digits)
        self.assertIn("0.3333", matrix)
        self.assertIn("0.3333333", vector)
        self.assertEqual(scalar, "0.3333333")
        self.assertEqual(error, "0.3333333")

    def test_bad_format_data_and_ascii_fallback(self):
        with self.assertRaises(ValueError):
            format_matrix("A", [[1, 2], [3]])
        with self.assertRaises(ValueError):
            format_number(math.nan)
        fallback = safe_unicode("λ₁ ≈ σ₁; A⁻¹", "ascii")
        self.assertTrue(fallback.isascii())
        self.assertIn("lambda_1", fallback)

    def test_legacy_notation_is_cleaned_for_student_output(self):
        rendered = prettify_math("a_1, a_ij, x_1^(k+1), lambda_1, sigma_1, A^(-1)")
        self.assertEqual(rendered, "a₁, aᵢⱼ, x₁⁽ᵏ⁺¹⁾, λ₁, σ₁, A⁻¹")
        for legacy in ("a_1", "a_ij", "x_1", "lambda_1", "sigma_1", "A^(-1)"):
            self.assertNotIn(legacy, rendered)

    def test_cholesky_parenthesis_is_preserved_and_formatter_is_idempotent(self):
        source = "u_ii = sqrt(a_ii - Σ(k=1..i-1) u_ki^2)"
        rendered = prettify_math(source)
        self.assertEqual(rendered.count("("), rendered.count(")"))
        self.assertTrue(rendered.endswith(")"))
        self.assertEqual(prettify_math(rendered), rendered)

    def test_multiple_powers_do_not_consume_neighbouring_parentheses(self):
        rendered = prettify_math("(x_i^2 + y_j^(k+1)) / (1 + z_n^3)")
        self.assertEqual(rendered.count("("), rendered.count(")"))
        self.assertIn("xᵢ²", rendered)
        self.assertIn("yⱼ⁽ᵏ⁺¹⁾", rendered)
        self.assertIn("zₙ³", rendered)

    def test_full_legacy_notation_set(self):
        source = (
            "a_1 a_12 a_ij u_ii u_ki x_k x_(k+1) x_{n+1} "
            "x^(k+1) lambda_1 sigma_i A^-1 A^(-1) A_(n-1)^(-1) "
            "A^T U^T ||A||_1 ||A||_2 ||A||_inf <= >= != ~="
        )
        rendered = prettify_math(source)
        for legacy in ("a_1", "x_k", "lambda_1", "A^(-1)", "^T", "<=", ">=", "!=", "~="):
            self.assertNotIn(legacy, rendered)
        for expected in (
            "a₁", "a₁₂", "aᵢⱼ", "uᵢᵢ", "uₖᵢ", "xₖ", "x⁽ᵏ⁺¹⁾",
            "xₙ₊₁", "λ₁", "σᵢ", "A⁻¹", "Aₙ₋₁⁻¹", "Aᵀ", "Uᵀ",
            "‖A‖₁", "‖A‖₂", "‖A‖∞", "≤", "≥", "≠", "≈",
        ):
            self.assertIn(expected, rendered)
        self.assertEqual(prettify_math(rendered), rendered)


class TestAdditionalNumericalEdges(unittest.TestCase):
    def test_bisection_rejects_discontinuity_with_clear_message(self):
        with self.assertRaisesRegex(ArithmeticError, "Không thể áp dụng phương pháp chia đôi"):
            bisection(lambda x: 1 / x, -1.0, 1.0, 1e-6)

    def test_exact_endpoints_are_certified_without_continuity_assumption(self):
        self.assertTrue(bisection(lambda x: x, 0.0, 2.0, 1e-6).certified)
        self.assertTrue(chord_method(lambda x: x, 0.0, 2.0, 1e-6).certified)
        self.assertTrue(
            safeguarded_newton(lambda x: x, lambda _x: 1.0, 0.0, 2.0, 1e-6).certified
        )

    def test_small_endpoint_residual_is_not_called_exact(self):
        result = bisection(lambda x: x + 1e-15, 0.0, 1.0, 1e-6)
        self.assertTrue(result.converged)
        self.assertFalse(result.certified)
        self.assertGreater(result.error_bound, 0.0)
        self.assertIn("chưa chứng nhận", result.reason)

    def test_nonfinite_epsilon_and_near_zero_pivot(self):
        with self.assertRaises(ValueError):
            bisection(lambda x: x, -1.0, 1.0, math.nan)
        _p, _l, _u, _swaps, near = plu_decomposition(
            [[1.0, 0.0], [0.0, 1e-14]], pivot_tolerance=1e-12
        )
        self.assertTrue(near)

    def test_cholesky_rejects_nonsymmetric_and_indefinite(self):
        with contextlib.redirect_stdout(io.StringIO()):
            nonsymmetric = cholesky.cholesky_decomposition([[1, 2], [0, 1]], 6, False)
            indefinite = cholesky.cholesky_decomposition([[1, 2], [2, 1]], 6, False)
        self.assertEqual(nonsymmetric, (None, "not_symmetric"))
        self.assertEqual(indefinite, (None, "not_positive_definite"))

    def test_newton_schulz_rejects_singular_matrix(self):
        matrix = [[1.0, 2.0], [2.0, 4.0]]
        initial = nghichdao_newton.automatic_initial_approximation(matrix)[0]
        with contextlib.redirect_stdout(io.StringIO()):
            result = nghichdao_newton.newton_inverse(matrix, initial, 1e-8, 20, 6)
        self.assertIsNone(result)

    def test_manual_symmetric_jacobi_eigenvalues(self):
        matrix = np.array([[4.0, 1.0, -2.0], [1.0, 3.0, 0.5], [-2.0, 0.5, 5.0]])
        values = lapdon_tuyentinh.symmetric_jacobi_eigenvalues(matrix)
        np.testing.assert_allclose(values, np.linalg.eigvalsh(matrix), rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            lapdon_tuyentinh.symmetric_jacobi_eigenvalues([[1.0, 2.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
