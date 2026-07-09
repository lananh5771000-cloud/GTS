import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT if (PROJECT_ROOT / "input_utils.py").exists() else PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from input_utils import (  # noqa: E402
    MathInputError,
    parse_exact,
    parse_exact_row,
    parse_math_expression,
    parse_real,
    parse_real_row,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1/3", 1 / 3),
        ("sqrt(2)", math.sqrt(2)),
        ("√2", math.sqrt(2)),
        ("2√(3)", 2 * math.sqrt(3)),
        ("∛8", 2.0),
        ("pi/4", math.pi / 4),
        ("sin(pi/6)", 0.5),
        ("2^(-3)", 0.125),
        ("½+¼", 0.75),
        ("1e-100", 1e-100),
        ("2E6", 2e6),
    ],
)
def test_parse_real_supported_forms(text, expected):
    assert parse_real(text) == pytest.approx(expected)


def test_exact_input_preserves_fraction_and_radical():
    assert parse_exact("2/6") == Fraction(1, 3)
    assert sp.simplify(parse_exact("sqrt(8)") - 2 * sp.sqrt(2)) == 0


def test_symbolic_expression_with_implicit_multiplication():
    x = sp.Symbol("x")
    expression = parse_math_expression("2x + √(x^2)", [x])
    assert sp.simplify(expression - (2 * x + sp.sqrt(x**2))) == 0


def test_rows_accept_fraction_radical_and_constants():
    values = parse_real_row("[1/2 √2 pi]", 3)
    assert values == pytest.approx([0.5, math.sqrt(2), math.pi])
    exact = parse_exact_row("1/2 sqrt(2)", 2)
    assert exact[0] == Fraction(1, 2)
    assert exact[1] == sp.sqrt(2)


def test_rows_accept_commas_semicolons_and_decimal_comma():
    assert parse_real_row("1, 2, 3", 3) == pytest.approx([1.0, 2.0, 3.0])
    assert parse_real_row("[1/2, -3.5, 1e-4]", 3) == pytest.approx([0.5, -3.5, 1e-4])
    assert parse_real_row("1; 2; 3", 3) == pytest.approx([1.0, 2.0, 3.0])
    assert parse_real_row("0,125", 1) == pytest.approx([0.125])
    assert parse_real_row("0,125 1", 2) == pytest.approx([0.125, 1.0])


@pytest.mark.parametrize(
    "text",
    ["", "1/0", "sqrt(-1)", "nan", "inf", "__import__('os')", "(1).__class__"],
)
def test_invalid_or_unsafe_input_is_rejected(text):
    with pytest.raises(MathInputError):
        parse_real(text)


def test_exact_matrix_readers_accept_radicals():
    import cholesky
    import danilevski_tri_rieng
    import gaussrank
    import nghichdao_vienquanh
    import phantach_lu

    for module in (
        cholesky,
        danilevski_tri_rieng,
        gaussrank,
        nghichdao_vienquanh,
        phantach_lu,
    ):
        with patch("builtins.input", return_value="1/2 √2 pi"):
            row = module.input_matrix_row("", 3)
        assert row[0] == Fraction(1, 2)
        assert sp.simplify(row[1] - sp.sqrt(2)) == 0
        assert row[2] == sp.pi


def test_float_matrix_readers_accept_radicals():
    import khai_trien_ky_di_svd as svd
    import lapdon_tuyentinh as richardson
    import lapjacobituyentinh as jacobi
    import newton_he_phi_tuyen as newton_system
    import seideltuyentinh as seidel
    import tri_rieng_troi_xuong_thang as eigen

    assert svd.parse_number("√2") == pytest.approx(math.sqrt(2))
    assert richardson.parse_number("pi/2") == pytest.approx(math.pi / 2)
    assert jacobi.read_number("1/√2") == pytest.approx(1 / math.sqrt(2))
    assert newton_system.parse_number("∛8") == pytest.approx(2)
    assert seidel.parse_number("sin(pi/6)") == pytest.approx(0.5)
    assert eigen.parse_real("2√3") == pytest.approx(2 * math.sqrt(3))


def test_polynomial_core_accepts_int_float_fraction_and_radical_coefficients():
    import phuong_phap_da_thuc as polynomial

    assert polynomial.sturm_real_root_count([1, -2.0, Fraction(1)]) == 1
    coefficients = [sp.sqrt(2), 0, -sp.sqrt(2)]
    assert polynomial.sturm_real_root_count(coefficients) == 2
    assert polynomial.polynomial_text(coefficients) == "sqrt(2)*x^2 - sqrt(2)"
