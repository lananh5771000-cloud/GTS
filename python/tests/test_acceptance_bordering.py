import contextlib
import io
from fractions import Fraction

import nghichdao_vienquanh as bordering


def test_bordering_six_by_six_with_zero_a11_and_permutations():
    A = [
        [Fraction(int(i == (j + 1) % 6)) for j in range(6)]
        for i in range(6)
    ]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        inverse = bordering.find_inverse_by_bordering(A, 7, output_mode="exam")
    assert inverse is not None
    assert bordering.multiply_matrices(A, inverse) == bordering.identity_matrix(6)
    assert bordering.multiply_matrices(inverse, A) == bordering.identity_matrix(6)
    text = output.getvalue()
    assert "B=P A Q" in text
    assert "θ =" in text
    assert "khối trên-trái" in text
    assert "AA⁻¹" in text and "A⁻¹A" in text


def test_bordering_three_output_modes_and_singular_matrix():
    A = [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(2)]]
    lengths = {}
    for mode in ("full", "exam", "result"):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            inverse = bordering.find_inverse_by_bordering(A, 6, output_mode=mode)
        assert inverse == [[Fraction(2, 3), Fraction(-1, 3)], [Fraction(-1, 3), Fraction(2, 3)]]
        lengths[mode] = len(output.getvalue().splitlines())
    assert lengths["full"] > lengths["exam"] > lengths["result"]

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        inverse = bordering.find_inverse_by_bordering(
            [[Fraction(1), Fraction(2)], [Fraction(2), Fraction(4)]],
            6,
            output_mode="result",
        )
    assert inverse is None
    assert "suy biến" in output.getvalue()

