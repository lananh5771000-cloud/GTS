import contextlib
import io

import numpy as np

import lapjacobituyentinh as jacobi


def _manual_column_norms(matrix, norm_kind):
    values = np.abs(np.asarray(matrix, dtype=float))
    if norm_kind == "inf":
        return np.max(values, axis=0)
    if norm_kind == "one":
        return np.sum(values, axis=0)
    raise AssertionError(f"unexpected norm kind: {norm_kind}")


def test_row_dominant_uses_alpha_infinity_certificate():
    A = np.array([[4.0, 1.0], [2.0, 3.0]])
    B = np.array([[1.0], [2.0]])

    result = jacobi.jacobi_solve(
        A, B, np.zeros_like(B), stop_mode="fixed", fixed_steps=5, max_iter=50
    )

    assert result["status"] == "fixed_steps"
    assert result["kind"] == "row"
    assert result["p"] == "inf"
    expected_q = float(np.max(np.sum(np.abs(result["alpha"]), axis=1)))
    assert result["q"] == expected_q
    assert result["certificate_matrix"] is result["alpha"]

    step = result["history"][-1]
    diff = step["X"] - result["history"][-2]["X"]
    np.testing.assert_allclose(
        step["diff_norms"], _manual_column_norms(diff, "inf")
    )


def test_column_dominant_not_row_uses_alpha_tilde_one_certificate_and_prints_it():
    A = np.array([[3.0, 4.0], [2.0, 5.0]])
    B = np.array([[7.0], [8.0]])

    result = jacobi.jacobi_solve(
        A, B, np.zeros_like(B), stop_mode="fixed", fixed_steps=5, max_iter=50
    )

    assert result["status"] == "fixed_steps"
    assert result["kind"] == "column"
    assert result["p"] == "one"
    alpha_tilde = np.eye(2) - result["A"] @ result["T"]
    np.testing.assert_allclose(result["alpha_tilde"], alpha_tilde)
    assert result["certificate_matrix"] is result["alpha_tilde"]

    old_wrong_q = float(np.max(np.sum(np.abs(result["alpha"]), axis=0)))
    expected_q = float(np.max(np.sum(np.abs(alpha_tilde), axis=0)))
    assert old_wrong_q > 1.0
    assert result["q"] == expected_q
    assert result["q"] < 1.0

    step = result["history"][-1]
    diff = step["X"] - result["history"][-2]["X"]
    np.testing.assert_allclose(
        step["diff_norms"], _manual_column_norms(diff, "one")
    )
    np.testing.assert_allclose(
        step["error_bounds"], result["error_factor"] * step["diff_norms"]
    )

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        jacobi.print_solution(result, task_mode=1, decimals=6)
    text = output.getvalue()
    assert "q = ‖M_J‖" not in text
    assert "α~ = I - A.T" in text
    assert "Các tổng cột của |α~|" in text


def test_row_permutation_still_iterates_and_uses_infinity_norm():
    A = np.array([[1.0, 3.0], [4.0, 1.0]])
    B = np.array([[4.0, 7.0], [5.0, 6.0]])

    result = jacobi.jacobi_solve(
        A, B, np.zeros_like(B), stop_mode="fixed", fixed_steps=5, max_iter=50
    )

    assert result["status"] == "fixed_steps"
    assert result["kind"] == "row"
    assert result["p"] == "inf"
    assert result["permutation"] == (1, 0)
    assert result["q"] < 1.0
    assert len(result["history"]) == 6

    step = result["history"][-1]
    diff = step["X"] - result["history"][-2]["X"]
    np.testing.assert_allclose(
        step["diff_norms"], _manual_column_norms(diff, "inf")
    )
    np.testing.assert_allclose(
        step["error_bounds"], result["error_factor"] * step["diff_norms"]
    )
