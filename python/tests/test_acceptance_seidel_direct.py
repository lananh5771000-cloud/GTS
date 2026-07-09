import contextlib
import io
from unittest.mock import patch

import numpy as np

import seideltuyentinh as seidel


B_SAMPLE = np.array(
    [
        [0.04, 0.07, -0.01, 0.00, -0.05],
        [-0.10, 0.04, -0.02, -0.01, 0.04],
        [-0.05, -0.04, 0.06, 0.03, 0.03],
        [-0.10, 0.09, 0.06, 0.04, -0.07],
        [-0.08, -0.10, -0.07, 0.05, -0.08],
    ]
)
D_SAMPLE = np.array([-7.0, 6.0, -4.0, 1.0, -7.0])


def test_seidel_direct_fixed_point_sample_exactly_five_steps():
    result, info = seidel.seidel_fixed_point(
        B_SAMPLE,
        D_SAMPLE,
        x0=np.zeros(5),
        fixed_iterations=5,
        show=False,
    )
    np.testing.assert_allclose(
        result,
        [-6.43189764, 6.72400932, -4.31753806, 2.52650200, -6.23085384],
        rtol=0.0,
        atol=5e-9,
    )
    assert info["iterations"] == 5
    assert info["fixed_iterations_completed"]
    assert len(info["history"]) == 6
    np.testing.assert_allclose(info["residual_vector"], result - B_SAMPLE @ result - D_SAMPLE)


def test_direct_fixed_point_steps_differ_from_transformed_linear_system():
    direct, _ = seidel.seidel_fixed_point(
        B_SAMPLE, D_SAMPLE, fixed_iterations=5, show=False
    )
    transformed, _ = seidel.gauss_seidel(
        np.eye(5) - B_SAMPLE,
        D_SAMPLE,
        fixed_iterations=5,
        auto_reorder=False,
        show=False,
    )
    assert np.linalg.norm(direct - transformed, np.inf) > 1e-6


def test_direct_fixed_steps_never_stop_early_and_reject_bad_data():
    result, info = seidel.seidel_fixed_point(
        np.zeros((2, 2)),
        np.array([1.0, -2.0]),
        x0=np.array([1.0, -2.0]),
        fixed_iterations=4,
        show=False,
    )
    np.testing.assert_array_equal(result, [1.0, -2.0])
    assert info["iterations"] == 4
    assert len(info["history"]) == 5
    for bad in (np.nan, np.inf, -np.inf):
        matrix = np.eye(2)
        matrix[0, 0] = bad
        try:
            seidel.seidel_fixed_point(matrix, np.zeros(2), show=False)
        except ValueError:
            pass
        else:
            raise AssertionError("B không hữu hạn phải bị từ chối")


def test_direct_menu_does_not_request_A_or_b():
    answers = [
        "2", "2",       # menu, n
        "0 0", "0 0",  # B
        "1 -2",          # d
        "",              # x^(0)=0
        "1", "1",       # đúng k bước, k=1
        "5",             # chữ số hiển thị
    ]
    output = io.StringIO()
    with patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(output):
        seidel.main()
    text = output.getvalue()
    assert "SEIDEL TRỰC TIẾP CHO x=B x+d" in text
    assert "Không yêu cầu nhập A hoặc b" in text
    assert "Tổng thứ hai chứa bᵢᵢ" in text
