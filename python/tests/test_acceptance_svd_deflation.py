import contextlib
import io

import numpy as np

import khai_trien_ky_di_svd as svd


A_SAMPLE = np.array(
    [[3.0, 3.0, 7.0], [3.0, 9.0, 10.0], [8.0, 4.0, 6.0], [2.0, 10.0, 9.0]]
)


def test_reduced_svd_sample_dimensions_values_and_checks():
    result = svd.svd_power(A_SAMPLE, epsilon=1e-10, full_matrices=False)
    assert result.U.shape == (4, 3)
    assert result.Vt.shape == (3, 3)
    assert result.rank == 3
    np.testing.assert_allclose(
        result.singular_values,
        [22.64825219, 6.28726328, 2.35095584],
        rtol=0.0,
        atol=5e-8,
    )
    assert result.converged
    assert result.relative_reconstruction_error < result.reconstruction_tolerance
    assert result.left_orthogonality_error < result.orthogonality_tolerance
    assert result.right_orthogonality_error < result.orthogonality_tolerance


def test_hotelling_deflation_really_updates_B_zero_through_B_three():
    result = svd.svd_power(A_SAMPLE, epsilon=1e-10, full_matrices=False)
    matrices = result.deflation_matrices
    powers = result.power_results
    assert matrices is not None and powers is not None
    assert len(matrices) == 4
    np.testing.assert_allclose(matrices[0], (A_SAMPLE / result.scale_factor).T @ (A_SAMPLE / result.scale_factor))
    for index, power in enumerate(powers, start=1):
        expected = matrices[index - 1] - max(power.eigenvalue, 0.0) * np.outer(
            power.eigenvector, power.eigenvector
        )
        np.testing.assert_allclose(matrices[index], expected, atol=2e-14)
        assert not np.array_equal(matrices[index], matrices[index - 1])

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        svd.print_deflation_history(A_SAMPLE, result, 7)
        svd.print_svd_result(A_SAMPLE, result, 7, form="reduced")
    text = output.getvalue()
    for label in ("B₀", "B₁", "B₂", "B₃", "Uᵣ", "Σᵣ", "Vᵣᵀ"):
        assert label in text
    assert "Kích thước: U=(4, 3), Σ=(3, 3), Vᵀ=(3, 3)" in text
    assert "SVD đạt đồng thời" in text


def test_full_and_economy_shapes_are_printed_without_shape_errors():
    result = svd.svd_power(A_SAMPLE, epsilon=1e-10, full_matrices=True)
    assert result.U.shape == (4, 4)
    assert result.Vt.shape == (3, 3)
    for form, expected in (
        ("full", "U=(4, 4), Σ=(4, 3), Vᵀ=(3, 3)"),
        ("economy", "U=(4, 3), Σ=(3, 3), Vᵀ=(3, 3)"),
    ):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            svd.print_svd_result(A_SAMPLE, result, 6, form=form)
        assert expected in output.getvalue()


def test_reduced_zero_matrix_prints_empty_factors_without_crashing():
    A = np.zeros((2, 3))
    result = svd.svd_power(A, full_matrices=False)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        svd.print_svd_result(A, result, 6, form="reduced")
    text = output.getvalue()
    assert "U=(2, 0), Σ=(0, 0), Vᵀ=(0, 3)" in text
    assert "SVD đạt đồng thời" in text
