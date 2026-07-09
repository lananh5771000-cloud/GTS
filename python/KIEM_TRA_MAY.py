"""Kiểm tra nhanh máy thi và các nhóm thuật toán, hoàn toàn ngoại tuyến."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def quiet_call(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def load_file(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không tạo được bộ nạp cho {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_checks() -> bool:
    checks: list[tuple[str, object]] = []

    def check(name: str, action):
        try:
            detail = action()
            checks.append((name, detail if detail is not None else "OK"))
        except Exception as error:  # Báo từng mục, không làm mất các kiểm tra sau.
            checks.append((name, error))

    check("Python >= 3.10", lambda: sys.version.split()[0] if sys.version_info >= (3, 10) else (_ for _ in ()).throw(RuntimeError(sys.version)))

    def dependencies():
        import mpmath
        import numpy
        import sympy

        return f"numpy {numpy.__version__}; sympy {sympy.__version__}; mpmath {mpmath.__version__}"

    check("Thư viện số", dependencies)
    check("UTF-8", lambda: "UTF-8" if "utf" in (getattr(sys.stdout, "encoding", "") or "").lower() else (_ for _ in ()).throw(RuntimeError("stdout không dùng UTF-8")))

    module_names = [
        "chiadoi_pt", "cholesky", "danilevski_tri_rieng", "daycung",
        "exam_format", "gaussrank", "hptlapdonphituyen",
        "khai_trien_ky_di_svd", "lapdon_pt", "lapdon_tuyentinh",
        "lapjacobituyentinh", "newton_he_phi_tuyen", "nghichdao_newton",
        "nghichdao_vienquanh", "phantach_lu", "phuong_phap_da_thuc",
        "seideltuyentinh", "tieptuyen", "tri_rieng_troi_xuong_thang",
        "truyen_sai_so", "input_utils",
    ]

    def import_everything():
        for name in module_names:
            importlib.import_module(name)
        load_file("gauss (2).py", "gauss_legacy_check")
        load_file("saiso (1).py", "saiso_legacy_check")
        return f"{len(module_names) + 2} module"

    check("Import toàn project", import_everything)

    def smoke_input():
        from exam_format import DisplayDigits, format_error, format_matrix_with_digits
        from input_utils import parse_exact, parse_real, parse_real_row
        import sympy as sp

        assert abs(parse_real("1/3+√2") - (1 / 3 + 2**0.5)) < 1e-14
        assert sp.simplify(parse_exact("sqrt(8)") - 2 * sp.sqrt(2)) == 0
        assert parse_real_row("1/2, -3.5, 1e-4", 3) == [0.5, -3.5, 1e-4]
        digits = DisplayDigits(matrix=4, vector=7, scalar=7, error=7)
        assert "0.3333" in format_matrix_with_digits("A", [[1 / 3]], digits)
        assert format_error(1 / 3, digits) == "0.3333333"
        return "phân số, căn, π, dấu phẩy, chữ số hiển thị"

    check("Nhập biểu thức", smoke_input)

    def smoke_equation():
        from chiadoi_pt import bisection

        result = bisection(lambda x: x * x - 2, 1.0, 2.0, 1e-8, continuity_verified=True)
        assert result.converged and abs(result.root**2 - 2) < 1e-7
        return "chia đôi OK"

    check("Phương trình một biến", smoke_equation)

    def smoke_nonlinear_system():
        import numpy as np
        import sympy as sp
        from newton_he_phi_tuyen import newton_system

        x, y = sp.symbols("x y")
        result = newton_system([x + y - 3, x - y - 1], [x, y], np.array([0.0, 0.0]), 1e-10, 20)
        assert result.converged and np.linalg.norm(result.x - [2, 1], np.inf) < 1e-9
        return "Newton hệ OK"

    check("Hệ phi tuyến", smoke_nonlinear_system)

    def smoke_linear():
        from phantach_lu import plu_solve

        result = quiet_call(plu_solve, [[2, 1], [1, 3]], [[1], [2]], show_steps=False)
        solution = result[0] if isinstance(result, tuple) else result
        assert abs(float(solution[0][0]) - 0.2) < 1e-12
        assert abs(float(solution[1][0]) - 0.6) < 1e-12
        return "PLU + residual OK"

    check("Hệ tuyến tính", smoke_linear)

    def smoke_cholesky():
        import sympy as sp
        from cholesky import cholesky_decomposition

        upper, _ = quiet_call(cholesky_decomposition, [[2, 1], [1, 2]], 7, show_steps=False)
        U = sp.Matrix(upper)
        assert sp.simplify(U.T * U - sp.Matrix([[2, 1], [1, 2]])) == sp.zeros(2)
        return "A = UᵀU OK"

    check("Cholesky", smoke_cholesky)

    def smoke_iterative():
        import numpy as np
        from seideltuyentinh import gauss_seidel, seidel_fixed_point

        solution, info = gauss_seidel([[4.0, 1.0], [1.0, 3.0]], [1.0, 2.0], epsilon=1e-8, show=False)
        assert info["converged"] and np.linalg.norm(np.array([[4, 1], [1, 3]]) @ solution - [1, 2], np.inf) < 1e-7
        B = np.array([
            [0.04, 0.07, -0.01, 0.00, -0.05],
            [-0.10, 0.04, -0.02, -0.01, 0.04],
            [-0.05, -0.04, 0.06, 0.03, 0.03],
            [-0.10, 0.09, 0.06, 0.04, -0.07],
            [-0.08, -0.10, -0.07, 0.05, -0.08],
        ])
        direct, direct_info = seidel_fixed_point(
            B, [-7, 6, -4, 1, -7], fixed_iterations=5, show=False
        )
        assert direct_info["iterations"] == 5
        assert np.allclose(
            direct,
            [-6.43189764, 6.72400932, -4.31753806, 2.52650200, -6.23085384],
            atol=5e-9,
            rtol=0.0,
        )
        return "Seidel Ax=b và x=Bx+d OK"

    check("Lặp tuyến tính", smoke_iterative)

    def smoke_inverse():
        import numpy as np
        from nghichdao_vienquanh import bordering_inverse_recursive

        inverse, _, _ = quiet_call(bordering_inverse_recursive, [[2, 0], [0, 4]])
        assert np.linalg.norm(np.array([[2, 0], [0, 4]], float) @ np.array(inverse, float) - np.eye(2), np.inf) < 1e-12
        return "viền quanh + AA⁻¹ OK"

    check("Nghịch đảo", smoke_inverse)

    def smoke_eigen_svd():
        import numpy as np
        from khai_trien_ky_di_svd import svd_power
        from tri_rieng_troi_xuong_thang import dominant_eigenpair

        eigen = dominant_eigenpair([[3.0, 0.0], [0.0, 1.0]], [1.0, 1.0], epsilon=1e-10)
        assert eigen.result.converged and abs(eigen.result.eigenvalue - 3) < 1e-8
        matrix = np.array([[3, 3, 7], [3, 9, 10], [8, 4, 6], [2, 10, 9.0]])
        svd = svd_power(matrix, epsilon=1e-10)
        assert svd.converged and svd.relative_reconstruction_error < 1e-10
        assert svd.U.shape == (4, 3) and svd.Vt.shape == (3, 3)
        assert len(svd.deflation_matrices or []) == 4
        assert np.allclose(svd.singular_values, [22.64825219, 6.28726328, 2.35095584], atol=5e-8)
        return "trị riêng, SVD rút gọn và B₀…B₃ OK"

    check("Trị riêng / SVD", smoke_eigen_svd)

    def smoke_polynomial_error():
        from fractions import Fraction
        from phuong_phap_da_thuc import square_free_decomposition, sturm_real_root_count
        from truyen_sai_so import propagate_bound

        coefficients = [1, -8, 25, -38, 28, -8]
        factors = square_free_decomposition(coefficients)
        assert sorted(m for _, m in factors) == [2, 3]
        assert sturm_real_root_count(coefficients) == 2
        bound = propagate_bound(lambda x: x * x, lambda x: 2 * x, 2.0, 0.01, (1.9, 2.1), derivative_bound=4.2, derivative_bound_verified=True)
        assert bound.absolute_bound == 4.2 * 0.01 and Fraction(1, 3) == Fraction("1/3")
        return "đa thức bội + sai số OK"

    check("Đa thức / sai số", smoke_polynomial_error)

    width = max(len(name) for name, _ in checks)
    all_ok = True
    for name, result in checks:
        failed = isinstance(result, Exception)
        all_ok &= not failed
        print(f"[{('LỖI' if failed else 'OK'):4}] {name.ljust(width)} : {result}")
    if all_ok:
        print("\nHỆ THỐNG SẴN SÀNG CHẠY OFFLINE")
    else:
        print("\nMÁY CHƯA SẴN SÀNG; hãy xử lý các mục LỖI ở trên.")
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if run_checks() else 1)
