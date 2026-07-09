"""Wrapper tương thích cho tệp lịch sử ``gauss (2).py``.

Phần cài đặt được giữ trong thư mục ``python``; wrapper này giúp các tài liệu và
kiểm thử cũ tiếp tục chạy từ thư mục gốc của project.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_python_dir = Path(__file__).resolve().parent / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))
_namespace = runpy.run_path(
    str(_python_dir / "gauss (2).py"),
    run_name="gauss_jordan_implementation",
)
globals().update({name: value for name, value in _namespace.items() if not name.startswith("__")})


if __name__ == "__main__":
    try:
        _namespace["solve_gauss_jordan"]()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã dừng chương trình; không có dữ liệu đầu vào đầy đủ.")
