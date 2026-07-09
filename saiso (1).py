"""Wrapper tương thích cho tệp lịch sử ``saiso (1).py``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_python_dir = Path(__file__).resolve().parent / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))
_namespace = runpy.run_path(
    str(_python_dir / "saiso (1).py"),
    run_name="sai_so_implementation",
)
globals().update({name: value for name, value in _namespace.items() if not name.startswith("__")})


if __name__ == "__main__":
    try:
        _namespace["chuong_trinh_sai_so"]()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã dừng chương trình; không có dữ liệu đầu vào đầy đủ.")
