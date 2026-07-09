import sympy as sp
from exam_format import exam_print as print
import sys
import math
from dataclasses import dataclass
from typing import Sequence
from input_utils import parse_math_expression, parse_real

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class ErrorPropagationResult:
    value: float
    absolute_error: float
    relative_error: float | None
    mode: str
    rigorous: bool


def propagate_first_order(
    value: float,
    partial_derivatives: Sequence[float],
    input_absolute_errors: Sequence[float],
) -> ErrorPropagationResult:
    """Xấp xỉ sai số bậc nhất, không gắn nhãn chặn nghiêm ngặt."""
    if len(partial_derivatives) != len(input_absolute_errors):
        raise ValueError("Số đạo hàm riêng và số sai số đầu vào phải bằng nhau.")
    data = [float(value), *map(float, partial_derivatives), *map(float, input_absolute_errors)]
    if not all(math.isfinite(item) for item in data):
        raise ValueError("Mọi dữ liệu phải hữu hạn.")
    if any(error < 0 for error in input_absolute_errors):
        raise ValueError("Sai số tuyệt đối không được âm.")
    absolute = sum(abs(d) * error for d, error in zip(partial_derivatives, input_absolute_errors))
    relative = absolute / abs(value) if value != 0 else None
    return ErrorPropagationResult(float(value), absolute, relative, "Xấp xỉ sai số bậc nhất", False)


def propagate_mean_value_bound(
    value: float,
    derivative_upper_bounds: Sequence[float],
    input_absolute_errors: Sequence[float],
) -> ErrorPropagationResult:
    """Chặn theo định lý giá trị trung bình khi M_i đã được chứng minh trên miền sai số."""
    if len(derivative_upper_bounds) != len(input_absolute_errors):
        raise ValueError("Số chặn đạo hàm và số sai số đầu vào phải bằng nhau.")
    data = [float(value), *map(float, derivative_upper_bounds), *map(float, input_absolute_errors)]
    if not all(math.isfinite(item) for item in data):
        raise ValueError("Mọi dữ liệu phải hữu hạn.")
    if any(bound < 0 for bound in derivative_upper_bounds) or any(
        error < 0 for error in input_absolute_errors
    ):
        raise ValueError("Chặn đạo hàm và sai số không được âm.")
    absolute = sum(bound * error for bound, error in zip(derivative_upper_bounds, input_absolute_errors))
    relative = absolute / abs(value) if value != 0 else None
    return ErrorPropagationResult(float(value), absolute, relative, "Chặn theo định lý giá trị trung bình", True)


def guaranteed_decimal_digits(relative_error: float) -> int:
    """Số chữ số thập phân chắc theo điều kiện delta <= 0.5*10^-n."""
    if not math.isfinite(relative_error) or relative_error < 0:
        raise ValueError("Sai số tương đối phải không âm và hữu hạn.")
    if relative_error == 0:
        raise ValueError("Sai số bằng 0 không cho một số hữu hạn chữ số chắc.")
    return max(0, math.floor(-math.log10(2.0 * relative_error)))


def in_buoc(tieu_de):
    print(f"\n{tieu_de}")
    print("-" * 50)


def chuong_trinh_sai_so():
    print("=" * 60)
    print(" CHƯƠNG TRÌNH GIẢI BÀI TOÁN SAI SỐ HÀM NHIỀU BIẾN ")
    print("=" * 60)
    print("Input: f(x₁,…,xₙ), giá trị gần đúng và sai số dữ liệu.")
    print("Output: sai số tuyệt đối/tương đối của f hoặc sai số cho phép của từng biến.")
    print("Xấp xỉ bậc nhất: Δf ≈ Σ|∂f/∂xᵢ|Δxᵢ; δf ≈ Δf/|f|.")
    print("Muốn có chặn nghiêm ngặt phải chặn đạo hàm trên toàn miền sai số.\n")

    # 1. Nhập thông tin hàm và biến
    ham_str = input(
        "1. Nhập biểu thức hàm f (VD: (r**2/2)*(x - sin(x)) hoặc pi*R**2*h): "
    )
    bien_str = input(
        "2. Nhập các biến, cách nhau bởi dấu phẩy (VD: r, x hoặc pi, R, h): "
    )

    # Ép sympy hiểu các chữ cái người dùng nhập là Symbol (kể cả chữ 'pi' nếu muốn đạo hàm theo pi như Ảnh 2)
    bien_list = [sp.Symbol(b.strip()) for b in bien_str.split(",")]
    n = len(bien_list)

    local_dict = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp}
    # Tự động gán các biến người dùng nhập vào local_dict để override các hằng số mặc định
    for b in bien_list:
        local_dict[b.name] = b

    f = parse_math_expression(ham_str, {b.name: b for b in bien_list})

    # 2. Nhập giá trị các biến
    print("\n3. Nhập giá trị tại điểm đang xét:")
    gia_tri_dict = {}
    for bien in bien_list:
        val = parse_real(input(f"   Giá trị của {bien} = "))
        if not math.isfinite(val):
            raise ValueError("Giá trị đầu vào phải hữu hạn.")
        gia_tri_dict[bien] = val

    # 3. Tính toán các thành phần cơ bản
    f_val = float(f.subs(gia_tri_dict).evalf())

    dao_ham_dict = {}
    dao_ham_val_dict = {}
    for bien in bien_list:
        df = sp.diff(f, bien)
        df_val = float(df.subs(gia_tri_dict).evalf())
        dao_ham_dict[bien] = df
        dao_ham_val_dict[bien] = df_val

    # 4. Lựa chọn bài toán
    print("\n" + "=" * 60)
    print("CHỌN LOẠI BÀI TOÁN:")
    print("1. BÀI TOÁN THUẬN (Tính sai số của hàm f)")
    print("2. BÀI TOÁN NGƯỢC (Nguyên lý ảnh hưởng đều / Chia đều sai số tuyệt đối)")
    lua_chon = input("Nhập lựa chọn (1 hoặc 2): ")

    if lua_chon == "1":
        in_buoc("BÀI TOÁN THUẬN")
        print("Nhập sai số tuyệt đối (Δ) của từng biến:")
        sai_so_bien = {}
        for bien in bien_list:
            sai_so_bien[bien] = parse_real(input(f"   Δ{bien} = "))
            if not math.isfinite(sai_so_bien[bien]) or sai_so_bien[bien] < 0:
                raise ValueError("Sai số đầu vào phải không âm và hữu hạn.")

        delta_f = sum(abs(dao_ham_val_dict[b]) * sai_so_bien[b] for b in bien_list)

        print("\nXấp xỉ sai số bậc nhất (không phải chặn nghiêm ngặt trong mọi trường hợp):")
        terms = [f"|∂f/∂{b}|Δ{b}" for b in bien_list]
        print(f"Δf ≈ {' + '.join(terms)}")

        print("\nThay số:")
        chi_tiet = " + ".join(
            [f"({abs(dao_ham_val_dict[b]):.4g} × {sai_so_bien[b]})" for b in bien_list]
        )
        print(f"Δf ≈ {chi_tiet} = {delta_f:.6g}")
        print(f"\n=> KẾT LUẬN: f = {f_val:.6g} ± {delta_f:.6g}")

    elif lua_chon == "2":
        in_buoc("BÀI TOÁN NGƯỢC")
        print("Loại sai số mục tiêu của hàm f:")
        print("a. Sai số tuyệt đối (Δf)")
        print(
            "b. Sai số tương đối (δf - nhập dưới dạng số thập phân, VD: 0.05% thì nhập 0.0005)"
        )
        loai_sai_so = input("Chọn a hoặc b: ").lower()

        if loai_sai_so == "a":
            target_delta_f = parse_real(input("Nhập sai số tuyệt đối yêu cầu Δf = "))
        else:
            if f_val == 0:
                print("Không định nghĩa được sai số tương đối theo |f| vì giá trị trung tâm f = 0.")
                return
            target_delta_rel_f = parse_real(input("Nhập sai số tương đối yêu cầu δf = "))
            target_delta_f = target_delta_rel_f * abs(f_val)
        if not math.isfinite(target_delta_f) or target_delta_f < 0:
            raise ValueError("Sai số mục tiêu phải không âm và hữu hạn.")

        print("\nChọn nguyên lý phân bổ sai số:")
        print("1. Nguyên lý ảnh hưởng đều (|∂f/∂xi| * Δxi = Δf / n)")
        print("2. Nguyên lý chia đều sai số tuyệt đối (Δx1 = Δx2 = ... = Δxn = Δx)")
        print("3. Nguyên lý chia đều sai số tương đối (δx1 = δx2 = ... = δxn = ε)")
        lua_chon_nguyen_ly = input("Nhập lựa chọn (1, 2 hoặc 3): ")

        print("\n" + "-" * 50)
        print(f"Ta có: f = {f}")

        # ... (Phần in biểu thức tổng quát giữ nguyên như cũ) ...

        # [Giữ nguyên nội dung khối if lua_chon_nguyen_ly == '1' và '2' ở đây...]

        if lua_chon_nguyen_ly == "3":
            print(
                "\nÁp dụng nguyên lý chia đều sai số tương đối (δx1 = δx2 = ... = ε):"
            )
            print("Có: Δxi = ε * |xi|")

            # Tính tổng các (|df/dxi| * |xi|)
            tong_he_so = sum(
                abs(dao_ham_val_dict[b]) * abs(gia_tri_dict[b]) for b in bien_list
            )

            if tong_he_so != 0:
                epsilon = target_delta_f / tong_he_so
                print(
                    f"=> ε = Δf / Σ(|∂f/∂xi|*|xi|) = {target_delta_f:.6g} / {tong_he_so:.6g} < {epsilon:.5g}\n"
                )

                print(
                    f"KẾT LUẬN: Dữ liệu đầu vào cho phép sai lệch tối đa ε = {epsilon * 100:.5g}%\n"
                )

                print("Chi tiết sai số tuyệt đối tương ứng:")
                for bien in bien_list:
                    delta_x = epsilon * abs(gia_tri_dict[bien])
                    print(f" +) Δ{bien} = ε × |{bien}| < {delta_x:.5g}")
            else:
                print("Cảnh báo: Tổng hệ số bằng 0, không thể tính toán!")

        print("\n" + "-" * 50)
        print(f"Ta có: f = {f}")

        # In biểu thức sai số tổng quát mô phỏng đúng phần khoanh đỏ ở Ảnh 1
        if loai_sai_so == "b":
            terms = []
            for bien in bien_list:
                ratio = sp.simplify(dao_ham_dict[bien] / f)
                terms.append(f"({ratio})Δ{bien}")
            print(f"=> Δf/f ≤ {' + '.join(terms)} ≤ {target_delta_rel_f}")
        else:
            terms = []
            for bien in bien_list:
                terms.append(f"({dao_ham_dict[bien]})Δ{bien}")
            print(f"=> Δf ≤ {' + '.join(terms)} ≤ {target_delta_f}")

        if lua_chon_nguyen_ly == "1":
            print("\nÁp dụng nguyên lý ảnh hưởng đều thứ nhất:")
            print("Có:")
            for bien in bien_list:
                df = dao_ham_dict[bien]
                df_val = dao_ham_val_dict[bien]

                if loai_sai_so == "a":  # Trình bày y hệt Ảnh 2
                    print(f" +) ∂f/∂{bien} = {df} = {abs(df_val):.5g}")
                    if df_val == 0:
                        print(" => Cảnh báo: Đạo hàm bằng 0!\n")
                        continue
                    delta_x = target_delta_f / (n * abs(df_val))
                    print(
                        f" => Δ{bien} = {target_delta_f:.4g} / ({n} × {abs(df_val):.5g}) < {delta_x:.4g}\n"
                    )
                else:  # Bài toán sai số tương đối (Trình bày y hệt Ảnh 1)
                    ratio_sym = sp.simplify(df / f)
                    ratio_val = abs(df_val / f_val)
                    print(f" +) (∂f/∂{bien})/f = {ratio_sym} = {ratio_val:.5g}")
                    if ratio_val == 0:
                        print(" => Cảnh báo: Đạo hàm bằng 0!\n")
                        continue
                    delta_x = target_delta_rel_f / (n * ratio_val)
                    print(
                        f" => Δ{bien} = {target_delta_rel_f:.4g} / ({n} × {ratio_val:.5g}) < {delta_x:.4g}"
                    )

                    # Tính luôn % sai lệch đầu vào (giải quyết câu hỏi ở Ảnh 1: "dữ liệu đầu vào cho phép sai lệch bao nhiêu %")
                    if gia_tri_dict[bien] != 0:
                        delta_rel_x = delta_x / abs(gia_tri_dict[bien])
                        print(
                            f"    (Sai lệch đầu vào: δ{bien} = Δ{bien}/{bien} ≤ {delta_rel_x * 100:.4g}%)\n"
                        )

        elif lua_chon_nguyen_ly == "2":
            print(
                "\nÁp dụng nguyên lý chia đều sai số tuyệt đối (Δx1 = Δx2 = ... = Δx):"
            )
            print("Có:")
            tong_dao_ham = sum(abs(dao_ham_val_dict[b]) for b in bien_list)
            print(f"Tổng các |∂f/∂xi| = {tong_dao_ham:.5g}")

            if tong_dao_ham != 0:
                delta_x_chung = target_delta_f / tong_dao_ham
                print(
                    f"=> Δx = {target_delta_f:.4g} / {tong_dao_ham:.5g} < {delta_x_chung:.4g}\n"
                )

                for bien in bien_list:
                    print(f" +) Δ{bien} < {delta_x_chung:.4g}")
                    if gia_tri_dict[bien] != 0:
                        delta_rel_x = delta_x_chung / abs(gia_tri_dict[bien])
                        print(
                            f"    (Sai lệch đầu vào: δ{bien} = Δ{bien}/{bien} ≤ {delta_rel_x * 100:.4g}%)"
                        )
                    print()
            else:
                print("Cảnh báo: Tổng các đạo hàm riêng bằng 0!")


if __name__ == "__main__":
    try:
        chuong_trinh_sai_so()
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc chương trình.")
    except (ValueError, TypeError, ZeroDivisionError) as error:
        print(f"\nKhông thể thực hiện: {error}")
