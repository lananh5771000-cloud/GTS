# Báo cáo nghiệm thu project Giải tích số

Ngày kiểm tra: 08-07-2026. Môi trường: Windows, Python 3.13.14, NumPy
2.4.6, SymPy 1.14.0, mpmath 1.3.0.

## 1. Tệp đã sửa

- Nhóm phương trình một biến: `python/chiadoi_pt.py`, `python/daycung.py`,
  `python/tieptuyen.py`, `python/lapdon_pt.py`.
- Nhóm hệ phi tuyến: `python/hptlapdonphituyen.py`,
  `python/newton_he_phi_tuyen.py`.
- Nhóm hệ tuyến tính: `python/gauss (2).py`, `python/gaussrank.py`,
  `python/phantach_lu.py`, `python/cholesky.py`, `python/lapdon_tuyentinh.py`,
  `python/lapjacobituyentinh.py`, `python/seideltuyentinh.py`.
- Nhóm nghịch đảo: `python/nghichdao_newton.py`,
  `python/nghichdao_vienquanh.py`.
- Nhóm trị riêng/SVD: `python/danilevski_tri_rieng.py`,
  `python/tri_rieng_troi_xuong_thang.py`, `python/khai_trien_ky_di_svd.py`.
- Nhóm đa thức/sai số: `python/phuong_phap_da_thuc.py`,
  `python/saiso (1).py`.
- Test cũ được tăng cường: `tests/test_numerical_safety.py`,
  `tests/test_second_round.py`.

## 2. Tệp hạ tầng và test mới trong bản project hiện tại

- Dùng chung: `python/input_utils.py`, `python/exam_format.py`.
- Chạy offline: `requirements.txt`, `pytest.ini`, `KIEM_TRA_MAY.py`,
  `KIEM_TRA_MAY.bat`, `CHAY_TEST.bat`, `CHAY_KIEM_THU.bat`,
  `HUONG_DAN_NHANH.txt`.
- Wrapper tương thích ở thư mục gốc: `gauss (2).py`, `saiso (1).py`.
- Test: `tests/test_input_utils.py`, `tests/test_exam_format_and_edges.py`,
  `tests/test_acceptance_seidel_direct.py`,
  `tests/test_acceptance_svd_deflation.py`,
  `tests/test_acceptance_bordering.py`,
  `tests/test_randomized_acceptance.py`.

Hai tệp nén `python.zip` và `python (2).zip` đã có trong worktree, không tham
gia import, chạy chương trình hay kiểm thử.

## 3. Menu và dạng input chính

- Chia đôi, dây cung, Newton một biến, lặp đơn một biến: nhập biểu thức an
  toàn, khoảng, điểm đầu; chọn sai số tuyệt đối/tương đối, số chữ số hoặc
  đúng k bước khi phương pháp hỗ trợ.
- Hệ lặp phi tuyến: nhập `F(X)=0` hoặc trực tiếp `X=Φ(X)`; miền hộp, chuẩn,
  điểm đầu và chế độ dừng. Newton hệ nhận `F`, tự lập Jacobian, epsilon hoặc k.
- Gauss/Gauss-rank: giải `AX=B` (kể cả nhiều vế phải) hoặc tìm nghịch đảo;
  xử lý hạng và nghiệm tổng quát.
- LU/PLU: chỉ phân tích, phân tích và giải nhiều vế phải, hoặc nghịch đảo.
- Cholesky: phân tích `A=UᵀU`, giải `AX=B`, hoặc nghịch đảo.
- Lặp đơn tuyến tính: tự chọn τ cho SPD, tự nhập τ, hoặc nhập thẳng `B,d`;
  chọn chuẩn và dừng theo chặn hoặc đúng k.
- Jacobi: giải nhiều vế phải/tìm nghịch đảo; tự đổi hàng; dừng hậu nghiệm,
  tiên nghiệm, đúng k hoặc số chữ số.
- Seidel: (1) `Ax=b`; (2) trực tiếp `x=Bx+d`; (3) tìm `A⁻¹` bằng `AX=I`;
  (4) nhiều vế phải. Nhánh 2 không hỏi A,b và không đổi dạng bài.
- Newton–Schulz: tự chọn/tự nhập `X⁽⁰⁾`; dừng hậu nghiệm, tiên nghiệm, số chữ
  số hoặc đúng k; kiểm tra residual trái/phải.
- Viền quanh: in đầy đủ, bản chép thi, hoặc chỉ kết quả; tự tìm hoán vị hàng
  và cột, theo dõi P,Q và khôi phục thứ tự ban đầu.
- Danilevski: đa thức đặc trưng, trị riêng/vector riêng và kiểm tra
  Cayley–Hamilton/residual.
- Lũy thừa–xuống thang: tự nhập vector đầu hoặc dùng vector mặc định; đúng k
  hoặc đến epsilon; kiểm tra residual trên A gốc.
- SVD: rút gọn (mặc định), đầy đủ, hoặc dạng mỏng `min(m,n)` kể cả sigma gần
  0; đúng k hoặc đến epsilon.
- Đa thức: hệ số nguyên/phân số/thập phân/căn; Sturm, chia đôi, Newton hiệu
  chỉnh và nghiệm bội; bảng gọn hoặc đầy đủ.
- Sai số: bài thuận/ngược, sai số tuyệt đối/tương đối, ba nguyên lý phân bổ.

Bộ nhập chung nhận số nguyên, thập phân dấu chấm hoặc dấu phẩy không nhập
nhằng, phân số, ký hiệu khoa học, căn, pi/e và biểu thức thông dụng; từ chối
NaN, vô cực, chia 0, mã lệnh và sai kích thước. Các trình đọc ma trận bắt nhập
lại riêng hàng sai. Không dùng `eval`.

## 4. Kết quả bắt buộc

### Seidel trực tiếp `x=Bx+d`, k=5

`x⁽⁵⁾ = (-6.4318976378, 6.7240093239, -4.3175380558, 2.5265020004,
-6.2308538428)ᵀ`.

Residual điểm bất động:
`‖x⁽⁵⁾-Bx⁽⁵⁾-d‖∞ = 2.579280942783e-05`.

Seidel trên `(I-B)x=d` sau 5 bước khác kết quả trên
`2.447683031193e-05` theo chuẩn vô cùng; test chống nhầm hai thuật toán đã qua.

### SVD mẫu 4×3

- `σ = (22.6482521856, 6.2872632819, 2.3509558402)`.
- `Uᵣ: 4×3`, `Σᵣ: 3×3`, `Vᵣᵀ: 3×3`.
- In và lưu đúng bốn ma trận `B₀,B₁,B₂,B₃`; mỗi bước thỏa công thức Hotelling.
- Sai số tái tạo tương đối `3.000246493745e-16`.
- Sai số trực giao U `9.137200893943e-11`, V `2.487967497245e-16`.
- Kết luận cuối: đạt đồng thời kiểm tra bộ ba kỳ dị, trực giao và tái tạo.

### Viền quanh mẫu 6×6, a₁₁=0

Dùng ma trận hoán vị chu kỳ cấp 6. Chương trình tìm P,Q, viền trên ma trận đã
đổi thứ tự, khôi phục nghịch đảo và xác nhận chính xác cả
`AA⁻¹=I` lẫn `A⁻¹A=I`. Ca suy biến cũng trả về `None` với kết luận rõ, không
traceback.

## 5. Kết quả kiểm thử cuối

- `python -m compileall -q .`: exit code 0.
- `python -m pytest -q`: **120 passed in 9.13s**.
- Trong đó `tests/test_randomized_acceptance.py` chạy 1.550 ca có seed
  `20260708`: 200 PLU, 100 Cholesky, 150 viền quanh, 100 Newton–Schulz,
  150 Jacobi, 150 Seidel Ax=b, 150 Seidel x=Bx+d, 100 Danilevski,
  100 trị riêng, 150 SVD, 100 đa thức và 100 phương trình phi tuyến.
- `python KIEM_TRA_MAY.py`: exit code 0 và dòng cuối
  `HỆ THỐNG SẴN SÀNG CHẠY OFFLINE`.
- `python -m ruff check python tests KIEM_TRA_MAY.py`: `All checks passed!`.
- Quét chạy trực tiếp 23 module với stdin rỗng: 23/23 thoát sạch, không
  traceback.

## 6. Giới hạn còn tồn tại

- Máy hiện tại cài pytest nhưng thư mục `Python313\Scripts` không nằm trong
  PATH, nên lệnh trần `pytest -q` của PowerShell không được tìm thấy. Lõi test
  chạy đầy đủ bằng `python -m pytest -q`; hai tệp BAT cũng dùng cách này nên
  không phụ thuộc PATH và vẫn chạy offline.
- Ba mức in được hoàn thiện rõ cho viền quanh; nhiều chương trình cũ vẫn dùng
  bản chi tiết mặc định, còn một số chỉ có hai mức gọn/đầy đủ. Chưa có cùng
  một menu ba mức ở tuyệt đối mọi file.
- Độ chính xác ma trận/vector/scalar/residual đã tách ở formatter và ở các nơi
  cần kiểm tra, nhưng chưa được đưa thành bốn câu hỏi riêng trong mọi giao
  diện legacy.
- Cholesky chỉ nhận số thực và chủ động từ chối trường hợp ngoài giả thiết;
  chưa cài Cholesky Hermite cho số phức.
- Các chứng nhận dùng lấy mẫu/interval xấp xỉ luôn được ghi đúng là khảo sát,
  không được quảng bá thành chứng minh giải tích tuyệt đối.

