#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>

using namespace std;

// ========================================================================
// NHAP CAC HAM PHI TUYEN
// ========================================================================
void setup_phi_functions(vector<function<double(const vector<double>&)>>& phi_funcs) {
    // VIDU 2 phuong trình:
   
    
    phi_funcs.push_back([](const vector<double>& x) {
        return sqrt (6-2*x[1]*x[1]);
    });
    
    phi_funcs.push_back([](const vector<double>& x) {
        return (4-5*x[1]*x[1])/x[0];
    });
    
   
}

// Hàm tính d?o hàm riêng x?p x? b?ng phuong pháp sai phân trung tâm
double partial_derivative(function<double(const vector<double>&)> f, vector<double> x, int j) {
    double h = 1e-5;
    double temp = x[j];
    x[j] = temp + h;
    double f_plus = f(x);
    x[j] = temp - h;
    double f_minus = f(x);
    return (f_plus - f_minus) / (2.0 * h);
}

int main() {
    try {
        cout << "=== GIAI HE PHUONG TRINH BANG PHUONG PHAP LAP DON ===" << endl;
        
        vector<function<double(const vector<double>&)>> phi_funcs;
        setup_phi_functions(phi_funcs);
        int n = phi_funcs.size();
        cout << "So luong phuong trinh (n) da duoc thiet lap la: " << n << "\n";
        cout << "(Vui long sua ham setup_phi_functions trong source code de doi he phuong trinh)\n";

        // 0. NH?P MI?N D VÀ ÐI?M X?P X? BAN Ð?U
        cout << "\n--- BUOC 0: NHAP MIEN CACH LY NGHIEM D VA DIEM BAT DAU ---" << endl;
        vector<pair<double, double>> bounds(n);
        for (int i = 0; i < n; ++i) {
            cout << "Nhap can duoi cua x" << i + 1 << " trong D (a" << i + 1 << "): ";
            cin >> bounds[i].first;
            cout << "Nhap can tren cua x" << i + 1 << " trong D (b" << i + 1 << "): ";
            cin >> bounds[i].second;
        }

        vector<double> x0_vals(n);
        cout << "\nNhap gia tri xap xi ban dau x^(0):" << endl;
        for (int i = 0; i < n; ++i) {
            cout << "x" << i + 1 << "^(0) = ";
            cin >> x0_vals[i];
            if (x0_vals[i] < bounds[i].first || x0_vals[i] > bounds[i].second) {
                cout << "[!] Canh bao: x" << i + 1 << "^(0) = " << x0_vals[i] 
                     << " khong nam trong khoang D = [" << bounds[i].first << ", " << bounds[i].second << "]." << endl;
            }
        }

        // 1. KI?M TRA ÐI?U KI?N H?I T? TRÊN MI?N D
        cout << "\n--- BUOC 1: KIEM TRA DIEU KIEN HOI TU TREN MIEN D ---" << endl;
        
        vector<vector<double>> sample_points;
        // 1.1. Các d?nh c?a siêu hình ch? nh?t D
        int num_corners = 1 << n; // 2^n
        for (int i = 0; i < num_corners; ++i) {
            vector<double> pt(n);
            for (int j = 0; j < n; ++j) {
                pt[j] = (i & (1 << j)) ? bounds[j].second : bounds[j].first;
            }
            sample_points.push_back(pt);
        }
        
        // 1.2. Thêm 100 di?m ng?u nhiên trong D
        random_device rd;
        mt19937 gen(rd());
        vector<uniform_real_distribution<double>> dists;
        for (int i = 0; i < n; ++i) {
            dists.push_back(uniform_real_distribution<double>(bounds[i].first, bounds[i].second));
        }
        for (int i = 0; i < 100; ++i) {
            vector<double> pt(n);
            for (int j = 0; j < n; ++j) {
                pt[j] = dists[j](gen);
            }
            sample_points.push_back(pt);
        }

        bool is_phi_in_D = true;
        double max_q = 0.0;
        vector<double> bad_pt;
        double bad_y = 0;
        int bad_idx = -1;

        cout << "\nDang kiem tra Phi(D) in D va tinh he so co q tren toan mien D..." << endl;
        for (const auto& pt : sample_points) {
            // Ki?m tra Phi(D) in D
            for (int i = 0; i < n; ++i) {
                double y = phi_funcs[i](pt);
                if (y < bounds[i].first || y > bounds[i].second) {
                    if (is_phi_in_D) { // Ch? luu ví d? d?u tiên b? sai
                        bad_pt = pt;
                        bad_y = y;
                        bad_idx = i + 1;
                    }
                    is_phi_in_D = false;
                }
            }

            // Tính q t?i di?m này b?ng ma tr?n Jacobi (d?o hàm s?)
            double q_local = 0.0;
            for (int i = 0; i < n; ++i) {
                double sum_row = 0.0;
                for (int j = 0; j < n; ++j) {
                    sum_row += abs(partial_derivative(phi_funcs[i], pt, j));
                }
                if (sum_row > q_local) q_local = sum_row;
            }
            if (q_local > max_q) max_q = q_local;
        }

        cout << fixed << setprecision(6);
        cout << "-> Danh gia he so co lon nhat tren D: q = " << max_q << endl;
        if (is_phi_in_D) {
            cout << "-> Danh gia: Thoa man dieu kien Phi(D) in D (dua tren lay mau)." << endl;
        } else {
            cout << "-> [!] KHONG THOA MAN: Ton tai diem trong D ma Phi(x) vang ra khoi D." << endl;
            cout << "   Vi du: Tai x = (";
            for (int i = 0; i < n; ++i) cout << bad_pt[i] << (i == n-1 ? "" : ", ");
            cout << "), ham phi_" << bad_idx << " = " << bad_y << " khong thuoc khoang." << endl;
        }

        if (max_q < 1.0 && is_phi_in_D) {
            cout << "\n=> KET LUAN HOI TU: He DAM BAO hoi tu duy nhat tren mien D." << endl;
        } else {
            cout << "\n=> CANH BAO: He KHONG DAM BAO hoi tu. Qua trinh lap co the phan ky." << endl;
        }

        // 2. CH?N ÐI?U KI?N D?NG
        cout << "\n--- BUOC 2: CHON DIEU KIEN DUNG ---" << endl;
        cout << "1. Co sai so muc tieu (cho truoc epsilon)" << endl;
        cout << "2. Khong cho sai so, chi cho so lan lap (danh gia hau nghiem)" << endl;
        cout << "3. Khong cho sai so, nhap so chu so thap phan tin cay" << endl;
        int choice;
        cout << "Nhap lua chon (1/2/3): ";
        cin >> choice;

        double epsilon = 0.0;
        int max_iter = 1000;
        int stop_type = 0; // 1: Tuyet doi, 2: Tuong doi, 3: So lan lap
        string error_formula = "";
        double q = max_q;

        if (choice == 1) {
            cout << "\nChon loai sai so:\n1. Sai so tuyet doi\n2. Sai so tuong doi\nLua chon (1/2): ";
            int err_choice;
            cin >> err_choice;
            cout << "Nhap sai so muc tieu epsilon = ";
            cin >> epsilon;
            if (err_choice == 1) {
                stop_type = 1;
                error_formula = "||x^(k) - x^(k-1)||_\u221E";
            } else {
                stop_type = 2;
                error_formula = "(||x^(k) - x^(k-1)||_\u221E) / (||x^(k)||_\u221E)";
            }
        } else if (choice == 2) {
            cout << "Nhap so lan lap: ";
            cin >> max_iter;
            stop_type = 3;
            if (q >= 1) {
                cout << "\n[!] LOI TOAN HOC: q >= 1. Khong the dung cong thuc hau nghiem." << endl;
                error_formula = "||x^(k) - x^(k-1)||_\u221E (Hau nghiem that bai)";
            } else {
                error_formula = "Hau nghiem: (q / (1 - q)) * ||x^(k) - x^(k-1)||_\u221E";
            }
        } else if (choice == 3) {
            int d;
            cout << "Nhap so chu so thap phan tin cay: ";
            cin >> d;
            epsilon = 0.5 * pow(10, -d);
            cout << "-> Sai so muc tieu tuong duong epsilon = " << scientific << epsilon << fixed << endl;
            stop_type = 1;
            error_formula = "||x^(k) - x^(k-1)||_\u221E";
        } else {
            throw invalid_argument("Lua chon khong hop le!");
        }

        // 3. QUÁ TRÌNH L?P
        cout << "\n--- BUOC 3: QUA TRINH LAP ---" << endl;
        cout << "Cong thuc tinh sai so: \u0394 = " << error_formula << "\n\n";

        // In Header
        int col_width_k = 5;
        int col_width_x = 18;
        int col_width_err = 25;
        
        cout << left << setw(col_width_k) << "k";
        for (int i = 0; i < n; ++i) cout << left << setw(col_width_x) << ("x" + to_string(i+1));
        cout << left << setw(col_width_err) << "Sai so \u0394" << endl;
        
        int total_width = col_width_k + col_width_x * n + col_width_err;
        cout << string(total_width, '-') << endl;

        vector<double> x_curr = x0_vals;
        int k = 1;

        while (true) {
            vector<double> x_next(n);
            double diff_norm = 0.0;
            double x_next_norm = 0.0;

            for (int i = 0; i < n; ++i) {
                x_next[i] = phi_funcs[i](x_curr);
                diff_norm = max(diff_norm, abs(x_next[i] - x_curr[i]));
                x_next_norm = max(x_next_norm, abs(x_next[i]));
            }

            double current_error = 0.0;
            if (stop_type == 1) {
                current_error = diff_norm;
            } else if (stop_type == 2) {
                current_error = (x_next_norm != 0) ? diff_norm / x_next_norm : diff_norm;
            } else if (stop_type == 3) {
                if (q < 1) current_error = (q / (1.0 - q)) * diff_norm;
                else current_error = diff_norm;
            }

            // In Row
            cout << left << setw(col_width_k) << k;
            for (int i = 0; i < n; ++i) cout << left << setw(col_width_x) << fixed << setprecision(8) << x_next[i];
            cout << left << setw(col_width_err) << scientific << setprecision(8) << current_error << fixed << endl;

            if ((stop_type == 1 || stop_type == 2) && current_error <= epsilon) {
                cout << "\n=> Thoa man dieu kien dung sau " << k << " buoc lap." << endl;
                break;
            }
            if (stop_type == 3 && k >= max_iter) {
                cout << "\n=> Da hoan thanh " << max_iter << " buoc lap yeu cau." << endl;
                break;
            }

            x_curr = x_next;
            k++;

            if (k > 1000 && stop_type != 3) {
                cout << "\n[!] Da dat gioi han 1000 buoc lap an toan. He co dau hieu phan ky." << endl;
                break;
            }
        }

        // 4. K?T LU?N
        cout << "\n--- KET LUAN ---" << endl;
        cout << "Nghiem xap xi cua he:" << endl;
        for (int i = 0; i < n; ++i) {
            cout << "  x" << i + 1 << " = " << fixed << setprecision(8) << x_curr[i] << endl;
        }

    } catch (const exception& e) {
        cerr << "\n[X] LOI: " << e.what() << endl;
    }

    return 0;
}
