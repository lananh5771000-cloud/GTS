#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

// ================== HÀM TÍNH CHUAN MA TRAN ==================
// Chuan vô cùng:
// ||A||8 = max tung tri tuyet doi trên tung hàng

double chuanVoCungMaTran(const vector<vector<double>>& A) {
    int n = A.size();
    double maxSum = 0;

    for (int i = 0; i < n; i++) {
        double sum = 0;

        for (int j = 0; j < n; j++) {
            sum += abs(A[i][j]);
        }

        if (sum > maxSum) {
            maxSum = sum;
        }
    }

    return maxSum;
}

// ================== HÀM TÍNH CHUAN VECTOR ==================
// ||x||8 = phan tu có tri tuyet doi lon nhat

double chuanVoCungVector(const vector<double>& v) {
    double maxVal = 0;

    for (double x : v) {
        if (abs(x) > maxVal) {
            maxVal = abs(x);
        }
    }

    return maxVal;
}

// ================== NHÂN MA TRAN VOI VECTOR ==================

vector<double> nhanMaTranVector(
    const vector<vector<double>>& A,
    const vector<double>& x
) {
    int n = A.size();

    vector<double> result(n, 0);

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {
            result[i] += A[i][j] * x[j];
        }
    }

    return result;
}

// ================== CONG 2 VECTOR ==================

vector<double> congVector(
    const vector<double>& a,
    const vector<double>& b
) {
    int n = a.size();

    vector<double> result(n);

    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }

    return result;
}

// ================== TRU 2 VECTOR ==================

vector<double> truVector(
    const vector<double>& a,
    const vector<double>& b
) {
    int n = a.size();

    vector<double> result(n);

    for (int i = 0; i < n; i++) {
        result[i] = a[i] - b[i];
    }

    return result;
}

// ================== IN VECTOR ==================

void inVector(const vector<double>& v) {

    cout << "[ ";

    for (double x : v) {
        cout << fixed << setprecision(6)
             << setw(12) << x << " ";
    }

    cout << "]";
}

// ================== KIEM TRA MA TRAN VUÔNG ==================

bool kiemTraMaTranVuong(const vector<vector<double>>& A) {

    int n = A.size();

    for (int i = 0; i < n; i++) {

        if (A[i].size() != n) {
            return false;
        }
    }

    return true;
}

// ================== HÀM MAIN ==================

int main() {

    int n;

    cout << "Nhap cap cua ma tran A: ";
    cin >> n;

    // ================== NHAP DU LIEU ==================

    vector<vector<double>> A(n, vector<double>(n));

    vector<double> b(n);

    vector<double> x0(n);

    cout << "\nNhap ma tran A:\n";

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            cin >> A[i][j];
        }
    }

    cout << "\nNhap vector b:\n";

    for (int i = 0; i < n; i++) {
        cin >> b[i];
    }

    cout << "\nNhap xap xi ban dau x0:\n";

    for (int i = 0; i < n; i++) {
        cin >> x0[i];
    }

    // ================== KIEM TRA Du LIEU ==================

    if (!kiemTraMaTranVuong(A)) {

        cout << "\nMa tran A khong vuong!\n";

        return 0;
    }

    // ================== XÂY DUNG ALPHA VÀ BETA ==================
    // Ax = b
    // <=> x = (I - A)x + b

    vector<vector<double>> alpha(n, vector<double>(n));

    vector<double> beta(n);

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            if (i == j) {
                alpha[i][j] = 1.0 - A[i][j];
            }
            else {
                alpha[i][j] = -A[i][j];
            }
        }

        beta[i] = b[i];
    }

    // ================== TÍNH q ==================

    double q = chuanVoCungMaTran(alpha);

    cout << "\n===== KIEM TRA DIEU KIEN HOI TU =====\n";

    cout << "q = ||alpha|| = " << q << endl;

    if (q >= 1.0) {

        cout << "\nPhuong phap lap don KHONG hoi tu!\n";

        return 0;
    }

    cout << "Phuong phap hoi tu.\n";

    // ================== CHoN ÐIEU KIEN DUNG ==================

    int luaChon;

    double epsilon;

    int soLanLap;

    cout << "\n===== LUA CHON DIEU KIEN DUNG =====\n";

    cout << "1. Dung theo sai so epsilon\n";
    cout << "2. Dung theo so lan lap\n";

    cout << "Nhap lua chon: ";
    cin >> luaChon;

    if (luaChon == 1) {

        cout << "Nhap epsilon: ";
        cin >> epsilon;
    }
    else {

        cout << "Nhap so lan lap: ";
        cin >> soLanLap;
    }

    // ================== BAT ÐAU LAP ==================

    vector<double> xCu = x0;

    vector<double> xMoi(n);

    const int MAX_ITER = 10000;

    int k = 0;

    cout << "\n===== QUA TRINH LAP =====\n";

    cout << "Buoc 0: ";
    inVector(xCu);
    cout << endl;

    while (true) {

        k++;

        // x(k+1) = alpha*x(k) + beta

        xMoi = congVector(
                    nhanMaTranVector(alpha, xCu),
                    beta
                );

        // Tính sai s? h?u nghi?m

        vector<double> diff = truVector(xMoi, xCu);

        double saiSo =
            (q / (1.0 - q))
            * chuanVoCungVector(diff);

        // In bu?c l?p

        cout << "Buoc "
             << setw(3) << k
             << ": ";

        inVector(xMoi);

        cout << "   Sai so = "
             << scientific
             << setprecision(6)
             << saiSo
             << endl;

        // ================== ÐI?U KI?N D?NG ==================

        if (luaChon == 1) {

            if (saiSo < epsilon) {
                break;
            }
        }
        else {

            if (k >= soLanLap) {
                break;
            }
        }

        // Ch?ng l?p vô h?n

        if (k >= MAX_ITER) {

            cout << "\nVuot qua so lan lap toi da!\n";

            return 0;
        }

        // C?p nh?t nghi?m

        xCu = xMoi;
    }

    // ================== K?T LU?N ==================

    cout << "\n===== KET QUA CUOI CUNG =====\n";

    cout << "So lan lap: " << k << endl;

    cout << "\nNghiem gan dung:\n";

    for (int i = 0; i < n; i++) {

        cout << "x[" << i << "] = "
             << fixed
             << setprecision(10)
             << xMoi[i]
             << endl;
    }

    return 0;
}
