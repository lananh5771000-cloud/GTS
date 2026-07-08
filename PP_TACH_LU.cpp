#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>

using namespace std;

const double EPS = 1e-12; // nguong sai so de kiem tra pivot = 0

// ==========================
// Ham in ma tran
// ==========================
void inMaTran(vector<vector<double>>& M, int n, string ten)
{
    cout << "\nMa tran " << ten << ":\n";

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << setw(10)
                 << fixed
                 << setprecision(3)
                 << M[i][j] << " ";
        }

        cout << endl;
    }
}

// ==========================
// Phan tach LU chi tiet
// Tra ve false neu ma tran khong phan tach LU duoc (pivot = 0)
// ==========================
bool phanTachLU(int n,
                vector<vector<double>>& A,
                vector<vector<double>>& L,
                vector<vector<double>>& U)
{
    cout << "\n==========================";
    cout << "\nPHAN TICH LU";
    cout << "\n==========================\n";

    for (int k = 0; k < n; k++)
    {
        cout << "\n========== BUOC k = " << k + 1
             << " ==========\n";

        // ==========================
        // Tinh U[k][j]  (hang k cua U)
        // ==========================
        for (int j = k; j < n; j++)
        {
            double sum = 0;

            cout << "\nTinh U[" << k + 1
                 << "][" << j + 1 << "]\n";

            cout << "Cong thuc:\n";

            cout << "U[" << k + 1
                 << "][" << j + 1
                 << "] = A[" << k + 1
                 << "][" << j + 1
                 << "] - (";

            for (int t = 0; t < k; t++)
            {
                cout << "L[" << k + 1 << "][" << t + 1
                     << "] * U[" << t + 1
                     << "][" << j + 1 << "]";

                if (t != k - 1)
                    cout << " + ";
            }

            cout << ")\n";

            for (int t = 0; t < k; t++)
            {
                double temp = L[k][t] * U[t][j];

                cout << " = "
                     << L[k][t]
                     << " * "
                     << U[t][j]
                     << " = "
                     << temp << endl;

                sum += temp;
            }

            U[k][j] = A[k][j] - sum;

            cout << "=> U[" << k + 1
                 << "][" << j + 1
                 << "] = "
                 << A[k][j]
                 << " - "
                 << sum
                 << " = "
                 << U[k][j]
                 << endl;
        }

        // Kiem tra pivot (dung nguong EPS thay vi so sanh == 0 tuyet doi)
        if (fabs(U[k][k]) < EPS)
        {
            cout << "\n*** LOI: U[" << k + 1 << "][" << k + 1
                 << "] = " << U[k][k]
                 << " ~ 0 => Khong the phan tich LU (dinh thuc con chinh cap "
                 << k + 1 << " bang 0)! ***\n";
            return false; // bao that bai, de main() dung chuong trinh
        }

        // Duong cheo L
        L[k][k] = 1;

        cout << "\nDat L[" << k + 1
             << "][" << k + 1
             << "] = 1\n";

        // ==========================
        // Tinh L[i][k]  (cot k cua L)
        // ==========================
        for (int i = k + 1; i < n; i++)
        {
            double sum = 0;

            cout << "\nTinh L[" << i + 1
                 << "][" << k + 1 << "]\n";

            cout << "Cong thuc:\n";

            cout << "L[" << i + 1
                 << "][" << k + 1
                 << "] = (A[" << i + 1
                 << "][" << k + 1
                 << "] - (";

            for (int t = 0; t < k; t++)
            {
                cout << "L[" << i + 1
                     << "][" << t + 1
                     << "] * U[" << t + 1
                     << "][" << k + 1 << "]";

                if (t != k - 1)
                    cout << " + ";
            }

            cout << ")) / U[" << k + 1
                 << "][" << k + 1 << "]\n";

            for (int t = 0; t < k; t++)
            {
                double temp = L[i][t] * U[t][k];

                cout << " = "
                     << L[i][t]
                     << " * "
                     << U[t][k]
                     << " = "
                     << temp << endl;

                sum += temp;
            }

            L[i][k] =
                (A[i][k] - sum) / U[k][k];

            cout << "=> L[" << i + 1
                 << "][" << k + 1
                 << "] = ("
                 << A[i][k]
                 << " - "
                 << sum
                 << ") / "
                 << U[k][k]
                 << " = "
                 << L[i][k]
                 << endl;
        }

        inMaTran(L, n, "L hien tai");
        inMaTran(U, n, "U hien tai");
    }

    return true; // phan tach thanh cong
}

// ==========================
// Giai LY = B
// ==========================
void giaiLY(int n,
            vector<vector<double>>& L,
            vector<double>& B,
            vector<double>& Y)
{
    cout << "\n==========================";
    cout << "\nGIAI LY = B";
    cout << "\n==========================\n";

    for (int i = 0; i < n; i++)
    {
        double sum = 0;

        cout << "\nTinh Y[" << i + 1 << "]\n";

        cout << "Y[" << i + 1
             << "] = B[" << i + 1
             << "] - (";

        for (int j = 0; j < i; j++)
        {
            cout << "L[" << i + 1
                 << "][" << j + 1
                 << "] * Y[" << j + 1 << "]";

            if (j != i - 1)
                cout << " + ";
        }

        cout << ")\n";

        for (int j = 0; j < i; j++)
        {
            double temp = L[i][j] * Y[j];

            cout << " = "
                 << L[i][j]
                 << " * "
                 << Y[j]
                 << " = "
                 << temp << endl;

            sum += temp;
        }

        Y[i] = B[i] - sum;

        cout << "=> Y[" << i + 1
             << "] = "
             << B[i]
             << " - "
             << sum
             << " = "
             << Y[i]
             << endl;
    }
}

// ==========================
// Giai UX = Y
// ==========================
void giaiUX(int n,
            vector<vector<double>>& U,
            vector<double>& Y,
            vector<double>& X)
{
    cout << "\n==========================";
    cout << "\nGIAI UX = Y";
    cout << "\n==========================\n";

    for (int i = n - 1; i >= 0; i--)
    {
        double sum = 0;

        cout << "\nTinh X[" << i + 1 << "]\n";

        cout << "X[" << i + 1
             << "] = (Y[" << i + 1
             << "] - (";

        for (int j = i + 1; j < n; j++)
        {
            cout << "U[" << i + 1
                 << "][" << j + 1
                 << "] * X[" << j + 1 << "]";

            if (j != n - 1)
                cout << " + ";
        }

        cout << ")) / U[" << i + 1
             << "][" << i + 1 << "]\n";

        for (int j = i + 1; j < n; j++)
        {
            double temp = U[i][j] * X[j];

            cout << " = "
                 << U[i][j]
                 << " * "
                 << X[j]
                 << " = "
                 << temp << endl;

            sum += temp;
        }

        X[i] = (Y[i] - sum) / U[i][i];

        cout << "=> X[" << i + 1
             << "] = ("
             << Y[i]
             << " - "
             << sum
             << ") / "
             << U[i][i]
             << " = "
             << X[i]
             << endl;
    }

    cout << "\n==========================";
    cout << "\nNGHIEM CUOI CUNG";
    cout << "\n==========================\n";

    for (int i = 0; i < n; i++)
    {
        cout << "X[" << i + 1
             << "] = "
             << X[i]
             << endl;
    }
}

// ==========================
// MAIN
// ==========================
int main()
{
    int n;

    cout << "Nhap n = ";
    cin >> n;

    if (n <= 0)
    {
        cout << "\nLoi: n phai la so nguyen duong!\n";
        return 1;
    }

    vector<vector<double>> A(n,
        vector<double>(n));

    vector<vector<double>> L(n,
        vector<double>(n, 0));

    vector<vector<double>> U(n,
        vector<double>(n, 0));

    vector<double> B(n);
    vector<double> Y(n);
    vector<double> X(n);

    // Nhap A
    cout << "\nNhap ma tran A:\n";

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cin >> A[i][j];
        }
    }

    // Nhap B
    cout << "\nNhap vector B:\n";

    for (int i = 0; i < n; i++)
    {
        cin >> B[i];
    }

    // Phan tach LU
    bool ok = phanTachLU(n, A, L, U);

    if (!ok)
    {
        // Neu khong phan tach duoc thi DUNG chuong trinh o day,
        // khong duoc tiep tuc giai LY = B va UX = Y vi se chia cho 0
        cout << "\nChuong trinh dung lai vi khong phan tich duoc LU.\n";
        return 1;
    }

    // In ket qua LU
    inMaTran(L, n, "L");
    inMaTran(U, n, "U");

    // Giai LY = B
    giaiLY(n, L, B, Y);

    // Giai UX = Y
    giaiUX(n, U, Y, X);

    return 0;
}
