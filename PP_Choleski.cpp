#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;

const double EPS = 1e-9; // nguong sai so

// ==========================
// Kiem tra doi xung
// ==========================
bool doiXung(vector<vector<double>>& A, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (fabs(A[i][j] - A[j][i]) > EPS)
                return false;
        }
    }

    return true;
}

// ==========================
// In ma tran
// ==========================
void inMaTran(vector<vector<double>>& A,
              int n,
              string ten)
{
    cout << "\nMa tran " << ten << ":\n";

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << setw(10)
                 << fixed
                 << setprecision(4)
                 << A[i][j] << " ";
        }

        cout << endl;
    }
}

// ==========================
// Phan tich Cholesky
// A = U^T * U
// ==========================
bool cholesky(vector<vector<double>>& A,
              vector<vector<double>>& U,
              int n)
{
    if (!doiXung(A, n))
    {
        cout << "\nMa tran khong doi xung!\n";
        return false;
    }

    cout << "\n========================";
    cout << "\nPHAN TICH CHOLESKY";
    cout << "\n========================\n";

    for (int i = 0; i < n; i++)
    {
        // ======================
        // Tinh Uii
        // ======================
        double sum = 0;

        cout << "\nTinh U[" << i + 1
             << "][" << i + 1 << "]\n";

        for (int k = 0; k < i; k++)
        {
            double temp =
                U[k][i] * U[k][i];

            cout << "("
                 << U[k][i]
                 << ")^2 = "
                 << temp << endl;

            sum += temp;
        }

        double value = A[i][i] - sum;

        cout << "=> "
             << A[i][i]
             << " - "
             << sum
             << " = "
             << value << endl;

        // Kiem tra xac dinh duong (dung nguong EPS thay vi <= 0 tuyet doi)
        if (value < EPS)
        {
            cout << "\nMa tran khong xac dinh duong!\n";
            return false;
        }

        U[i][i] = sqrt(value);

        cout << "=> U[" << i + 1
             << "][" << i + 1
             << "] = sqrt("
             << value
             << ") = "
             << U[i][i]
             << endl;

        // ======================
        // Tinh Uij
        // ======================
        for (int j = i + 1; j < n; j++)
        {
            double sum2 = 0;

            cout << "\nTinh U[" << i + 1
                 << "][" << j + 1 << "]\n";

            for (int k = 0; k < i; k++)
            {
                double temp =
                    U[k][i] * U[k][j];

                cout << U[k][i]
                     << " * "
                     << U[k][j]
                     << " = "
                     << temp << endl;

                sum2 += temp;
            }

            U[i][j] =
                (A[i][j] - sum2) / U[i][i];

            cout << "=> U[" << i + 1
                 << "][" << j + 1
                 << "] = ("
                 << A[i][j]
                 << " - "
                 << sum2
                 << ") / "
                 << U[i][i]
                 << " = "
                 << U[i][j]
                 << endl;
        }

        inMaTran(U, n, "U hien tai");
    }

    return true;
}

// ==========================
// Giai U^T * Y = B
// ==========================
void giaiUTY(int n,
             vector<vector<double>>& U,
             vector<double>& B,
             vector<double>& Y)
{
    cout << "\n========================";
    cout << "\nGIAI U^T * Y = B";
    cout << "\n========================\n";

    for (int i = 0; i < n; i++)
    {
        double sum = 0;

        for (int j = 0; j < i; j++)
        {
            double temp =
                U[j][i] * Y[j];

            cout << U[j][i]
                 << " * "
                 << Y[j]
                 << " = "
                 << temp << endl;

            sum += temp;
        }

        Y[i] = (B[i] - sum) / U[i][i];

        cout << "=> Y[" << i + 1
             << "] = ("
             << B[i]
             << " - "
             << sum
             << ") / "
             << U[i][i]
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
    cout << "\n========================";
    cout << "\nGIAI UX = Y";
    cout << "\n========================\n";

    for (int i = n - 1; i >= 0; i--)
    {
        double sum = 0;

        for (int j = i + 1; j < n; j++)
        {
            double temp =
                U[i][j] * X[j];

            cout << U[i][j]
                 << " * "
                 << X[j]
                 << " = "
                 << temp << endl;

            sum += temp;
        }

        X[i] =
            (Y[i] - sum) / U[i][i];

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

    vector<vector<double>> A(
        n,
        vector<double>(n));

    vector<vector<double>> U(
        n,
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

    // Cholesky
    bool ok =
        cholesky(A, U, n);

    if (!ok)
    {
        cout << "\nChuong trinh dung lai vi khong phan tich duoc Cholesky.\n";
        return 1;
    }

    // In U
    inMaTran(U, n, "U cuoi");

    // Giai U^T Y = B
    giaiUTY(n, U, B, Y);

    // Giai UX = Y
    giaiUX(n, U, Y, X);

    // In nghiem
    cout << "\n========================";
    cout << "\nNGHIEM CUOI";
    cout << "\n========================\n";

    for (int i = 0; i < n; i++)
    {
        cout << "X[" << i + 1
             << "] = "
             << X[i]
             << endl;
    }

    return 0;
}
