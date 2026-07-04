#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
using namespace std;

// LUU Y: code nay chi dung cho cheo troi da o tren duong cheo chinh
// nen can doi hang/cot tuy chinh ms dung dc. ;))

void printMatrix(const vector<vector<double>>& A, const string& name) {
    cout << "\n===== " << name << " =====\n";
    for (auto row : A) {
        for (double x : row)
            cout << setw(15) << fixed << setprecision(8) << x;
        cout << endl;
    }
}

void printVector(const vector<double>& v, const string& name) {
    cout << "\n===== " << name << " =====\n";
    for (double x : v)
        cout << setw(15) << fixed << setprecision(10) << x;
    cout << endl;
}

// Kiem tra cheo troi hang
bool rowDominant(const vector<vector<double>>& A) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < n; j++)
            if (i != j) s += fabs(A[i][j]);
        if (fabs(A[i][i]) <= s) return false;
    }
    return true;
}

// kiem tra cheo troi cot
bool colDominant(const vector<vector<double>>& A) {
    int n = A.size();
    for (int j = 0; j < n; j++) {
        double s = 0;
        for (int i = 0; i < n; i++)
            if (i != j) s += fabs(A[i][j]);
        if (fabs(A[j][j]) <= s) return false;
    }
    return true;
}

double normInfVector(const vector<double>& v) {
    double mx = 0;
    for (double x : v) mx = max(mx, fabs(x));
    return mx;
}

double norm1Vector(const vector<double>& v) {
    double s = 0;
    for (double x : v) s += fabs(x);
    return s;
}

vector<double> multiply(const vector<vector<double>>& A,
                        const vector<double>& x) {
    int n = A.size();
    vector<double> res(n, 0);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            res[i] += A[i][j] * x[j];

    return res;
}

int main() {
    int n;

    cout << "Nhap n = ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    cout << "\nNhap ma tran A:\n";
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> A[i][j];

    cout << "\nNhap vector b:\n";
    for (int i = 0; i < n; i++)
        cin >> b[i];

    vector<double> x(n);

    cout << "\nNhap xap xi dau x(0):\n";
    for (int i = 0; i < n; i++)
        cin >> x[i];

    double eps;
    cout << "\nNhap epsilon = ";
    cin >> eps;

    vector<vector<double>> D(n, vector<double>(n, 0));
    vector<vector<double>> Dinv(n, vector<double>(n, 0));

    for (int i = 0; i < n; i++) {
        D[i][i] = A[i][i];
        Dinv[i][i] = 1.0 / A[i][i];
    }

    printMatrix(D, "Ma tran D");
    printMatrix(Dinv, "Ma tran D^-1");

    vector<vector<double>> alpha(n, vector<double>(n, 0));
    vector<double> beta(n);

    double q = 0;
    bool useInfNorm = true;

    if (rowDominant(A)) {
        cout << "\nMa tran cheo troi hang.\n";

        for (int i = 0; i < n; i++) {
            beta[i] = b[i] / A[i][i];

            for (int j = 0; j < n; j++) {
                if (i != j)
                    alpha[i][j] = -A[i][j] / A[i][i];
            }
        }

        printMatrix(alpha, "Alpha = I - D^-1A");
        printVector(beta, "Beta = D^-1b");

        cout << "\nTinh q = ||alpha||inf\n";
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++)
                sum += fabs(alpha[i][j]);

            cout << "Hang " << i + 1 << " = " << sum << endl;
            q = max(q, sum);
        }
    }
    else if (colDominant(A)) {
        cout << "\nMa tran cheo troi cot.\n";
        useInfNorm = false;

        for (int i = 0; i < n; i++)
            beta[i] = b[i] / A[i][i];

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (i != j)
                    alpha[i][j] = -A[i][j] / A[j][j];

        printMatrix(alpha, "Alpha = I - AD^-1");
        printVector(beta, "Beta = D^-1b");

        cout << "\nTinh q = ||alpha||1\n";
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int i = 0; i < n; i++)
                sum += fabs(alpha[i][j]);

            cout << "Cot " << j + 1 << " = " << sum << endl;
            q = max(q, sum);
        }
    }
    else {
        cout << "\nKhong dam bao hoi tu Jacobi.\n";
        return 0;
    }

    cout << "\nq = " << q << endl;

    if (q >= 1) {
        cout << "\nq >= 1 => Khong dam bao hoi tu.\n";
        return 0;
    }

    vector<vector<double>> history;
    vector<double> errors;

    int k = 0;

    while (true) {
        vector<double> xNew = multiply(alpha, x);

        for (int i = 0; i < n; i++)
            xNew[i] += beta[i];

        vector<double> diff(n);

        for (int i = 0; i < n; i++)
            diff[i] = xNew[i] - x[i];

        double error;

        if (useInfNorm)
            error = q / (1 - q) * normInfVector(diff);
        else
            error = q / (1 - q) * norm1Vector(diff);

        history.push_back(xNew);
        errors.push_back(error);

        k++;

        if (error < eps) {
            x = xNew;
            break;
        }

        x = xNew;
    }

    cout << "\n============================";
    cout << "\n5 VONG LAP DAU";
    cout << "\n============================\n";

    int first = min(5, (int)history.size());

    for (int i = 0; i < first; i++) {
        cout << "\nLan lap " << i + 1 << ":\n";
        for (double v : history[i])
            cout << setw(15) << fixed << setprecision(10) << v;
        cout << "\nSai so = " << errors[i] << endl;
    }

    cout << "\n============================";
    cout << "\n3 VONG LAP CUOI";
    cout << "\n============================\n";

    int start = max(0, (int)history.size() - 3);

    for (int i = start; i < (int)history.size(); i++) {
        cout << "\nLan lap " << i + 1 << ":\n";
        for (double v : history[i])
            cout << setw(15) << fixed << setprecision(10) << v;
        cout << "\nSai so = " << errors[i] << endl;
    }

    cout << "\n============================";
    cout << "\nNGHIEM CUOI";
    cout << "\n============================\n";

    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << x[i] << endl;

    cout << "\nSo lan lap: " << k << endl;
    cout << "Sai so hau nghiem: " << errors.back() << endl;

    vector<double> residual(n, 0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            residual[i] += A[i][j] * x[j];

        residual[i] -= b[i];
    }

    printVector(residual, "A*x - b");

    return 0;
}
