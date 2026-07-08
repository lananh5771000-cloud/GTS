#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

const double EPS = 1e-9;

// Hàm in ma tr?n
void printMatrix(const vector<vector<double>>& M, int n) {
    cout << fixed << setprecision(6);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << setw(12)
                 << (abs(M[i][j]) < EPS ? 0.0 : M[i][j])
                 << " ";
        }
        cout << endl;
    }
}

int main() {
    int n;

    cout << "Nhap cap cua ma tran n = ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n));

    cout << "\nNhap cac phan tu cua ma tran A:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }

    // ==========================
    // Buoc 1: In ma tran A
    // ==========================

    cout << "\n==============================";
    cout << "\nMA TRAN A";
    cout << "\n==============================\n";

    printMatrix(A, n);

    // ==========================
    // Buoc 2: Kiem tra doi xung
    // ==========================

    bool isSymmetric = true;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (abs(A[i][j] - A[j][i]) > EPS) {
                isSymmetric = false;
                break;
            }
        }

        if (!isSymmetric)
            break;
    }

    vector<vector<double>> B(n, vector<double>(n, 0.0));

    if (isSymmetric) {
        B = A;

        cout << "\n=> A la ma tran doi xung";
        cout << "\n=> B = A\n";
    }
    else {
        cout << "\n=> A KHONG doi xung";
        cout << "\n=> Chuan hoa B = A^T * A\n";

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {

                for (int k = 0; k < n; k++) {
                    B[i][j] += A[k][i] * A[k][j];
                }

                cout << "B[" << i << "][" << j << "] = "
                     << B[i][j] << endl;
            }
        }
    }

    cout << "\n==============================";
    cout << "\nMA TRAN B";
    cout << "\n==============================\n";

    printMatrix(B, n);

    // ==========================
    // Buoc 3: Cholesky
    // ==========================

    vector<vector<double>> Q(n, vector<double>(n, 0.0));

    cout << "\n==============================";
    cout << "\nPHAN TICH CHOLESKY";
    cout << "\n==============================\n";

    for (int i = 0; i < n; i++) {

        double sumDiag = 0.0;

        for (int k = 0; k < i; k++) {
            sumDiag += Q[k][i] * Q[k][i];
        }

        double temp = B[i][i] - sumDiag;

        cout << "\nQ[" << i << "][" << i << "] = sqrt("
             << B[i][i]
             << " - "
             << sumDiag
             << ")";

        if (temp < EPS) {
            cout << "\n\n[Loi] Ma tran khong xac dinh duong!";
            return 0;
        }

        Q[i][i] = sqrt(temp);

        cout << " = " << Q[i][i] << endl;

        for (int j = i + 1; j < n; j++) {

            double sumUpper = 0.0;

            for (int k = 0; k < i; k++) {
                sumUpper += Q[k][i] * Q[k][j];
            }

            Q[i][j] = (B[i][j] - sumUpper) / Q[i][i];

            cout << "Q[" << i << "][" << j << "] = ("
                 << B[i][j]
                 << " - "
                 << sumUpper
                 << ") / "
                 << Q[i][i]
                 << " = "
                 << Q[i][j]
                 << endl;
        }
    }

    cout << "\n==============================";
    cout << "\nMA TRAN Q";
    cout << "\n==============================\n";

    printMatrix(Q, n);

    // ==========================
    // Buoc 4: Tim X = Q^-1
    // ==========================

    vector<vector<double>> X(n, vector<double>(n, 0.0));

    cout << "\n==============================";
    cout << "\nTIM X = Q^-1";
    cout << "\n==============================\n";

    for (int i = n - 1; i >= 0; i--) {

        X[i][i] = 1.0 / Q[i][i];

        cout << "X[" << i << "][" << i << "] = 1/"
             << Q[i][i]
             << " = "
             << X[i][i]
             << endl;

        for (int j = i + 1; j < n; j++) {

            double sumInv = 0.0;

            for (int k = i + 1; k <= j; k++) {
                sumInv += Q[i][k] * X[k][j];
            }

            X[i][j] = -sumInv / Q[i][i];

            cout << "X[" << i << "][" << j << "] = -("
                 << sumInv
                 << ")/"
                 << Q[i][i]
                 << " = "
                 << X[i][j]
                 << endl;
        }
    }

    cout << "\n==============================";
    cout << "\nMA TRAN X = Q^-1";
    cout << "\n==============================\n";

    printMatrix(X, n);

    // ==========================
    // Buoc 5: B^-1 = X * X^T
    // ==========================

    vector<vector<double>> invB(n, vector<double>(n, 0.0));

    cout << "\n==============================";
    cout << "\nTINH B^-1 = X * X^T";
    cout << "\n==============================\n";

    for (int i = 0; i < n; i++) {

        for (int j = i; j < n; j++) {

            double sumB = 0.0;

            for (int k = j; k < n; k++) {
                sumB += X[i][k] * X[j][k];
            }

            invB[i][j] = sumB;
            invB[j][i] = sumB;

            cout << "B^-1[" << i << "][" << j
                 << "] = "
                 << sumB
                 << endl;
        }
    }

    cout << "\n==============================";
    cout << "\nMA TRAN B^-1";
    cout << "\n==============================\n";

    printMatrix(invB, n);

    // ==========================
    // Buoc 6: Tinh A^-1
    // ==========================

    vector<vector<double>> invA(n, vector<double>(n, 0.0));

    if (isSymmetric) {

        invA = invB;
    }
    else {

        cout << "\n==============================";
        cout << "\nTINH A^-1 = B^-1 * A^T";
        cout << "\n==============================\n";

        for (int i = 0; i < n; i++) {

            for (int j = 0; j < n; j++) {

                for (int k = 0; k < n; k++) {
                    invA[i][j] += invB[i][k] * A[j][k];
                }

                cout << "A^-1[" << i << "][" << j
                     << "] = "
                     << invA[i][j]
                     << endl;
            }
        }
    }

    // ==========================
    // Ket qua cuoi
    // ==========================

    cout << "\n==============================";
    cout << "\nMA TRAN NGHICH DAO A^-1";
    cout << "\n==============================\n";

    printMatrix(invA, n);

    return 0;
}
