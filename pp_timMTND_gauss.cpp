#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

const double EPS = 1e-9;

// In ma tran mo rong
void printMatrix(double a[][100], int n) {
    cout << fixed << setprecision(4);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 2 * n; j++) {
            if (j == n) cout << " | ";

            cout << setw(10) << a[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main() {
    int n;

    cout << "Nhap cap ma tran n = ";
    cin >> n;

    double a[100][100];

    cout << "\nNhap ma tran A:\n";
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> a[i][j];

    // Tao ma tran ma rong [A | I]
    double aug[100][100];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            aug[i][j] = a[i][j];

        for (int j = n; j < 2 * n; j++)
            aug[i][j] = (i == j - n) ? 1 : 0;
    }

    cout << "\nMa tran mo rong ban dau [A|I]:\n";
    printMatrix(aug, n);

    int step = 1;

    for (int col = 0; col < n; col++) {

        // Tìm phan tu troi lon nhat theo tri tuyet doi
        int pivotRow = col;

        for (int i = col + 1; i < n; i++) {
            if (fabs(aug[i][col]) > fabs(aug[pivotRow][col]))
                pivotRow = i;
        }

        if (fabs(aug[pivotRow][col]) < EPS) {
            cout << "Ma tran khong kha nghich!\n";
            return 0;
        }

        // Doi hang neu can
        if (pivotRow != col) {

            cout << "===== Buoc " << step++ << " =====\n";
            cout << "Doi hang R" << col + 1
                 << " va R" << pivotRow + 1 << endl;

            for (int j = 0; j < 2 * n; j++)
                swap(aug[col][j], aug[pivotRow][j]);

            printMatrix(aug, n);
        }

        // Chuan hoa hang tru
        double pivot = aug[col][col];

        cout << "===== Buoc " << step++ << " =====\n";
        cout << "Chuan hoa R" << col + 1
             << " (chia cho " << pivot << ")\n";

        for (int j = 0; j < 2 * n; j++)
            aug[col][j] /= pivot;

        printMatrix(aug, n);

        // Khu các hàng khác
        for (int i = 0; i < n; i++) {

            if (i == col) continue;

            double factor = aug[i][col];

            if (fabs(factor) < EPS) continue;

            cout << "===== Buoc " << step++ << " =====\n";

            cout << "R" << i + 1
                 << " = R" << i + 1
                 << " - (" << factor
                 << ")*R" << col + 1 << endl;

            for (int j = 0; j < 2 * n; j++)
                aug[i][j] -= factor * aug[col][j];

            printMatrix(aug, n);
        }
    }

    cout << "\n================================================\n";
    cout << "Da thu duoc dang [I|A^-1]\n";
    cout << "================================================\n";

    printMatrix(aug, n);

    cout << "\nMa tran nghich dao A^-1:\n\n";

    for (int i = 0; i < n; i++) {
        for (int j = n; j < 2 * n; j++) {
            cout << setw(10) << fixed
                 << setprecision(6)
                 << aug[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
