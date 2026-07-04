#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;

void print_matrix(vector<vector<double>> &matrix, int m, int n, int k,
                  int decimals, string step_name = "")
{
    if (!step_name.empty())
        cout << "\n--- " << step_name << " ---\n";

    for (int i = 0; i < m; i++)
    {
        cout << "[";
        for (int j = 0; j < n; j++)
            cout << setw(8) << fixed << setprecision(decimals)
                 << matrix[i][j] << "  ";

        cout << " | ";

        for (int j = n; j < n + k; j++)
            cout << setw(8) << fixed << setprecision(decimals)
                 << matrix[i][j] << "  ";

        cout << "]\n";
    }
    cout << string(50, '-') << endl;
}

void solve_gauss_jordan_complete()
{
    int m, n, k;

    cout << "Nhap so dong A: ";
    cin >> m;

    cout << "Nhap so cot A (so bien): ";
    cin >> n;

    vector<vector<double>> A(m, vector<double>(n));
    vector<vector<double>> B;

    cout << "\nNhap ma tran A:\n";
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            cin >> A[i][j];

    cout << "\nNhap so cot cua B: ";
    cin >> k;

    B.resize(m, vector<double>(k));

    cout << "Nhap ma tran B:\n";
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            cin >> B[i][j];

    int decimals;
    cout << "\nNhap so chu so sau dau phay: ";
    cin >> decimals;

    char in_thuat_toan;
    cout << "In thuat toan? (y/n): ";
    cin >> in_thuat_toan;

    // t?o ma tr?n m? r?ng
    vector<vector<double>> aug(m, vector<double>(n + k));

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            aug[i][j] = A[i][j];

        for (int j = 0; j < k; j++)
            aug[i][n + j] = B[i][j];
    }

    cout << "\nMA TRAN MO RONG BAN DAU:\n";
    print_matrix(aug, m, n, k, decimals);

    int row = 0, col = 0;

    vector<int> pivot_cols;
    vector<int> var_order(n);

    for (int i = 0; i < n; i++)
        var_order[i] = i;

    while (row < m && col < n)
    {
        bool found_one = false;
        int pivot_r = row;
        int pivot_c = col;
        double max_val = -1;

        // tìm ph?n t? = 1
        for (int i = row; i < m; i++)
        {
            for (int j = col; j < n; j++)
            {
                if (fabs(fabs(aug[i][j]) - 1.0) < 1e-10)
                {
                    pivot_r = i;
                    pivot_c = j;
                    found_one = true;
                    break;
                }
            }
            if (found_one)
                break;
        }

        // n?u không có 1 thì tìm max
        if (!found_one)
        {
            for (int i = row; i < m; i++)
                for (int j = col; j < n; j++)
                    if (fabs(aug[i][j]) > max_val)
                    {
                        max_val = fabs(aug[i][j]);
                        pivot_r = i;
                        pivot_c = j;
                    }
        }

        if (fabs(aug[pivot_r][pivot_c]) < 1e-10)
        {
            col++;
            continue;
        }

        // d?i dòng
        if (pivot_r != row)
        {
            swap(aug[row], aug[pivot_r]);
            cout << "-> Doi dong " << row + 1
                 << " va " << pivot_r + 1 << endl;
        }

        // d?i c?t
        if (pivot_c != col)
        {
            for (int i = 0; i < m; i++)
                swap(aug[i][col], aug[i][pivot_c]);

            swap(var_order[col], var_order[pivot_c]);

            cout << "-> Doi cot " << col + 1
                 << " va " << pivot_c + 1 << endl;
        }

        double pivot_val = aug[row][col];

        cout << "-> Pivot tai (" << row + 1
             << "," << col + 1 << ") = "
             << fixed << setprecision(decimals)
             << pivot_val << endl;

        pivot_cols.push_back(col);

        // chu?n hóa
        for (int j = col; j < n + k; j++)
            aug[row][j] /= pivot_val;

        // kh?
        for (int i = 0; i < m; i++)
        {
            if (i != row)
            {
                double factor = aug[i][col];

                for (int j = col; j < n + k; j++)
                    aug[i][j] -= factor * aug[row][j];
            }
        }

        print_matrix(aug, m, n, k, decimals,
                     "Sau khi khu cot " + to_string(col + 1));

        row++;
        col++;
    }

    cout << "\nKET LUAN NGHIEM:\n";

    for (int b_col = n; b_col < n + k; b_col++)
    {
        cout << "\n--- Cot B " << b_col - n + 1 << " ---\n";

        bool no_solution = false;

        for (int i = 0; i < m; i++)
        {
            bool all_zero = true;

            for (int j = 0; j < n; j++)
                if (fabs(aug[i][j]) > 1e-10)
                    all_zero = false;

            if (all_zero && fabs(aug[i][b_col]) > 1e-10)
                no_solution = true;
        }

        if (no_solution)
        {
            cout << "VO NGHIEM\n";
            continue;
        }

        if (pivot_cols.size() == n)
        {
            cout << "NGHIEM DUY NHAT:\n";

            vector<double> ans(n);

            for (int i = 0; i < n; i++)
                ans[var_order[i]] = aug[i][b_col];

            for (int i = 0; i < n; i++)
                cout << "x" << i + 1 << " = "
                     << fixed << setprecision(decimals)
                     << ans[i] << endl;
        }
        else
        {
            cout << "VO SO NGHIEM\n";
        }
    }

    if (in_thuat_toan == 'y')
    {
        cout << "\nTHUAT TOAN GAUSS-JORDAN COMPLETE PIVOTING\n";
        cout << "1. Tim pivot trong toan bo ma tran\n";
        cout << "2. Hoan vi dong va cot\n";
        cout << "3. Chuan hoa pivot = 1\n";
        cout << "4. Khu cac dong khac\n";
        cout << "5. Dua ve RREF\n";
    }
}

int main()
{
    solve_gauss_jordan_complete();
    return 0;
}
