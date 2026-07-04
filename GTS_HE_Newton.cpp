#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
using namespace std;

typedef vector<double> Vec;
typedef vector<vector<double>> Mat;

// =======================
// S?A HÀM F T?I ÐÂY
// ví d?:
// f1 = x^2 + y^2 - 4
// f2 = x - y
// =======================
Vec F(const Vec& x) {
    double x1 = x[0];
    double x2 = x[1];

    Vec f(2);
    f[0] = x1*x1 + x2*x2 - 4;
    f[1] = x1 - x2;

    return f;
}

// =======================
// MA TR?N JACOBI
// =======================
Mat J(const Vec& x) {
    double x1 = x[0];
    double x2 = x[1];

    Mat j(2, Vec(2));

    j[0][0] = 2*x1;
    j[0][1] = 2*x2;

    j[1][0] = 1;
    j[1][1] = -1;

    return j;
}

// =======================
// Gi?i h? tuy?n tính Gauss
// =======================
Vec solveLinear(Mat A, Vec b) {
    int n = b.size();

    for(int i=0;i<n;i++){
        A[i].push_back(b[i]);
    }

    for(int i=0;i<n;i++){
        double pivot = A[i][i];
        for(int j=i;j<=n;j++)
            A[i][j] /= pivot;

        for(int k=0;k<n;k++){
            if(k==i) continue;
            double factor = A[k][i];
            for(int j=i;j<=n;j++)
                A[k][j] -= factor*A[i][j];
        }
    }

    Vec x(n);
    for(int i=0;i<n;i++)
        x[i] = A[i][n];

    return x;
}

// =======================
// chu?n vô cùng
// =======================
double normInf(const Vec& v){
    double m = 0;
    for(double x : v)
        m = max(m, abs(x));
    return m;
}

// =======================
// NEWTON RAPHSON
// =======================
int main(){

    int n = 2;
    Vec x(n);

    cout << "Nhap x0:\n";
    for(int i=0;i<n;i++)
        cin >> x[i];

    double eps;
    int maxIter;

    cout << "epsilon = ";
    cin >> eps;

    cout << "maxIter = ";
    cin >> maxIter;

    cout << fixed << setprecision(8);

    for(int k=1;k<=maxIter;k++){

        Vec fx = F(x);
        Mat jx = J(x);

        for(double &v : fx) v = -v;

        Vec delta = solveLinear(jx, fx);

        Vec x_new(n);
        for(int i=0;i<n;i++)
            x_new[i] = x[i] + delta[i];

        double err = normInf(delta);

        cout << "k = " << k << "  ";
        for(double v : x_new)
            cout << v << "  ";
        cout << "err = " << err << endl;

        if(err < eps){
            cout << "\nHoi tu\n";
            x = x_new;
            break;
        }

        x = x_new;
    }

    cout << "\nNghiem:\n";
    for(int i=0;i<n;i++)
        cout << "x" << i+1 << " = " << x[i] << endl;

}
