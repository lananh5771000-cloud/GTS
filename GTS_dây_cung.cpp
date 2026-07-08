#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
using namespace std;
 
// ham fx
double f(double x){
    return pow(x,5) - 625*x -3125;
}
// dao ham lan 1
double f1(double x){
    return 5*pow(x,4) - 625;
}
//dao ham lan 2
double f2(double x){
    return 20*pow(x,3);
}

int main(){
    cout << "--- HE THONG GIAI PT: LY LUAN + TINH TOAN + LUA CHON IN ---\n";

    double a, b, eps;
    int precision;

    cout << "Nhap khoang cach ly a: "; cin >> a;
    cout << "Nhap khoang cach ly b: "; cin >> b;
    cout << "Sai so eps: "; cin >> eps;
    cout << "So chu so sau dau phay: "; cin >> precision;

    cout << "\nChon phuong phap:\n";
    cout << "1. Chia doi\n2. Lap don\n3. Tiep tuyen (Newton)\n4. Day cung\n";
    cout << "Lua chon (1-4): ";
    
    int choice;
    cin >> choice;

    vector<vector<double> > history;

    // ================= CHIA DOI =================
    if(choice == 1){
        // KI?M TRA ÐI?U KI?N
        if(f(a) * f(b) >= 0) {
            cout << "Loi: f(a)*f(b) >= 0. Khoang cach ly khong hop le!\n";
            return 1;
        }

        double curr_a = a, curr_b = b;
        int n = 0;

        while(true){
            double x = (curr_a + curr_b) / 2.0;
            double fx = f(x);
            double sai_so = fabs(curr_b - curr_a) / 2.0; // Thêm fabs cho an toàn

            history.push_back({(double)n, curr_a, curr_b, x, fx, sai_so});
            if(sai_so <= eps) break;

            if(f(curr_a) * fx < 0) curr_b = x;
            else curr_a = x;

            n++;
        }

        cout << "\n" << left << setw(5) << "n" << setw(12) << "a" << setw(12) << "b" 
             << setw(12) << "x" << setw(12) << "f(x)" << "sai so\n";
        for(auto &r : history){
            cout << left << setw(5) << (int)r[0] // Ép ki?u int cho n
                 << fixed << setprecision(precision)
                 << setw(12) << r[1] << setw(12) << r[2] << setw(12) << r[3] << setw(12) << r[4]
                 << scientific << r[5] << "\n";
        }
        cout << "\n=> NGHIEM: " << fixed << setprecision(precision) << history.back()[3] << endl;
    }

    // ================= LAP DON =================
    else if(choice == 2){
        auto phi = [](double x){
            return x - sin(x); // Ðã s?a l?i logic: x = x - sin(x) <=> sin(x) = 0
        };

        double x0 = (a + b) / 2.0;
        int n = 0;

        while(n < 100){
            double x1 = phi(x0);
            double ss = fabs(x1 - x0);

            history.push_back({(double)n, x0, x1, ss});
            if(ss < eps) break;

            x0 = x1;
            n++;
        }

        cout << "\n" << left << setw(5) << "n" << setw(15) << "x_n" << setw(15) << "x_n+1" << "sai so\n";
        for(auto &r : history){
            cout << left << setw(5) << (int)r[0]
                 << fixed << setprecision(precision)
                 << setw(15) << r[1] << setw(15) << r[2]
                 << scientific << r[3] << "\n";
        }
        cout << "\n=> NGHIEM: " << fixed << setprecision(precision) << history.back()[2] << endl;
    }

    // ================= NEWTON =================
    else if(choice == 3){
        double x0 = (f(a) * f2(a) > 0) ? a : b;
        int n = 0;
        
        while(n < 100){
            double fx = f(x0);
            double fpx = f1(x0);
            
            if (fpx == 0) { // Ch?ng chia cho 0
                cout << "Loi: Dao ham bang 0 tai n = " << n << endl;
                break;
            }

            double x1 = x0 - fx / fpx;
            double ss = fabs(x1 - x0);

            history.push_back({(double)n, x0, fx, x1, ss});
            if(ss < eps) break;

            x0 = x1;
            n++;
        }

        cout << "\n" << left << setw(5) << "n" << setw(15) << "x_n" << setw(15) << "f(x)" 
             << setw(15) << "x_n+1" << "sai so\n";
        for(auto &r : history){
            cout << left << setw(5) << (int)r[0]
                 << fixed << setprecision(precision)
                 << setw(15) << r[1] << setw(15) << r[2] << setw(15) << r[3]
                 << scientific << r[4] << "\n";
        }
        cout << "\n=> NGHIEM: " << fixed << setprecision(precision) << history.back()[3] << endl;
    }

    // ================= DAY CUNG =================
    else if(choice == 4){
        double d, x;
        if(f(a) * f2(a) > 0){ d = a; x = b; }
        else { d = b; x = a; }

        int n = 0;
        while(n < 100){
            double fx = f(x);
            double fd = f(d);
            
            if (fx - fd == 0) { // Ch?ng chia cho 0
                cout << "Loi: Mau so (fx - fd) = 0 tai n = " << n << endl;
                break;
            }

            double x_next = x - fx * (x - d) / (fx - fd);
            double ss = fabs(x_next - x);

            history.push_back({(double)n, d, x, x_next, ss});
            if(ss < eps) break;

            x = x_next;
            n++;
        }

        cout << "\n" << left << setw(5) << "n" << setw(12) << "d" << setw(15) << "x_n" 
             << setw(15) << "x_n+1" << "sai so\n";
        for(auto &r : history){
            cout << left << setw(5) << (int)r[0]
                 << fixed << setprecision(precision)
                 << setw(12) << r[1] << setw(15) << r[2] << setw(15) << r[3]
                 << scientific << r[4] << "\n";
        }
        cout << "\n=> NGHIEM: " << fixed << setprecision(precision) << history.back()[3] << endl;
    }

    return 0;
}
