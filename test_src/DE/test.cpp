#include "linalg.hpp"
#include "mf.hpp"
#include <chrono>
#include <stdlib.h>

#include <assert.h>

using namespace std;
using namespace LinAlg;
using namespace TreeFactor2D;

typedef std::complex<double> dcomplex;

typedef Dense<double> RMat;
typedef Vec<double> RVec;
typedef Dense<dcomplex> CMat;
typedef Vec<dcomplex> CVec;
typedef std::vector<int> IdxVec;

typedef Sparse<double> SMat;
typedef std::vector<double> SclVec;

#define LINE "--------------------------------------------------------------------------------\n"

int mf_test() {
	cout << "Testing MF" << endl;
	int Nx = 30, Ny = 20;
	SMat A("laplacian.dat");

	MF2D<double> M(A, Nx, Ny,1,4);
	cout << "Ready to factor" << endl;
	M.Factor();

	RVec x(Nx*Ny);
	for (int i =0; i< Nx*Ny; i++) {
		x(i) = i;
	}
	RVec btrue = A * x;

	btrue.print();
	RVec b = M.Apply(x);
	b-= btrue;
	b.print();
	cout << b.norm() / btrue.norm() << endl;
	return 0;
}


int mf_test_sizes() {
	srand(1234);
	const int n_tests = 5;
	int Nxs[] = {128, 256, 512, 1024, 2048};
	int Nys[] = {64, 128, 256, 512, 1024};
	string filenames[] = {"laplacian64_128.dat","laplacian128_256.dat","laplacian256_512.dat", "laplacian512_1024.dat","laplacian1024_2048.dat"};
	for (int k = 0; k < n_tests; k++){
		int Nx = Nxs[k], Ny = Nys[k];
		cout << "Testing MF with Nx = " <<Nx <<" and Ny = " << Ny << endl;
		cout << "(filename is " << filenames[k] << ")" << endl;
		SMat A(filenames[k]);


		RVec x(Nx*Ny);
		for (int i =0; i< Nx*Ny; i++) {
			x(i) = rand();
		 }
		RVec btrue = A * x;


		MF2D<double> M(A, Nx, Ny,1,16);
		auto t1 = std::chrono::high_resolution_clock::now();
		M.Factor();
		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << LINE << "Factor took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds (walltime)\n";

		t1 = std::chrono::high_resolution_clock::now();
		RVec b = M.Apply(x);
		t2 = std::chrono::high_resolution_clock::now();

		std::cout << LINE << "Apply took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds (walltime)\n";

		RVec c(btrue);
		b-= btrue;
		cout << "Relative apply error is: " << endl;
		cout << b.norm() / btrue.norm() << endl;

		t1 = std::chrono::high_resolution_clock::now();
		RVec y = M.Solve(c);
		t2 = std::chrono::high_resolution_clock::now();

		std::cout << LINE << "Solve took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds (walltime)\n";

		y -= x;
		cout << "Relative solve error is: " << endl;
		cout << y.norm() / x.norm() << endl;
		cout << LINE << LINE << endl;
	}
	return 0;
}

int mf_testBig() {
	cout << "Testing MF" << endl;
	int Nx = 60, Ny = 40;
	SMat A("laplacianBig.dat");

	MF2D<double> M(A, Nx, Ny,1,4);
	cout << "Ready to factor" << endl;
	M.Factor();

	RVec x(Nx*Ny);
	for (int i =0; i< Nx*Ny; i++) {
		x(i) = i;
	}
	RVec btrue = A * x;

	RVec b = M.Apply(x);
	RVec c(b);
	b-= btrue;
	cout << b.norm() / btrue.norm() << endl;

	RVec y = M.Solve(c);
	y -= x;
	cout << y.norm() / x.norm() << endl;
	return 0;
}

int mf_test2() {
	cout << "Testing MF2" << endl;
	int Nx = 30, Ny = 20;
	SMat A("laplacian2.dat");


	MF2D<double> M(A, Nx, Ny, 2);
	cout << "Ready to factor" << endl;
	M.Factor();


	RVec x(Nx*Ny);
	for (int i =0; i< Nx*Ny; i++) {
		x(i) = i;
	}
	RVec btrue = A * x;

	RVec b = M.Apply(x);
	RVec c(b);
	b-= btrue;
	cout << b.norm() / btrue.norm() << endl;

	RVec y = M.Solve(c);
	y -= x;
	cout << y.norm() / x.norm() << endl;
	return 0;
}

int sparse_mat_vec_test() {
	// Sparse test
	cout << "Testing sparse" << endl;
	int m = 4, n = 4;

	int r[] = { 0, 2, 4, 7, 9};
	int c[] = {0, 1, 1, 2, 0, 2, 3, 1, 3};
	double v[] = {1,7,2,8,5,3,9,6,4};
	IdxVec rr(r, r+5);
	IdxVec cc(c, c+9);
	SclVec vv(v, v+9);


	SMat A(vv, cc, rr,m,n);
	A.print();

	RVec x(n);
	x(0) = 1;
	x(1) = 2;
	x(2) = 3;
	x(3) = 4;
	x.print();
	RVec y = A*x;
	y.print();
	IdxVec is(2), js(2);
	is[0] = 0;
	is[1] = 3;
	js[0] = 2;
	js[1] = 1;

	RMat B = A(is,js);
	B.print();

	return 0;
}

int real_mat_vec_test() {
	cout << "Testing real MatVec" << endl;
	int m = 5, n = 3;
	RVec v(n);
	RMat A(m,n);
	for (int i = 0; i < n; i++){
		v(i)   = i;
		A(i,i) = 1;
		A(i+1,i) = 2;
	}
	RVec y = A*v;
	A.print();
	v.print();
	y.print();
	RVec z = A.T()*y;
	A.T().print();
	z.print();
	return 0;
}

int real_vec_test() {
	cout << "Testing assorted routines on real vectors" << endl;
	int n = 3;
	RVec v(n);
	v(0) = 1;
	v(1) = 2;
	v(2) = 3;
	v.print();
	v *=2;
	v.print();
	v += v;
	v.print();

	IdxVec idxs(2);
	idxs[0] = 0;
	idxs[1] = 2;
	RVec y = v(idxs);
	cout << &y << endl;
	cout << &v << endl;
	y.print();

	v -= v;
	v.print();


	return 0;
}

int real_test() {
	cout << "Testing assorted routines on real matrices" << endl;
	int m = 3;
	int n = 3;

	RMat A(m,n);
	A(0,0) = 1;
	A(1,1) = 2;
	A(0,1) = 10;
	A(1,2) = 100;

	cout << "A" << endl;
	A.print();
	cout << "A transpose" << endl;
	A.T().print();
	A /= 2;
	cout << "A/2" << endl;
	A.print();
	A -= A;
	cout << "Zeros" << endl;
	A.print();
	A(1,1) = 1;
	RMat B = A;
	B.print();
	B *= 2;
	B.print();
	A = B;
	A.print();

	IdxVec rows(2), cols(2);
	rows[0] = 1;
	rows[1] = 2;
	cols[0] = 0;
	cols[1] = 1;
	RMat C = A(rows,cols);
	C.print();

	return 0;
}

int complex_test() {
	cout << "Testing assorted routines on complex matrices" << endl;
	int m = 2;
	int n = 3;

	CMat A(m,n);
	A(0,0) = 1;
	A(1,1) = 2;
	A(0,1) = 10;
	A(1,2) = 100;

	cout << "A" << endl;
	A.print();
	cout << "A transpose" << endl;
	A.T().print();
	A /= 2;
	cout << "A/2" << endl;
	A.print();
	A -= A;
	cout << "Zeros" << endl;
	A.print();


	return 0;
}

int test_real_inv() {

	cout << "testing inverse" << endl;
	int n = 3;

	RMat A(n,n);
	A(0,0) = 1;
	A(1,1) = 1;
	A(2,2) = 1;
	A(0,1) = -2;
	A.print();
	Inv(A);
	A.print();
	return 0;
}

int test_real_mult() {
	int m = 2, n = 3, k = 4, p = 6;
	RMat A(m,k), B(k,n), C(p,k);
	A(0,0) = 1;
	A(1,2) = 2;
	A(1,1) = -2;

	B(0,0) = 1;
	B(2,2) = 4;
	B(1,2) = 1;

	C(0,0) = 5;
	C(2,1) = 10;

	RMat D = A * B;
	RMat E = A * C.T();
	cout << "A" << endl;
	A.print();
	cout << "B" << endl;
	B.print();
	cout << "A*B" << endl;
	D.print();
	cout << "A" << endl;
	A.print();
	cout << "C^T" << endl;
	C.T().print();
	cout << "A*C^T" << endl;
	E.print();
	RMat F = B.T()*C.T();
	cout << "B^T" << endl;
	B.T().print();
	cout << "C^T" << endl;
	C.T().print();
	cout << "B^T*C^T" << endl;
	F.print();


	return 0;
}

int test_complex_mult() {
	int m = 2, n = 3, k = 4, p = 6;
	CMat A(m,k), B(k,n), C(p,k);
	A(0,0) = dcomplex(0,1);
	A(1,2) = dcomplex(1,1);
	A(1,1) = -2;

	B(0,0) = 1;
	B(2,2) = dcomplex(1,2);
	B(1,2) = 1;

	C(0,0) = 5;
	C(2,1) = dcomplex(2,5);

	CMat D = A * B;
	CMat E = A * C.T();
	CMat F = C * A.H();

	A.print();
	B.print();
	D.print();

	A.print();
	C.T().print();
	E.print();

	C.print();
	A.H().print();
	F.print();


	return 0;
}

void test_lusolve(int N)
{
	RMat A(N,N);
	RMat ALU(N,N);
	RMat B(N,N);
	RMat BLU(N,N);
	RMat Approx(N,N);

	cout << "Real LUSolve test" << endl;
	for(int i=0; i<N; i++)
	{
		for(int j=0; j<N; j++)
		{
			A(j,i) = 0;
			B(j,i) = 1;
		}
	}
	for(int j=0; j<N; j++)
	{
		A(j,j) = 1;
		B(j,j) = 3;
	}
	A(1,2) = 1;
	A(2,2) = 3;
	A(7,7) = 7;
	A(4,0) = 2;

	ALU = A;
	BLU = B;

	LU(ALU);
	LUSolve(ALU, BLU);

	Approx = A*BLU;
	Approx -= B;
	cout << "Error" << endl;
	cout << Approx.norm() << endl;
}

void test_basicinterp(int N)
{
	RMat A(N,N);
	RMat Approx(1,1);
	RMat T(1,1);
	IdxVec sk(1);
	IdxVec rd(1);
	IdxVec all(N);

	cout << "Real InterpDecomp test" << endl;
	for(int i=0; i<N; i++)
	{
		all[i] = i;
	}

	for(int i=0; i<N; i++)
	{
		for(int j=0; j<N; j++)
		{
			A(j,i) = 0;
		}
	}
	for(int j=0; j<N; j++)
	{
		A(j,j) = 1;
	}
	A(1,2) = 1;
	A(2,2) = 1e-12;
	A(7,7) = 1e-12;
	InterpDecomp(A,T,sk,rd,1e-8);
	// cout << "sk" << endl;
	// for (auto c : sk)
	// 	cout << c << endl;
	// cout << "rd" << endl;
	// for (auto c : rd)
	// 	cout << c << endl;
	// cout << "T" << endl;
	// T.print();

	cout << "Number of rd columns" << endl;
	cout << rd.size() << endl;
	Approx.resize(N,rd.size());
	Approx = A(all,sk)*T;
	Approx -= A(all,rd);
	cout << "Error (should be < 1e-8)" << endl;
	cout << Approx.norm() << endl;
	// cout << "rd columns" << endl;
	// A(all,rd).print();
	// cout << "approximation" << endl;
	// Approx.print();


}


void test_basicinterpcpx(int N)
{
	CMat A(N,N);
	CMat Approx(1,1);
	CMat T(1,1);
	IdxVec sk(1);
	IdxVec rd(1);
	IdxVec all(N);


	cout << "Complex InterpDecomp test" << endl;
	for(int i=0; i<N; i++)
	{
		all[i] = i;
	}

	for(int i=0; i<N; i++)
	{
		for(int j=0; j<N; j++)
		{
			A(j,i) = 0;
		}
	}
	for(int j=0; j<N; j++)
	{
		A(j,j) = 1;
	}
	A(1,2) = std::polar(1.0,2.15);
	A(2,2) = 1e-12;
	A(7,7) = 1e-12;
	InterpDecomp(A,T,sk,rd,1e-8);
	// cout << "sk" << endl;
	// for (auto c : sk)
	// 	cout << c << endl;
	// cout << "rd" << endl;
	// for (auto c : rd)
	// 	cout << c << endl;
	// cout << "T" << endl;
	// T.print();


	cout << "Number of rd columns" << endl;
	cout << rd.size() << endl;
	Approx.resize(N,rd.size());
	Approx = A(all,sk)*T;
	Approx -= A(all,rd);
	cout << "Error (should be < 1e-8)" << endl;
	cout << Approx.norm() << endl;
	// cout << "rd columns" << endl;
	// A(all,rd).print();
	// cout << "approximation" << endl;
	// Approx.print();
}

void test_transpose(int M, int N)
{
	RMat A(M,N);
	RMat AT(N,M);
	cout << "Transpose test" << endl;

	for(int i=0; i<N; i++)
	{
		for(int j=0; j<M; j++)
		{
			A(j,i) = i*j;
		}
	}
	A.print();
	transpose(A,AT);
	AT.print();
}


int main() {
	//for (int i =0; i<100;i++){
		//real_test();
		//complex_test();
		// test_real_mult();
		// test_complex_mult();

		// test_real_inv();
		// test_basicinterp(10);
		// test_basicinterpcpx(10);

		//real_vec_test();
		//real_mat_vec_test();

		//sparse_mat_vec_test();
		// mf_testBig();
		// mf_test2();
		// mf_test_sizes();

		test_transpose(5,3);
		test_transpose(4,7);
	//}

	return 0;
}