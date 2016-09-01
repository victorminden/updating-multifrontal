#include "linalg.hpp"
#include "buildLaplacian.hpp"
#include "mf.hpp"

#include <chrono>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>

using namespace std;
using namespace LinAlg;
using namespace TreeFactor2D;

typedef Dense<double> RMat;
typedef Vec<double> RVec;

typedef Dense<dcomplex> CMat;
typedef Vec<dcomplex> CVec;

typedef std::vector<int> IdxVec;

typedef Sparse<double> SMat;
typedef std::vector<double> SclVec;

#undef LINE
#define LINE "--------------------------------------------------------------------------------\n"

// Forward-declare tests
void testProblemUniform(int Nx, int Ny, int width=1);
// void testProblemAdv(int Nx, int Ny, double vx, double vy);
void testProblem_adv(int Nx, int Ny, SMat M, int width=1);
void testProblemRandom(int Nx, int Ny, int width=1);

void testProblem(int Nx, int Ny, SMat& M, int width=1);


void testProblemUpdating(int Nx, int Ny, int width=1);

int main(int argc, char* argv[]) {
	int Nx = 1024, Ny =1024, width=1;

	if(!(argc ==1 || argc == 4)){
		cout << "Usage: ./testMF2D Nx Ny width" << endl;
		return 0;
	}

	if (argc != 1) {
		Nx    = atoi(argv[1]);
		Ny    = atoi(argv[2]);
		width = atoi(argv[3]);
	}
	// cout << "ADVECTION" << endl;
	//testProblemAdv(Nx, Ny, 100, 20);
	testProblemUniform(Nx, Ny, width);
	//testProblemUniform(32, 32, width);
	// testProblemUniform(64, 64, width);
	// testProblemUniform(128, 128, width);
	// testProblemUniform(256, 256, width);
	// testProblemUniform(512, 512, width);
	// testProblemUniform(1024, 1024, width);
	// testProblemUniform(2048, 2048, width);
	// testProblemUniform(4096, 4096, width);


	// testProblemRandom(Nx, Ny, width);
	// testProblemUpdating(Nx, Ny, width);
	// testProblemUpdating(23, 57, width);
	// testProblemUpdating(48, 75, width);
	// testProblemUpdating(212, 95, width);
	// testProblemUpdating(216, 309, width);
	// testProblemUpdating(518, 578, width);
	// testProblemUpdating(578, 802, width);
	// testProblemUpdating(1024, 912, width);
	// testProblemUpdating(2047, 1540, width);

	// 	testProblemUpdating(32, 32, width);
	// testProblemUpdating(64, 64, width);
	// testProblemUpdating(128, 128, width);
	// testProblemUpdating(256, 256, width);
	// testProblemUpdating(512, 512, width);
	// testProblemUpdating(1024, 1024, width);
	// testProblemUpdating(2048, 2048, width);
	// testProblemUpdating(4096, 4096, width);
	return 0;
}


/* implementation */
double rand_unif(int max=INT_MAX, double scal=1) {
	int num = rand() % max;
	return scal * double(num) / double(max);

}


void genRandomCoefficientField(int Nx, int Ny, int width, vector<vector<double> > &A) {
	double tol = 1e-10;
	A.resize(Nx+2*width);
	for (int i = 0; i < Nx+2*width; i++){
		A[i].resize(Ny+2*width);
	}
	for (int i = 0; i < Nx+2*width; i++){
		for (int j = 0; j < Ny+2*width; j++) {
			double val = rand_unif();
			A[i][j] = val > tol? val : tol;
		}
	}
}


void genUniformCoefficientField(int Nx, int Ny, int width, vector<vector<double> > &A) {
	// Build a uniform cofficient field
	A.resize(Nx+2*width);
	for (int i = 0; i < Nx+2*width; i++){
		A[i].resize(Ny+2*width);
	}
	for (int i = 0; i < Nx+2*width; i++){
		for (int j = 0; j < Ny+2*width; j++) {
			A[i][j] = 1;
		}
	}
}

void testProblemUpdating(int Nx, int Ny, int width) {
	srand(1234);
	int blockSize =8;
	assert(Nx >= blockSize);
	assert(Ny >= blockSize);

	vector<vector<double> > A1, A2;
	genUniformCoefficientField(Nx, Ny, width, A1);
	A2 = A1;
	for (int i = 0; i < blockSize; i++) {
		for (int j = 0; j < blockSize; j++) {
			A2[i][j] *= 2;
		}
	}

	SMat M1 = buildLaplacian(Nx, Ny, A1, width, 1.0, 1.0);
	SMat M2 = buildLaplacian(Nx, Ny, A2, width, 1.0, 1.0);

	/* The DOFs that were modified */
	set<int> modDOF;
	set<int> allDOF;
	for (int idx = 0; idx < M1.nnz(); idx++) {
		allDOF.insert(M1.col_ind(idx));
		if (M1.val(idx) != M2.val(idx)) {
			modDOF.insert(M1.col_ind(idx));
		}
	}
	cout << "Testing a modification of " << modDOF.size() << " DOFs" << endl;

	// Do a true application of the matrix to a random vector
	RVec x(Nx*Ny);
	for (int i =0; i< Nx*Ny; i++) {
		x(i) = rand_unif();
	}


	MF2D<double> F(M1, Nx, Ny, width, 8);
	auto t1 = chrono::high_resolution_clock::now();
	F.Factor();
	auto t2 = chrono::high_resolution_clock::now();

	cout << LINE << "Factorization took "
         << (double) chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds (wall time)\n";

	RVec btrue = M2 * x;

	RVec b1 = F.Apply(x);
	t1 = chrono::high_resolution_clock::now();
	F.Update(M2, modDOF);
	t2 = chrono::high_resolution_clock::now();

	cout << LINE << "Update took "
         << (double) chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds (wall time)\n";

	RVec b2 = F.Apply(x);


	b1 -= btrue;
	b2 -= btrue;
	cout << LINE;
	cout << "Relative apply error without updating is: " << endl
		 << b1.norm() / btrue.norm() << endl;
	cout << "Relative apply error after updating is: " << endl
		 << b2.norm() / btrue.norm() << endl;
	/* Gotta implement updating now that we have modified DOF sets */
	// testProblem(Nx, Ny, M1, width);
	// testProblem(Nx, Ny, M2, width);


}

/* For sanity, try with regular Laplacian no funny stuff */
void testProblemUniform(int Nx, int Ny, int width) {
	vector<vector<double> > A;
	genUniformCoefficientField(Nx, Ny, width, A);
	SMat M = buildLaplacian(Nx, Ny, A, width, 1.0, 1.0);
	//M.printMatlab();
	testProblem(Nx, Ny, M, width);
}


// void testProblemAdv(int Nx, int Ny, double vx, double vy) {
// 	int width = 1;
// 	vector<vector<double> > A;
// 	genRandomCoefficientField(Nx, Ny, width, A);
// 	double hx = 1./Nx;
// 	double hy = 1./Ny;
// 	SMat M = buildAdvectionDiffusion(Nx, Ny, A, vx, vy, hx, hy);
// 	//M.print();
// 	M.printMatlab();
// 	testProblem_adv(Nx, Ny, M, width);
// }

void testProblemRandom(int Nx, int Ny, int width) {
	vector<vector<double> > A;
	genRandomCoefficientField(Nx, Ny, width, A);
	SMat M = buildLaplacian(Nx, Ny, A, width, 1.0, 1.0);
	testProblem(Nx, Ny, M, width);
}


/* Implementation of funtions */
void testProblem(int Nx, int Ny, SMat& M, int width) {
	// Seed for consistency
	srand(1234);

	cout << "Testing MF factorization with Nx = " <<Nx <<" and Ny = " << Ny
		 << " (" << Nx*Ny << " unknowns)"<< endl;

	// Do a true application of the matrix to a random vector
	RVec x(Nx*Ny);
	for (int i =0; i< Nx*Ny; i++) {
		x(i) = rand_unif();
	 }
	RVec btrue = M * x;

	// Now build a multifrontal factorization and time how long it takes
	MF2D<double> F(M, Nx, Ny, width, 8);
	auto t1 = chrono::high_resolution_clock::now();
	F.Factor();
	auto t2 = chrono::high_resolution_clock::now();
	cout << LINE << "Factorization took "
         << (double) chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds (wall time)\n";

    // Now apply the factorization and time it
    t1 = chrono::high_resolution_clock::now();
	RVec b = F.Apply(x);
	t2 = chrono::high_resolution_clock::now();
	cout << LINE << "Apply took "
         << (double) chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds (wall time)\n";

    RVec c(btrue);
	b-= btrue;
	cout << "Relative apply error is: " << endl
		 << b.norm() / btrue.norm() << endl;

	// Now solve with the factorization and time it
	t1 = chrono::high_resolution_clock::now();
	RVec y = F.Solve(c);
	t2 = chrono::high_resolution_clock::now();

	cout << LINE << "Solve took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count()
         << " milliseconds (walltime)\n";

	y -= x;
	cout << "Relative solve error is: " << endl
	     << y.norm() / x.norm() << endl
	     << LINE << LINE << endl;

}


/* Implementation of funtions */
void testProblem_adv(int Nx, int Ny, SMat M, int width) {
	// Seed for consistency
	srand(1234);

	cout << "Testing MF factorization with Nx = " <<Nx <<" and Ny = " << Ny
		 << " (" << Nx*Ny << " unknowns)"<< endl;

	// Do a true application of the matrix to a random vector
	RVec x(Nx*Ny);
	RVec e(Nx*Ny);
	for (int i =0; i< Nx*Ny; i++) {
		x(i) = rand_unif();
	 }
	e(Nx*Ny/2 + Nx/2) = 1;
	RVec btrue = M * x;

	// Now build a multifrontal factorization and time how long it takes
	MF2D<double> F(M, Nx, Ny, width, 8);
	auto t1 = chrono::high_resolution_clock::now();
	F.Factor();
	auto t2 = chrono::high_resolution_clock::now();
	cout << LINE << "Factorization took "
         << (double) chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds (wall time)\n";

    // Now apply the factorization and time it
    t1 = chrono::high_resolution_clock::now();
	RVec b = F.Apply(x);
	t2 = chrono::high_resolution_clock::now();
	cout << LINE << "Apply took "
         << (double) chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds (wall time)\n";

    RVec c(btrue);
	b-= btrue;
	cout << "Relative apply error is: " << endl
		 << b.norm() / btrue.norm() << endl;

	// Now solve with the factorization and time it
	t1 = chrono::high_resolution_clock::now();
	RVec y = F.Solve(c);
	t2 = chrono::high_resolution_clock::now();

	cout << LINE << "Solve took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count()
         << " milliseconds (walltime)\n";

	y -= x;
	cout << "Relative solve error is: " << endl
	     << y.norm() / x.norm() << endl
	     << LINE << LINE << endl;

	     	// Now solve with the factorization and time it
	t1 = chrono::high_resolution_clock::now();
	RVec prof = F.Solve(e);
	t2 = chrono::high_resolution_clock::now();

	cout << LINE << "Solve took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count()
         << " milliseconds (walltime)\n";

    prof.print();


}
