#include "linalg.hpp"
#include "buildLaplacian.hpp"
#include "mf.hpp"

#include <chrono>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>

using namespace std;
using namespace LinAlg;
using namespace TreeFactor3D;

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
void testProblemUniform(int Nx, int Ny, int Nz, int width=1);
void testProblemRandom(int Nx, int Ny, int Nz, int width=1);

// void testProblemRandom(int Nx, int Ny, int width=1);

void testProblem(int Nx, int Ny, int Nz, SMat M, int width=1);

void testProblemUpdating(int Nx, int Ny, int Nz, int width=1);

int main(int argc, char* argv[]) {
	int Nx = 32, Ny = 32, Nz = 32, width=1;

	if(!(argc ==1 || argc == 5)){
		cout << "Usage: ./testMF3D Nx Ny Nz width" << endl;
		return 0;
	}

	if (argc != 1) {
		Nx    = atoi(argv[1]);
		Ny    = atoi(argv[2]);
		Nz    = atoi(argv[3]);
		width = atoi(argv[4]);
	}
	//cout << "UNIFORM TEST PROBLEM" << endl;
	testProblemUniform(Nx,Ny,Nz,width);
	// testProblemUniform(16,16,16,width);
	// testProblemUniform(24,24,24,width);
	// testProblemUniform(32,32,32,width);
	// testProblemUniform(48,48,48,width);
	// testProblemUniform(64,64,64,width);
	// testProblemUniform(90,90,90,width);
	//testProblemUniform(128,128,128,width);
	//testProblemUniform(128,128,128,width);
	//testProblemUniform(128,128,128,width);

	//testProblemUniform(128,128,128,width);
	//testProblemUniform(256,256,256,width);
	// cout << "RANDOM TEST PROBLEM" << endl;
	//testProblemRandom(Nx,Ny,Nz,width);
	// cout << "UPDATING TEST PROBLEM" << endl;
	testProblemUpdating(Nx, Ny, Nz, width);

	// testProblemUpdating(16,16,16,width);
	// testProblemUpdating(24,24,24,width);
	// testProblemUpdating(32,32,32,width);
	// testProblemUpdating(48,48,48,width);
	// testProblemUpdating(64,64,64,width);
	// testProblemUpdating(90,90,90,width);
	return 0;
}


/* implementation */
double rand_unif(int max=INT_MAX, double scal=1) {
	int num = rand() % max;
	return scal * double(num) / double(max);

}


void genRandomCoefficientField(int Nx, int Ny, int Nz, int width, vector<vector<vector<double> > > &A) {
	double tol = 1e-10;
	A.resize(Nx+2*width);
	for (int i = 0; i < Nx+2*width; i++){
		A[i].resize(Ny+2*width);
		for (int j = 0; j < Ny+2*width; j++) {
			A[i][j].resize(Nz+2*width);
		}
	}
	for (int i = 0; i < Nx+2*width; i++){
		for (int j = 0; j < Ny+2*width; j++) {
			for (int k = 0; k < Nz+2*width; k++) {
				double val = 1 + rand_unif();
				A[i][j][k] = val > tol? val : tol;
			}
		}
	}
}


/* For sanity, try with regular Laplacian no funny stuff */
void testProblemRandom(int Nx, int Ny, int Nz, int width) {
	vector<vector<vector<double> > > A;
	genRandomCoefficientField(Nx, Ny, Nz, width, A);
	SMat M = buildLaplacian(Nx, Ny, Nz, A, width, 1.0, 1.0, 1.0);
	testProblem(Nx, Ny, Nz, M, width);
}



void genUniformCoefficientField(int Nx, int Ny, int Nz, int width, vector<vector<vector<double > > > &A) {
	// Build a uniform cofficient field
	A.resize(Nx+2*width);
	for (int i = 0; i < Nx+2*width; i++){
		A[i].resize(Ny+2*width);
		for (int j = 0; j < Ny + 2*width; j++) {
			A[i][j].resize(Nz+2*width);
		}
	}
	for (int i = 0; i < Nx+2*width; i++){
		for (int j = 0; j < Ny+2*width; j++) {
			for (int k = 0; k < Nz+2*width; k++) {
				A[i][j][k] = 1;
			}
		}
	}
}

void testProblemUpdating(int Nx, int Ny, int Nz, int width) {
	srand(1234);
	int blockSize =4;
	assert(Nx >= blockSize);
	assert(Ny >= blockSize);
	assert(Nz >= blockSize);

	vector<vector<vector<double> > > A1, A2;
	genUniformCoefficientField(Nx, Ny, Nz, width, A1);
	SMat M1 = buildLaplacian(Nx, Ny, Nz, A1, width, 1.0, 1.0, 1.0);

	A2 = A1;
	for (int i = 0; i < blockSize; i++) {
		for (int j = 0; j < blockSize; j++) {
			for (int k = 0; k < blockSize; k++) {
				A2[i][j][k] *= 2;
			}
		}
	}

	SMat M2 = buildLaplacian(Nx, Ny, Nz, A2, width, 1.0, 1.0, 1.0);

	/* The DOFs that were modified */
	set<int> modDOF, allDOF;
	for (int idx = 0; idx < M1.nnz(); idx++) {
		allDOF.insert(M1.col_ind(idx));
		if (M1.val(idx) != M2.val(idx)) {
			modDOF.insert(M1.col_ind(idx));
		}
	}
	cout << "Testing a modification of " << modDOF.size() << " DOFs" << endl;

	// Do a true application of the matrix to a random vector
	RVec x(Nx*Ny*Nz);
	for (int i =0; i< Nx*Ny*Nz; i++) {
		x(i) = rand_unif();
	}

	RVec btrue = M2*x;



	MF3D<double> F(M1, Nx, Ny, Nz, width, 4);

	auto t1 = chrono::high_resolution_clock::now();
	F.Factor();
	auto t2 = chrono::high_resolution_clock::now();

	cout << LINE << "Factorization took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count()
         << " milliseconds (wall time)\n";


	RVec b1 = F.Apply(x);
	b1 -= btrue;



	t1 = chrono::high_resolution_clock::now();
	F.Update(M2, modDOF);
	t2 = chrono::high_resolution_clock::now();

	cout << LINE << "Update took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count()
         << " milliseconds (wall time)\n";


	cout << LINE;
	cout << "Relative apply error without updating is: " << endl
		 << b1.norm() / btrue.norm() << endl;



	RVec b2 = F.Apply(x);
	b2 -= btrue;

	cout << "Relative apply error after updating is: " << endl
		 << b2.norm() / btrue.norm() << endl;


}

/* For sanity, try with regular Laplacian no funny stuff */
void testProblemUniform(int Nx, int Ny, int Nz, int width) {
	vector<vector<vector<double> > > A;
	genUniformCoefficientField(Nx, Ny, Nz, width, A);
	SMat M = buildLaplacian(Nx, Ny, Nz, A, width, 1.0, 1.0, 1.0);
		//M.printMatlab();
	testProblem(Nx, Ny, Nz, M, width);
}

// /* For sanity, try with regular Laplacian no funny stuff */
// void testProblemRandom(int Nx, int Ny, int Nz, int width) {
// 	vector<vector<vector<double> > > A;
// 	genRandomCoefficientField(Nx, Ny, Nz, width, A);
// 	SMat M = buildLaplacian(Nx, Ny, Nz, A, width, 1.0, 1.0, 1.0);
// 	testProblem(Nx, Ny, Nz, M, width);
// }


/* Implementation of funtions */
void testProblem(int Nx, int Ny, int Nz, SMat M, int width) {
	// Seed for consistency
	srand(1234);

	cout << "Testing MF factorization with Nx = " <<Nx <<" Ny = " << Ny
		 << " Nz = " << Nz << " (" << Nx*Ny*Nz << " unknowns)"<< endl;

	// Do a true application of the matrix to a random vector
	RVec x(Nx*Ny*Nz);
	for (int i =0; i< Nx*Ny*Nz; i++) {
		x(i) = rand_unif();
	 }
	RVec btrue = M * x;

	// Now build a multifrontal factorization and time how long it takes
	MF3D<double> F(M, Nx, Ny, Nz, width, 4);
	auto t1 = chrono::high_resolution_clock::now();
	F.Factor();
	auto t2 = chrono::high_resolution_clock::now();
	cout << LINE << "Factorization took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count()
         << " milliseconds (wall time)\n";

    // Now apply the factorization and time it
    t1 = chrono::high_resolution_clock::now();
	RVec b = F.Apply(x);
	t2 = chrono::high_resolution_clock::now();
	cout << LINE << "Apply took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count()
         << " milliseconds (wall time)\n";

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
