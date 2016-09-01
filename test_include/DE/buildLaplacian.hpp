#ifndef BUILD_LAPLACIAN
#define BUILD_LAPLACIAN 1
#include <vector>
#include "linalg.hpp"

typedef std::vector<int> IdxVec;
typedef LinAlg::Sparse<double> SMat;
typedef std::vector<double> SclVec;

SMat buildLaplacian(int Nx, int Ny, std::vector<std::vector<double> > A, int width, double hx, double hy) {
	// A is coefficient field of size Nx+2*width by Ny+2*width

	// MISSING ALL AVERAGING
	/* 2D problem */
	int dim = 2;
	double hhx = hx * hx, hhy = hy * hy;
	/* Assert for now that width is 1 or 2  or 3*/
	assert (width == 1 || width == 2 || width == 3);

	std::vector<std::vector<double> > coeffs(3);
	/* width 1 */
	coeffs[0].resize(2);
	coeffs[0][0] = 0; // has to just be the sum of the other guys
	coeffs[0][1] = -1;
	/* width 2 */
	coeffs[1].resize(3);
	coeffs[1][0] = 0; // has to just be the sum of the other guys
	coeffs[1][1] = -4.0/3.0;
	coeffs[1][2] = 1.0/12.0;
	/* width 3 */
	coeffs[2].resize(4);
	coeffs[2][0] = 0; // has to just be the sum of the other guys
	coeffs[2][1] = -3.0/2.0;
	coeffs[2][2] = 3.0/20.0;
	coeffs[2][3] = -1.0/90.0;


	/* We build the Laplacian in coordinate format and then convert to CSR */
	IdxVec row_ptr, col_idx;
	SclVec val_arr;

	int nnz_est = (width * 2 * dim + 1) * Nx * Ny;

	row_ptr.reserve(nnz_est);
	col_idx.reserve(nnz_est);
	val_arr.reserve(nnz_est);

	int offset = 0;
	for (int j = 0; j < Ny; j++) {
		for (int i = 0; i < Nx; i++){
			// Row starts at this idx
			row_ptr.push_back(offset);

			/* idxs relative to coefficient grid (one width unit larger in each direction)*/
			int ii = i+width;
			int jj = j+width;

			/* linear index */
			int idx = i + j * Nx;

			/* MUST go in order of column index */


			/* down neighbors */
			for (int k = width; k > 0; k--) {
				if (j-k >= 0) {
					int nbrIdx = i + (j-k) * Nx;
					double ahalf = 0.5 * (A[ii][jj-k] + A[ii][jj-k+1]);
					double v = coeffs[width-1][k] * ahalf / hhy;
					offset++;
					col_idx.push_back(nbrIdx);
					val_arr.push_back(v);
				}
			}
			/* left neighbor */
			for (int k = width; k > 0; k--){
				if (i-k >= 0) {
					int nbrIdx = (i-k) + j * Nx;
					double ahalf = 0.5*( A[ii-k][jj] + A[ii-k+1][jj]);
					double v = coeffs[width-1][k] * ahalf / hhx;
					offset++;
					col_idx.push_back(nbrIdx);
					val_arr.push_back(v);
				}
			}

			/* self */
			double v = 0;
			for (int k = width; k > 0; k--){
				double ahalf1 = 0.5* (A[ii-k][jj] + A[ii-k+1][jj]);
				double ahalf2 = 0.5* (A[ii+k][jj] + A[ii+k-1][jj]);
				double ahalf3 = 0.5* (A[ii][jj-k] + A[ii][jj-k+1]);
				double ahalf4 = 0.5* (A[ii][jj+k] + A[ii][jj+k-1]);
				v -= coeffs[width-1][k] * (ahalf1 + ahalf2) / hhx;
				v -= coeffs[width-1][k] * (ahalf3 + ahalf4) / hhy;
			}

			offset++;
			col_idx.push_back(idx);
			val_arr.push_back(v);

			/* right neighbor */
			for (int k = 1; k <= width; k++){
				if (i+k <= Nx-1) {
					int nbrIdx = (i+k) + j * Nx;
					double ahalf = 0.5*(A[ii+k][jj] + A[ii+k-1][jj]);
					double v = coeffs[width-1][k] * ahalf / hhx;

					offset++;
					col_idx.push_back(nbrIdx);
					val_arr.push_back(v);
				}
			}

			/* up neighbor */
			for (int k = 1; k <= width; k++){
				if (j+k <= Ny-1) {
					int nbrIdx = i + (j+k) * Nx;
					double ahalf = 0.5*(A[ii][jj+k] + A[ii][jj+k-1]);
					double v = coeffs[width-1][k] * ahalf / hhy;
					offset++;
					col_idx.push_back(nbrIdx);
					val_arr.push_back(v);
				}
			}
		}
	}
	// Put nnz at the end
	row_ptr.push_back(offset);

	return SMat(val_arr,col_idx,row_ptr, Ny*Nx, Ny*Nx);
}


// SMat buildAdvectionDiffusion(int Nx, int Ny, std::vector<std::vector<double> > A, double v_x, double v_y, double hx, double hy) {
// 	// A is coefficient field of size Nx+2*width by Ny+2*width

// 	/* 2D problem */
// 	int dim = 2;
// 	double hhx = hx * hx, hhy = hy * hy;
// 	/* Assert for now that width is 1*/
// 	int width = 1;
// 	assert (v_x > 0);
// 	assert (v_y > 0);

// 	std::vector<std::vector<double> > coeffs(3);
// 	/* width 1 */
// 	coeffs[0].resize(2);
// 	coeffs[0][0] = 0; // has to just be the sum of the other guys
// 	coeffs[0][1] = -1;


// 	/* We build the Laplacian in coordinate format and then convert to CSR */
// 	IdxVec row_ptr, col_idx;
// 	SclVec val_arr;

// 	int nnz_est = (width * 2 * dim + 1) * Nx * Ny;

// 	row_ptr.reserve(nnz_est);
// 	col_idx.reserve(nnz_est);
// 	val_arr.reserve(nnz_est);

// 	int offset = 0;
// 	for (int j = 0; j < Ny; j++) {
// 		for (int i = 0; i < Nx; i++){
// 			// Row starts at this idx
// 			row_ptr.push_back(offset);

// 			/* idxs relative to coefficient grid (one width unit larger in each direction)*/
// 			int ii = i+width;
// 			int jj = j+width;

// 			/* linear index */
// 			int idx = i + j * Nx;

// 			/* MUST go in order of column index */


// 			/* down neighbors */
// 			for (int k = width; k > 0; k--) {
// 				if (j-k >= 0) {
// 					int nbrIdx = i + (j-k) * Nx;
// 					double v = coeffs[width-1][k] * A[ii][jj-k] / hhy - v_y /hy;
// 					offset++;
// 					col_idx.push_back(nbrIdx);
// 					val_arr.push_back(v);
// 				}
// 			}
// 			/* left neighbor */
// 			for (int k = width; k > 0; k--){
// 				if (i-k >= 0) {
// 					int nbrIdx = (i-k) + j * Nx;
// 					double v = coeffs[width-1][k] * A[ii-k][jj] / hhx - v_x / hx;
// 					offset++;
// 					col_idx.push_back(nbrIdx);
// 					val_arr.push_back(v);
// 				}
// 			}

// 			/* self */
// 			double v = 0;
// 			for (int k = width; k > 0; k--){
// 				v -= coeffs[width-1][k] * (A[ii-k][jj]  + A[ii+k][jj]) / hhx;
// 				v -= coeffs[width-1][k] * (A[ii][jj-k]  + A[ii][jj+k]) / hhy;
// 				v += v_x/hx;
// 				v += v_y/hy;
// 			}

// 			offset++;
// 			col_idx.push_back(idx);
// 			val_arr.push_back(v);

// 			/* right neighbor */
// 			for (int k = 1; k <= width; k++){
// 				if (i+k <= Nx-1) {
// 					int nbrIdx = (i+k) + j * Nx;
// 					double v = coeffs[width-1][k] * A[ii+k][jj] / hhx;
// 					offset++;
// 					col_idx.push_back(nbrIdx);
// 					val_arr.push_back(v);
// 				}
// 			}

// 			/* up neighbor */
// 			for (int k = 1; k <= width; k++){
// 				if (j+k <= Ny-1) {
// 					int nbrIdx = i + (j+k) * Nx;
// 					double v = coeffs[width-1][k] * A[ii][jj+k] / hhy;
// 					offset++;
// 					col_idx.push_back(nbrIdx);
// 					val_arr.push_back(v);
// 				}
// 			}
// 		}
// 	}
// 	// Put nnz at the end
// 	row_ptr.push_back(offset);

// 	return SMat(val_arr,col_idx,row_ptr, Ny*Nx, Ny*Nx);
// }



SMat buildLaplacian(int Nx, int Ny, int Nz, std::vector<std::vector<std::vector<double> > > A, int width, double hx, double hy, double hz) {
	// A is coefficient field of size Nx+2*width by Ny+2*width by Nz +2*width

	/* 3D problem */
	int dim = 3;
	double hhx = hx * hx, hhy = hy * hy, hhz = hz * hz;
	/* Assert for now that width is 1 or 2  or 3*/
	assert (width == 1 || width == 2 || width == 3);

	std::vector<std::vector<double> > coeffs(3);
	/* width 1 */
	coeffs[0].resize(2);
	coeffs[0][0] = 0; // has to just be the sum of the other guys
	coeffs[0][1] = -1;
	/* width 2 */
	coeffs[1].resize(3);
	coeffs[1][0] = 0; // has to just be the sum of the other guys
	coeffs[1][1] = -4.0/3.0;
	coeffs[1][2] = 1.0/12.0;
	/* width 3 */
	coeffs[2].resize(4);
	coeffs[2][0] = 0; // has to just be the sum of the other guys
	coeffs[2][1] = -3.0/2.0;
	coeffs[2][2] = 3.0/20.0;
	coeffs[2][3] = -1.0/90.0;


	/* We build the Laplacian in coordinate format and then convert to CSR */
	IdxVec row_ptr, col_idx;
	SclVec val_arr;

	int nnz_est = (width * 2 * dim + 1) * Nx * Ny * Nz;

	row_ptr.reserve(nnz_est);
	col_idx.reserve(nnz_est);
	val_arr.reserve(nnz_est);

	int offset = 0;
	for (int z = 0; z < Nz; z++) {
		for (int j = 0; j < Ny; j++) {
			for (int i = 0; i < Nx; i++){
				// Row starts at this idx
				row_ptr.push_back(offset);

				/* idxs relative to coefficient grid (one width unit larger in each direction)*/
				int ii = i+width;
				int jj = j+width;
				int zz = z+width;

				/* linear index */
				int idx = i + j * Nx + z * Nx * Ny;

				/* MUST go in order of column index */

				/* back neighbors */
				for (int k = width; k > 0; k--) {
					if (z-k >= 0) {
						int nbrIdx = i + j * Nx + (z-k) * Nx * Ny;
						double ahalf = 0.5*(A[ii][jj][zz-k] + A[ii][jj][zz-k+1]);
						double v = coeffs[width-1][k] * ahalf/ hhz;
						offset++;
						col_idx.push_back(nbrIdx);
						val_arr.push_back(v);
					}
				}
				/* down neighbors */
				for (int k = width; k > 0; k--) {
					if (j-k >= 0) {
						int nbrIdx = i + (j-k) * Nx + z * Nx * Ny;
						double ahalf = 0.5*(A[ii][jj-k][zz] + A[ii][jj-k+1][zz]);
						double v = coeffs[width-1][k] * ahalf / hhy;
						offset++;
						col_idx.push_back(nbrIdx);
						val_arr.push_back(v);
					}
				}
				/* left neighbor */
				for (int k = width; k > 0; k--){
					if (i-k >= 0) {
						int nbrIdx = (i-k) + j * Nx + z * Nx * Ny;
						double ahalf = 0.5*(A[ii-k][jj][zz] + A[ii-k+1][jj][zz]);
						double v = coeffs[width-1][k] * ahalf / hhx;
						offset++;
						col_idx.push_back(nbrIdx);
						val_arr.push_back(v);
					}
				}

				/* self */
				double v = 0;
				for (int k = width; k > 0; k--){
					double ahalf1 = 0.5*(A[ii-k][jj][zz] + A[ii-k+1][jj][zz]);
					double ahalf2 = 0.5*(A[ii+k][jj][zz] + A[ii+k-1][jj][zz]);
					double ahalf3 = 0.5*(A[ii][jj-k][zz] + A[ii][jj-k+1][zz]);
					double ahalf4 = 0.5*(A[ii][jj+k][zz] + A[ii][jj+k-1][zz]);
					double ahalf5 = 0.5*(A[ii][jj][zz-k] + A[ii][jj][zz-k+1]);
					double ahalf6 = 0.5*(A[ii][jj][zz+k] + A[ii][jj][zz+k-1]);
					v -= coeffs[width-1][k] * (ahalf1 + ahalf2) / hhx;
					v -= coeffs[width-1][k] * (ahalf3 + ahalf4) / hhy;
					v -= coeffs[width-1][k] * (ahalf5 + ahalf6) / hhz;
				}

				offset++;
				col_idx.push_back(idx);
				val_arr.push_back(v);

				/* right neighbor */
				for (int k = 1; k <= width; k++){
					if (i+k <= Nx-1) {
						int nbrIdx = (i+k) + j * Nx + z * Nx * Ny;
						double ahalf = 0.5*(A[ii+k][jj][zz] + A[ii+k-1][jj][zz]);
						double v = coeffs[width-1][k] * ahalf / hhx;
						offset++;
						col_idx.push_back(nbrIdx);
						val_arr.push_back(v);
					}
				}

				/* up neighbor */
				for (int k = 1; k <= width; k++){
					if (j+k <= Ny-1) {
						int nbrIdx = i + (j+k) * Nx + z * Nx * Ny;
						double ahalf = 0.5*(A[ii][jj+k][zz] + A[ii][jj+k-1][zz]);
						double v = coeffs[width-1][k] * ahalf / hhy;
						offset++;
						col_idx.push_back(nbrIdx);
						val_arr.push_back(v);
					}
				}
				/* front neighbor */
				for (int k = 1; k <= width; k++){
					if (z+k <= Nz-1) {
						int nbrIdx = i + j * Nx + (z+k) * Nx * Ny;
						double ahalf = 0.5*(A[ii][jj][zz+k] + A[ii][jj][zz+k-1]);
						double v = coeffs[width-1][k] * ahalf / hhz;
						offset++;
						col_idx.push_back(nbrIdx);
						val_arr.push_back(v);
					}
				}
			}
		}
	}
	// Put nnz at the end
	row_ptr.push_back(offset);

	return SMat(val_arr,col_idx,row_ptr, Ny*Nx*Nz, Ny*Nx*Nz);
}


#endif // ifndef BUILD_LAPLACIAN