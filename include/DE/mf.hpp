/*
   Name:   mf.hpp
   Author: Victor Minden
   Purpose: A multifrontal method for 2D or 3D sparse structured matrices (i.e., pentadiagonal, etc.)
   Comments: Based on code by Austin Benson, Jack Poulson, and Lexing Ying
   			 We use a combined grid indexing scheme to count boxes and edges on the same grid
             Boxes have even indexes in the grid, edges have odd indexes

             This code allows for unsymmetric input as long as it is pattern-symmetric, but
             reserves the right to fail or give bad results due to ill-conditioned or singular
             pivot blocks (i.e., if you want to try multifrontal ordering, go ahead, but it is essentially
             LU without pivoting )

             TODO: write a symmetric one too to check timing ?

 */

#ifndef TREEFACTOR_MF_HPP
#define TREEFACTOR_MF_HPP 1

#include "tree2d_common.hpp"
#include "tree3d_common.hpp"

#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <assert.h>
#include <cmath>

#ifdef HASCPP11TIMING
#include <chrono>
#endif



typedef std::vector<int> IdxVec;

namespace TreeFactor2D {

	template <class Scalar>
	class MF2D {
		struct Box {
			IdxVec intDOFs;            // eliminate all interior,
			IdxVec bdDOFs;	 	        // keep boundary DOFs
			LinAlg::Dense<Scalar> A22;        // matrix restricted to interactions
    		LinAlg::Dense<Scalar> A22Inv;    // explicit inverse of A_22
    		LinAlg::Dense<Scalar> XL;         // A_12 * A_22_inv
    		LinAlg::Dense<Scalar> XR;         // A_22_inv * A_21
    		LinAlg::Dense<Scalar> SchurComp;  // -A_12 * A_22_inv*A_21
    		LinAlg::Dense<Scalar> A11post;

		}; // struct Box

		struct Edge {
			IdxVec DOFs;
			LinAlg::Dense<Scalar> A11; // self interactions

		}; // struct Edge

		public:
	    /* Constructor */
		/* Sparse matrix, number of points in x direction,
		number of points in y direction, separator width, max length of a leaf box side */
		MF2D(const LinAlg::Sparse<Scalar>& A, int Nx, int Ny, int width=1, int maxSideLength=4);
		/* TODO: should change constructor to take CSR vecs */

		/* Both return and in-place versions for optimality */
		LinAlg::Vec<Scalar> Apply(const LinAlg::Vec<Scalar>& x);
		//void Apply(const LinAlg::Vec<Scalar>& x, LinAlg::Vec<Scalar>& b);

		LinAlg::Vec<Scalar> Solve(const LinAlg::Vec<Scalar>& b);
		//void Solve(const LinAlg::Vec<Scalar>& b, LinAlg::Vec<Scalar>& x);

		/* The core of the class: creating a multifrontal factorization */
		void Factor();

		void Update(const LinAlg::Sparse<Scalar>& A, std::set<int> modifiedDOFs);

		private:
		#ifdef HASCPP11TIMING
		int durationLA_;
		int durationData_;
		#endif

		/* Print coordinates of DOFs*/
		void Diagnostic_(IdxVec dofs) {
			for (int i = 0; i < dofs.size(); i++) {
				int row, col;
				row = dofs[i] % Nx_;
				col = (dofs[i] - row) / Nx_;
				std::cout << row << ' ' << col << std::endl;
			}
			std::cout << std::endl;
		}
		/* Factor the boxes on a given level */
		void FactorUpperLevel_(int ell);
		void FactorLeafLevel_();

		/* Internal sparse matrix */
		LinAlg::Sparse<Scalar> A_;

		/* Number of points in each direction */
		int Nx_, Ny_;

		/* Separator width (i.e., nearest neighbors is 1) */
		int width_;
		int boxWidthXBottom_, boxWidthYBottom_;

		/* number of levels in tree */
		int levels_;


		/* Find the points that belong to a parent edge that don't belong to any child */
		inline IdxVec getEdgeCenterPoints_(int edgeIdx, int ell);
		/* Find the points that belong to a parent box that don't belong to any child */
		inline IdxVec getBoxCenterPoints_(int boxIdx, int ell);

		void updateEdgeData_(int ell);
		void updateBoxData_(int ell);

		/* A <- A + B only for indices of B that also exist in A */
		void ScatterAdd_(LinAlg::Dense<Scalar>& A, const IdxVec& bigDOFs, const LinAlg::Dense<Scalar>& B, const IdxVec& smallDOFs);
		void ScatterReplace_(LinAlg::Dense<Scalar>& A, const IdxVec& bigDOFsRow, const IdxVec& bigDOFsCol, const LinAlg::Dense<Scalar>& B, const IdxVec& smallDOFs);
		void ScatterReplace_(LinAlg::Dense<Scalar>& A, const IdxVec& bigDOFs, const LinAlg::Dense<Scalar>& B, const IdxVec& smallDOFs);

		/* A has been extended to include some newDOFs, pull those out of the big matrix */
		void AddNewInteractions_(LinAlg::Dense<Scalar>& A, const IdxVec& totalDOFs, const IdxVec& newDOFs);
		void AddNewRowInteractions_(LinAlg::Dense<Scalar>& A, const IdxVec& rowDOFs, const IdxVec& colsDOFs, const IdxVec& newDOFs);
		void AddNewColInteractions_(LinAlg::Dense<Scalar>& A, const IdxVec& rowDOFs, const IdxVec& colsDOFs, const IdxVec& newDOFs);
		/* The factor information */
		std::vector< std::vector<Box> >  boxData_;
		std::vector< std::vector<Edge> > edgeData_;

		/* The arrays of DOFs over which we will loop */
		std::set<int> markedBoxes_;
		std::set<int> markedEdges_;


	}; // class MF2D

}; // namespace TreeFactor2D



namespace TreeFactor3D {
	template <class Scalar>
	class MF3D {
		struct Box {
			IdxVec intDOFs;            // eliminate all interior,
			IdxVec bdDOFs;	 	        // keep boundary DOFs
			LinAlg::Dense<Scalar> A22;        // matrix restricted to interactions
    		LinAlg::Dense<Scalar> A22Inv;    // explicit inverse of A_22
    		LinAlg::Dense<Scalar> XL;         // A_12 * A_22_inv
    		LinAlg::Dense<Scalar> XR;         // A_22_inv * A_21
    		LinAlg::Dense<Scalar> SchurComp;  // -A_12 * A_22_inv*A_21
    		LinAlg::Dense<Scalar> A11post;

		}; // struct Box

		struct Face {
			IdxVec DOFs;
			LinAlg::Dense<Scalar> A11; // self interactions

		}; // struct Face

		public:
	    /* Constructor */
		/* Sparse matrix, number of points in x direction,
		number of points in y direction, separator width, max length of a leaf box side */
		MF3D(const LinAlg::Sparse<Scalar>& A, int Nx, int Ny, int Nz, int width=1, int maxSideLength=4);
		/* TODO: should change constructor to take CSR vecs */

		/* Both return and in-place versions for optimality */
		LinAlg::Vec<Scalar> Apply(const LinAlg::Vec<Scalar>& x);
		//void Apply(const LinAlg::Vec<Scalar>& x, LinAlg::Vec<Scalar>& b);

		LinAlg::Vec<Scalar> Solve(const LinAlg::Vec<Scalar>& b);
		//void Solve(const LinAlg::Vec<Scalar>& b, LinAlg::Vec<Scalar>& x);

		/* The core of the class: creating a multifrontal factorization */
		void Factor();

		void Update(const LinAlg::Sparse<Scalar>& A, std::set<int> modifiedDOFs);

		void help();

		private:
		#ifdef HASCPP11TIMING
		int durationLA_;
		int durationData_;
		#endif

		/* Factor the boxes on a given level */
		void FactorUpperLevel_(int ell);
		void FactorLeafLevel_();

		/* Internal sparse matrix */
		LinAlg::Sparse<Scalar> A_;

		/* Number of points in each direction */
		int Nx_, Ny_, Nz_;

		/* Separator width (i.e., nearest neighbors is 1) */
		int width_;
		int boxWidthXBottom_, boxWidthYBottom_, boxWidthZBottom_;

		/* number of levels in tree */
		int levels_;

		/* Find the points that belong to a parent face that don't belong to any child */
		inline IdxVec getFaceCenterPoints_(int faceIdx, int ell);
		/* Find the points that belong to a parent box that don't belong to any child */
		inline IdxVec getBoxCenterPoints_(int boxIdx, int ell);

		void updateFaceData_(int ell);
		void updateBoxData_(int ell);

		/* A <- A + B only for indices of B that also exist in A */
		void ScatterAdd_(LinAlg::Dense<Scalar>& A, const IdxVec& bigDOFs, const LinAlg::Dense<Scalar>& B, const IdxVec& smallDOFs);
		void ScatterReplace_(LinAlg::Dense<Scalar>& A, const IdxVec& bigDOFsRow, const IdxVec& bigDOFsCol, const LinAlg::Dense<Scalar>& B, const IdxVec& smallDOFs);
		void ScatterReplace_(LinAlg::Dense<Scalar>& A, const IdxVec& bigDOFs, const LinAlg::Dense<Scalar>& B, const IdxVec& smallDOFs);

		/* A has been extended to include some newDOFs, pull those out of the big matrix */
		void AddNewInteractions_(LinAlg::Dense<Scalar>& A, const IdxVec& totalDOFs, const IdxVec& newDOFs);
		void AddNewRowInteractions_(LinAlg::Dense<Scalar>& A, const IdxVec& rowDOFs, const IdxVec& colsDOFs, const IdxVec& newDOFs);
		void AddNewColInteractions_(LinAlg::Dense<Scalar>& A, const IdxVec& rowDOFs, const IdxVec& colsDOFs, const IdxVec& newDOFs);
		/* The factor information */
		std::vector< std::vector<Box> >  boxData_;
		std::vector< std::vector<Face> > faceData_;

		/* The arrays of DOFs over which we will loop */
		std::set<int> markedBoxes_;
		std::set<int> markedFaces_;


	}; // class MF3D

}; // namespace TreeFactor3D

#include "../../src/DE/mf2d.cpp"
#include "../../src/DE/mf3d.cpp"
#endif // ifndef TREEFACTOR_MF_HPP