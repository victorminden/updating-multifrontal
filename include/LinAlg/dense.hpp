/*
   Name:   dense.hpp
   Author: Victor Minden and Anil Damle
   Purpose: Implement a dense matrix class with BLAS routines
   Comments: Based off of code written by Lexing Ying and Jack Poulson.  For double or dcomplex
 */

#pragma once
#ifndef LINALG_DENSE_HPP
#define LINALG_DENSE_HPP 1

#include "blas.hpp"
#include "lapack.hpp"

#include "vec.hpp"

#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <vector>
#include <assert.h>
#include <cmath>
#include <complex>
#include <string.h>
#include <utility>
#include <algorithm>

typedef std::vector<int> IdxVec;

namespace LinAlg {
	template <typename U> class Dense;
	template <typename U> class Sparse;
	template <typename U> class Vec;
	template <typename U> class TransposeView_;


	/* Interpolative decomposition */
	// Note: we return with the last parameters by reference, differing from our usual style

	void InterpDecomp(Dense<double> A, Dense<double>& T, IdxVec& sk, IdxVec& rd, double tol);
	void InterpDecomp(Dense<dcomplex>  A, Dense<dcomplex>& T, IdxVec& sk, IdxVec& rd, double tol);


	// TODO: Currently we provide no additional functionality for symmetric or Hermitian matrices

	template <class Scalar>
	class Dense {
		public:
		friend class TransposeView_<Scalar>;
		// Don't use this except for debugging
		friend void Inv(Dense<Scalar>&);

		// LU to be used for unsymmetric
		friend void LU(Dense<Scalar>&);
		friend void LUSolve(Dense<Scalar>& , Dense<Scalar>&);
		friend void LUSolveVec(Dense<Scalar>& , Vec<Scalar>&);
		friend void LUSolveT(Dense<Scalar>& , Dense<Scalar>&);

		// Cholesky to be used for SPD
		friend void Chol(Dense<Scalar>&);
		friend void CholSolve(Dense<Scalar>& , Dense<Scalar>&);
		friend void CholSolveVec(Dense<Scalar>& , Vec<Scalar>&);
		friend void CholSolveT(Dense<Scalar>& , Dense<Scalar>&);

		friend void InterpDecomp(Dense<Scalar> , Dense<Scalar>& , IdxVec& , IdxVec& , double );
		friend void transpose(Dense<Scalar>&, Dense<Scalar>&);
		friend void GEMM(double, const Dense<double>&, char, const Dense<double>&, char, double, Dense<double>&);

		// Constructor
		Dense(int m=0, int n=0);
		// Copy constructor
		// Note: deep copy
		Dense(const Dense& A);
		// Submat constructor
		Dense(const Dense& A, const IdxVec& rows, const IdxVec& cols);

		void resize(int m, int n);

		// Overloaded operators
		// Note: deep copy
		Dense& operator=(Dense A);

		Dense& operator+=(const Dense& A);
		Dense& operator-=(const Dense& A);
		Dense& operator*=(const Scalar& alpha);
		Dense& operator/=(const Scalar& alpha);

		// The true workhorses
		// GEMM call with C=0 initially
		Dense operator*(const Dense& B) const;
		Dense operator*(const TransposeView_<Scalar>& B) const;
		Vec<Scalar> operator*(const Vec<Scalar>& v) const;
		/* to do matrix solve, do A.inv() * B */

		// Return views in-place
		// Transpose view
		TransposeView_<Scalar> T();
		// Hermitian view
		TransposeView_<Scalar> H();


		double norm() const;

		// Indexing
		Scalar& operator()(int i, int j);
		// Not sure we need const indexing
		const Scalar& operator()(int i, int j) const;

		// Utility indexing
		Dense operator()(const IdxVec& rows, const IdxVec& cols);

		// Printing
		void print() const;
		int m() const { return m_;}
		int n()const { return n_;}


		// Swap functionality as described at
		// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
		void swap(Dense<Scalar>& that);

		private:
		// Number of rows, columns
		int m_, n_, size_;
		// The numerical data in column-major form
		std::vector<Scalar> data_;
		IdxVec ipiv_;
		// Gross constants that we need for BLAS
		Scalar one_, m_one_, zero_;
		void constants_();


	}; // class Dense

	// in-place inverse
	void Inv(Dense<double>& A) {
		assert(A.m_ == A.n_);
		IdxVec ipiv(A.n_);
		int info = 0;

		LAPACK(dgetrf)(&A.m_, &A.n_, &A.data_[0], &A.m_, &ipiv[0], &info);
		assert(info==0);

		int lwork = A.m_ * A.m_;
		std::vector<double> work(lwork);

		LAPACK(dgetri)(&A.m_, &A.data_[0], &A.m_, &ipiv[0], &work[0], &lwork, &info);
		assert(info==0);
	}

	void Inv(Dense<dcomplex>& A) {
		assert(A.m_ == A.n_);
		IdxVec ipiv(A.n_);
		int info = 0;

		LAPACK(zgetrf)(&A.m_, &A.n_, &A.data_[0], &A.m_, &ipiv[0], &info);
		assert(info==0);

		int lwork = A.m_ * A.m_;
		std::vector<dcomplex> work(lwork);

		LAPACK(zgetri)(&A.m_, &A.data_[0], &A.m_, &ipiv[0],&work[0], &lwork, &info);
		assert(info==0);
	}

	// LU and solve
	void transpose(Dense<double>& A, Dense<double>& AT){
		// find a way to avoid using this, or make it more efficient
  		AT.resize(A.n_, A.m_);
		for (int j = 0; j < A.n_; j++) {
  			for (int i = 0; i < A.m_; i++) {
  				AT(j,i) = A(i,j);
  			}
  		}

	}


// LU
	void LU(Dense<double>& A) {
		#ifdef DEBUG
			assert(A.m_ == A.n_);
		#endif
		(A.ipiv_).resize(A.m_);
		int info = 0;
		LAPACK(dgetrf)(&A.m_, &A.n_, &A.data_[0], &A.m_, &A.ipiv_[0], &info);
		assert(info==0);
	}

	void LUSolve(Dense<double>& A, Dense<double>& B) {
		#ifdef DEBUG
			assert(A.m_ == A.n_);
			assert(A.m_ == B.m_);
		#endif
		int info = 0;
		char NT = 'N';

		LAPACK(dgetrs)(&NT, &A.m_, &B.n_, &A.data_[0], &A.m_, &A.ipiv_[0], &B.data_[0], &B.m_, &info);
		assert(info==0);
	}

	void LUSolveVec(Dense<double>& A, Vec<double>& b) {
		#ifdef DEBUG
			assert(A.m_ == A.n_);
			assert(A.m_ == b.n_);
		#endif
		int info = 0;
		int one = 1;
		char NT = 'N';
		LAPACK(dgetrs)(&NT, &A.m_, &one, &A.data_[0], &A.m_, &A.ipiv_[0], &b.data_[0], &b.n_, &info);
		assert(info==0);
	}

	void LUSolveT(Dense<double>& A, Dense<double>& B) {
		// solve X A = B
		#ifdef DEBUG
			assert(A.m_ == A.n_);
			assert(A.m_ == B.n_);
		#endif
		int info = 0;
		char NT = 'T';
		Dense<double> X;
		transpose(B,X);
		LAPACK(dgetrs)(&NT, &A.m_, &X.n_, &A.data_[0], &A.m_, &A.ipiv_[0], &X.data_[0], &X.m_, &info);
		assert(info==0);
		transpose(X,B);
	}


// Chol
	void Chol(Dense<double>& A) {
		#ifdef DEBUG
			assert(A.m_ == A.n_);
		#endif
		int info = 0;
		char uplo = 'U';
		LAPACK(dpotrf)(&uplo, &A.n_, &A.data_[0], &A.m_, &info);
		// std::cout << "Info: " << info << std::endl;
		assert(info==0);
	}

	void CholSolve(Dense<double>& A, Dense<double>& B) {
		#ifdef DEBUG
			assert(A.m_ == A.n_);
			assert(A.m_ == B.m_);
		#endif
		int info = 0;
		char uplo = 'U';
		LAPACK(dpotrs)(&uplo, &A.m_, &B.n_, &A.data_[0], &A.m_, &B.data_[0], &B.m_, &info);
		assert(info==0);
	}

	void CholSolveVec(Dense<double>& A, Vec<double>& b) {
		#ifdef DEBUG
			assert(A.m_ == A.n_);
			assert(A.m_ == b.n_);
		#endif
		int info = 0;
		int one = 1;
		char uplo = 'U';
		LAPACK(dpotrs)(&uplo, &A.m_, &one, &A.data_[0], &A.m_, &b.data_[0], &b.n_, &info);
		assert(info==0);
	}

	void CholSolveT(Dense<double>& A, Dense<double>& B) {
		// solve X A = B
		#ifdef DEBUG
			assert(A.m_ == A.n_);
			assert(A.m_ == B.n_);
		#endif
		int info = 0;
		char uplo = 'U';
		Dense<double> X;
		transpose(B,X);
		LAPACK(dpotrs)(&uplo, &A.m_, &X.n_, &A.data_[0], &A.m_, &X.data_[0], &X.m_, &info);
		assert(info==0);
		transpose(X,B);
	}


	void GEMM(double alpha,
			  const Dense<double> &A, char transa, const Dense<double> &B, char transb, double beta, Dense<double> &C) {
	   //  C := alpha*op( A )*op( B ) + beta*C,

	   // where  op( X ) is one of

	   //  op( X ) = X   or   op( X ) = X**T,

	   // alpha and beta are scalars, and A, B and C are matrices, with op( A )
	   // an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
		int n,m,k;
		if (transa == 'N' || transa == 'n') {
			m = A.m();
			k = A.n();
		} else {
			m = A.n();
			k = A.m();
		}

		if (transb == 'N' || transb == 'n') {
			n = B.n();
		} else {
			n = B.m();
		}

		if (m != 0 && n != 0 && k != 0) {
			BLAS(dgemm)(&transa, &transb, &m, &n, &k, &alpha, &A.data_[0], &A.m_, &B.data_[0], &B.m_, &beta, &C.data_[0], &C.m_);
		}

	}


	/* A utility class for constructing transpose views of things */
	template <class Scalar>
	class TransposeView_ {
	public:
		friend class Dense<Scalar>;
		// Invalidate all operators -- you can't do much with a view except apply or look
		TransposeView_() {}
		TransposeView_(const Dense<Scalar>& A, char transA);
		~TransposeView_() {}
		TransposeView_& operator=(TransposeView_ A) {}


		Dense<Scalar> operator*(const Dense<Scalar>& B) const;
		Dense<Scalar> operator*(const TransposeView_& B) const;
		Vec<Scalar> operator*(const Vec<Scalar>& v) const;


		void print() const;
	private:
		//
		const Dense<Scalar>* mat_;
		const Scalar* buffer_;
		char transA_;
		Scalar one_, m_one_, zero_;

	}; //TransposeView_




}; // namespace LinAlg
#include "../../src/LinAlg/dense.cpp"
#endif // ifndef LINALG_DENSE_HPP

