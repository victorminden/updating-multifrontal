/*
   Name:   vec.hpp
   Author: Victor Minden
   Purpose: Implement a vector class with BLAS routines
   Comments: For double or dcomplex.  Probably mostly used to wrap raw pointers?  Based on code by Lexing Ying.
 */

#pragma once
#ifndef LINALG_VEC_HPP
#define LINALG_VEC_HPP 1


#include "blas.hpp"

#include <stdexcept>
#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <complex>

typedef std::vector<int> IdxVec;
namespace LinAlg {
	template <typename U> class Vec;
	template <typename U> class Sparse;
	template <typename U> class Dense;
	template <typename U> class TransposeView_;

 	void LUSolveVec(Dense<double>& A, Vec<double>& b);
 	void LUSolveVec(Dense<dcomplex>& A, Vec<dcomplex>& b);

 	void CholSolveVec(Dense<double>& A, Vec<double>& b);

	template <class Scalar>
	class Vec {
		public:
		friend class Dense<Scalar>;
		friend class TransposeView_<Scalar>;
		friend class Sparse<Scalar>;
		friend void LUSolveVec(Dense<Scalar>& , Vec<Scalar>&);
		friend void CholSolveVec(Dense<Scalar>& , Vec<Scalar>&);
		// Constructor
		Vec() {}
		Vec(int n=0);
		// Copy constructor
		// Note: deep copy
		Vec(const Vec& v);
		// Subvec constructor
		Vec(const Vec& v, const IdxVec& idxs);

		// Operators
		Vec& operator=(Vec A);

		Vec& operator+=(const Vec& A);
		Vec& operator-=(const Vec& A);
		Vec& operator*=(const Scalar& alpha);
		Vec& operator/=(const Scalar& alpha);

		/* norm */
		double norm() const;

		void resize(int n);

		void set(const IdxVec& idxs, const Vec& vals);

		// Indexing
		Scalar& operator()(int i);
		const Scalar& operator()(int i) const;

		// Utility indexing
		Vec operator()(const IdxVec& idxs);

		// Printing
		void print() const;

		// Swap functionality as described at
		// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
		void swap(Vec<Scalar>& that);

		int n() const {return n_;}

		private:
		// Number of entries
		int n_;
		// The raw numerical data
		std::vector<Scalar> data_;
		// Gross constants that we need for BLAS
		Scalar one_, m_one_, zero_;
		void constants_();
	}; // class Vec

}; // namespace LinAlg
#include "../../src/LinAlg/vec.cpp"

#endif // ifndef LINALG_VEC_HPP