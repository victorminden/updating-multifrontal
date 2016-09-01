/*
   Name:   sparse.hpp
   Author: Victor Minden
   Purpose: Implement a bare-minimum CSR sparse matrix class
   Comments: Based off of code written by Lexing Ying and Jack Poulson.  For double or dcomplex.
 */

#pragma once
#ifndef LINALG_SPARSE_HPP
#define LINALG_SPARSE_HPP 1

#include "blas.hpp"
#include "vec.hpp"
#include "dense.hpp"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <complex>


typedef std::vector<int> IdxVec;


namespace LinAlg {

	template <typename U> class Dense;
	template <typename U> class Vec;

	template <class Scalar>
	class Sparse {
	public:
		// Note: BROKE THE RULE OF THREE.  There is no copy or assignment
		Sparse() {}
		Sparse(const std::string& filename);
		Sparse(const std::vector<Scalar>& val, const IdxVec& col_ind, const IdxVec& row_ptr, int m, int n);
		Sparse(std::vector<Scalar>& val, const IdxVec);
		Sparse(const Sparse& other);

		// Operator overloading
		void operator=(Sparse other) {
			m_ = other.m_;
			n_ = other.n_;
			nnz_ = other.nnz_;
			val_ = other.val_;
			col_ind_ = other.col_ind_;
			row_ptr_ = other.row_ptr_;
		}
		Vec<Scalar> operator*(const Vec<Scalar>& v) const;

		// Indexing
		const Scalar operator()(int i, int j) const;
		int m() { return m_; }
		int n() { return n_; }
		int nnz() { return nnz_; }
		Scalar val(int i) { return val_[i]; }
		int col_ind(int i) { return col_ind_[i]; }
		int row_ptr(int i) { return row_ptr_[i]; }

		// Dense subblocks (utility indexing)
		Dense<Scalar> operator()(const IdxVec& rows, const IdxVec& cols) const;
		// Vertical stack A(rows,cols); A(cols,rows)^T
		Dense<Scalar> operator()(const IdxVec& rows, const IdxVec& cols, bool vstack) const;

		// Norm
		double norm() const;

		// Printing
		void print() const;
		void printDense() const;
		void printMatlab() const;

	private:
		int m_, n_, nnz_;
		// CSR internals
		std::vector<Scalar> val_;
		IdxVec col_ind_;
		IdxVec row_ptr_;

	}; // class Sparse





}; //namespace LinAlg
#include "../../src/LinAlg/sparse.cpp"

#endif //ifndef LINALG_SPARSE_HPP