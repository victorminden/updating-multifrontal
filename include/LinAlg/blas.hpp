/* Name:   blas.hpp
   Author: Victor Minden
   Purpose: Bring in all the BLAS routines
   Comments: Based off of code written by Jack Poulson and Lexing Ying in HIFDE3D
 */

#pragma once
#ifndef EXTERN_BLAS_HPP
#define EXTERN_BLAS_HPP 1

// Deal with the trailing underscore if it exists
#if defined(BLAS_POST)
#define BLAS(name) name ## _
#else
#define BLAS(name) name
#endif // if defined(BLAS_POST)

#include <stdexcept>
#include <complex>

typedef std::complex<double> dcomplex;

extern "C"{
/* Scalar mult */
void BLAS(dscal)
( const int* n, const double* alpha, const double* x, const int* incx);
void BLAS(zscal)
( const int* n, const dcomplex* alpha, const dcomplex* x, const int* incx);

/* AXPY */
void BLAS(daxpy)
( const int* n, const double* alpha, const double* x, const int* incx, const double* y, const int* incy );
void BLAS(zaxpy)
( const int* n, const dcomplex* alpha, const dcomplex* x, const int* incx, const dcomplex* y, const int* incy );

/* NRM2 */
double BLAS(dnrm2)
( const int* n, const double* x, const int* incx );
double BLAS(dznrm2)
( const int* n, const dcomplex* x, const int* incx );

/* General matrix multiplication */
void BLAS(dgemm)
( const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* A, const int* lda,
                       const double* B, const int* ldb,
  const double* beta,        double* C, const int* ldc );
  void BLAS(zgemm)
( const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const dcomplex* alpha, const dcomplex* A, const int* lda,
                         const dcomplex* B, const int* ldb,
  const dcomplex* beta,        dcomplex* C, const int* ldc );

/* General matrix-vector multiplication */
void BLAS(dgemv)
( const char* trans, const int* m, const int* n,
  const double* alpha, const double* A, const int* lda,
                       const double* x, const int* incx,
  const double* beta,        double* y, const int* incy );

void BLAS(zgemv)
( const char* trans, const int* m, const int* n,
  const dcomplex* alpha, const dcomplex* A, const int* lda,
                         const dcomplex* x, const int* incx,
  const dcomplex* beta,        dcomplex* y, const int* incy );

} // extern "C"


#endif // ifndef EXTERN_BLAS_HPP