/* Name:   lapack.hpp
   Author: Victor Minden and Anil Damle
   Purpose: Bring in all the LAPACK routines
   Comments: Based off of code written by Jack Poulson and Lexing Ying in HIFDE3D
 */

#pragma once
#ifndef EXTERN_LAPACK_HPP
#define EXTERN_LAPACK_HPP 1

// Deal with the trailing underscore if it exists
#if defined(LAPACK_POST)
#define LAPACK(name) name ## _
#else
#define LAPACK(name) name
#endif // if defined(LAPACK_POST)

#include <stdexcept>
#include <complex>

typedef std::complex<double> dcomplex;

extern "C"{

/* Pivoted QR factorization */
void LAPACK(dgeqp3)
( const int* m, const int* n, double* A,
  const int* lda, int* jpvt, double* tau,
  double* work, const int* lwork, int* info);

void LAPACK(zgeqp3)
( const int* m, const int* n, dcomplex* A,
  const int* lda, int* jpvt, dcomplex* tau,
  dcomplex* work, const int* lwork, double* rwork,
  int* info);

/* Triangular solve */
void LAPACK(dtrtrs)
( const char* uplo, const char* trans, const char* diag,
  const int* n, const int* nrhs, double* A, const int* lda,
  double* b, const int* ldb, int* info);

void LAPACK(ztrtrs)
( const char* uplo, const char* trans, const char* diag,
  const int* n, const int* nrhs, dcomplex* A, const int* lda,
  dcomplex* b, const int* ldb, int* info);


/* PLU */
void LAPACK(dgetrf)
( const int* m, const int* n,
  double* A, const int* lda,
  int* ipiv, int* info );

void LAPACK(zgetrf)
( const int* m, const int* n,
  dcomplex* A, const int* lda,
  int* ipiv, int* info );


/* solve PLU */
void LAPACK(dgetrs)
( const char* trans, const int* n,
  const int* nrhs, double* A, const int* lda,
  int* ipiv, double* b,  const int* ldb,
  int* info );

void LAPACK(zgetrs)
( const char* trans, const int* n,
  const int* nrhs, dcomplex* A, const int* lda,
  int* ipiv, dcomplex* b,  const int* ldb,
  int* info );

// Note: currently no chol or LDL for complex

/* Cholesky */
void LAPACK(dpotrf)
( const char* uplo, const int* n, double* A, const int* lda, int* info);

void LAPACK(dpotrs)
( const char* uplo, const int* n,
  const int* nrhs, double* A, const int* lda,
  double* b,  const int* ldb, int* info );

/* LDL */

void LAPACK(dsytrf)
( const char* uplo, const int* n, double* A, const int* lda,
  int* ipiv, double* work, const int* lwork,  int* info);

void LAPACK(dsytrs)
( const char* uplo, const int* n,
  const int* nrhs, double* A, const int* lda,
  int* ipiv, double* b,  const int* ldb, int* info );

/* Form explicit inverse from PLU */

void LAPACK(dgetri)
( const int* n,
  double* A, const int* lda,
  const int* ipiv,
  double* work, const int* lwork,
  int* info );


void LAPACK(zgetri)
( const int* n,
  dcomplex* A, const int* lda,
  const int* ipiv,
  dcomplex* work, const int* lwork,
  int* info );

} // extern "C"
#endif // ifnder EXTERN_LAPACK_HPP