/*----------------------------------------------------------------------------
 * Copyright © Microsoft Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not 
 * use this file except in compliance with the License.  You may obtain a copy 
 * of the License at http://www.apache.org/licenses/LICENSE-2.0  
 * 
 * THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED 
 * WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, 
 * MERCHANTABLITY OR NON-INFRINGEMENT. 
 *
 * See the Apache Version 2.0 License for specific language governing 
 * permissions and limitations under the License.
 *---------------------------------------------------------------------------
 * 
 * BLAS levels 1,2,3 library header for AMP CBLAS.
 *
 * This file contains traditional C BLAS level 1,2,3 APIs.
 *
 *---------------------------------------------------------------------------*/
#ifndef AMPCBLAS_H
#define AMPCBLAS_H

#include <stddef.h>
#include "ampblas_defs.h"
#include "ampblas_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

//----------------------------------------------------------------------------
// Prototypes for level 1 BLAS functions (complex are recast as routines)
//----------------------------------------------------------------------------

float  ampblas_sdsdot(const int N, const float alpha, const float *X,
                      const int incX, const float *Y, const int incY);
double ampblas_dsdot(const int N, const float *X, const int incX, const float *Y,
                     const int incY);
float  ampblas_sdot(const int N, const float  *X, const int incX,
                    const float  *Y, const int incY);
double ampblas_ddot(const int N, const double *X, const int incX,
                    const double *Y, const int incY);

//
// Functions having prefixes Z and C only
//
void   ampblas_cdotu_sub(const int N, const void *X, const int incX,
                         const void *Y, const int incY, void *dotu);
void   ampblas_cdotc_sub(const int N, const void *X, const int incX,
                         const void *Y, const int incY, void *dotc);

void   ampblas_zdotu_sub(const int N, const void *X, const int incX,
                         const void *Y, const int incY, void *dotu);
void   ampblas_zdotc_sub(const int N, const void *X, const int incX,
                         const void *Y, const int incY, void *dotc);


//
// Functions having prefixes S D SC DZ
//
float  ampblas_snrm2(const int N, const float *X, const int incX);
float  ampblas_sasum(const int N, const float *X, const int incX);

double ampblas_dnrm2(const int N, const double *X, const int incX);
double ampblas_dasum(const int N, const double *X, const int incX);

float  ampblas_scnrm2(const int N, const void *X, const int incX);
float  ampblas_scasum(const int N, const void *X, const int incX);

double ampblas_dznrm2(const int N, const void *X, const int incX);
double ampblas_dzasum(const int N, const void *X, const int incX);


//
// Functions having standard 4 prefixes (S D C Z)
//
int ampblas_isamax(const int N, const float  *X, const int incX);
int ampblas_idamax(const int N, const double *X, const int incX);
int ampblas_icamax(const int N, const void   *X, const int incX);
int ampblas_izamax(const int N, const void   *X, const int incX);


//----------------------------------------------------------------------------
// Prototypes for level 1 BLAS routines
//----------------------------------------------------------------------------

// 
// Routines with standard 4 prefixes (s, d, c, z)
//
void ampblas_sswap(const int N, float *X, const int incX, 
                   float *Y, const int incY);
void ampblas_scopy(const int N, const float *X, const int incX, 
                   float *Y, const int incY);
void ampblas_saxpy(const int N, const float alpha, const float *X,
                   const int incX, float *Y, const int incY);

void ampblas_dswap(const int N, double *X, const int incX, 
                   double *Y, const int incY);
void ampblas_dcopy(const int N, const double *X, const int incX, 
                   double *Y, const int incY);
void ampblas_daxpy(const int N, const double alpha, const double *X,
                   const int incX, double *Y, const int incY);

void ampblas_cswap(const int N, void *X, const int incX, 
                   void *Y, const int incY);
void ampblas_ccopy(const int N, const void *X, const int incX, 
                   void *Y, const int incY);
void ampblas_caxpy(const int N, const void *alpha, const void *X,
                   const int incX, void *Y, const int incY);

void ampblas_zswap(const int N, void *X, const int incX, 
                   void *Y, const int incY);
void ampblas_zcopy(const int N, const void *X, const int incX, 
                   void *Y, const int incY);
void ampblas_zaxpy(const int N, const void *alpha, const void *X,
                   const int incX, void *Y, const int incY);


// 
// Routines with S and D prefix only
//
void ampblas_srotg(float *a, float *b, float *c, float *s);
void ampblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
void ampblas_srot(const int N, float *X, const int incX,
                  float *Y, const int incY, const float c, const float s);
void ampblas_srotm(const int N, float *X, const int incX,
                   float *Y, const int incY, const float *P);

void ampblas_drotg(double *a, double *b, double *c, double *s);
void ampblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
void ampblas_drot(const int N, double *X, const int incX,
                  double *Y, const int incY, const double c, const double  s);
void ampblas_drotm(const int N, double *X, const int incX,
                   double *Y, const int incY, const double *P);


// 
// Routines with S D C Z CS and ZD prefixes
//
void ampblas_sscal(const int N, const float alpha, float *X, const int incX);
void ampblas_dscal(const int N, const double alpha, double *X, const int incX);
void ampblas_cscal(const int N, const void *alpha, void *X, const int incX);
void ampblas_zscal(const int N, const void *alpha, void *X, const int incX);
void ampblas_csscal(const int N, const float alpha, void *X, const int incX);
void ampblas_zdscal(const int N, const double alpha, void *X, const int incX);


//----------------------------------------------------------------------------
// Prototypes for level 2 BLAS
//----------------------------------------------------------------------------

// 
// Routines with standard 4 prefixes (S, D, C, Z)
//
void ampblas_sgemv(const enum AMPBLAS_ORDER order,
                   const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N,
                   const float alpha, const float *A, const int lda,
                   const float *X, const int incX, const float beta,
                   float *Y, const int incY);
void ampblas_sgbmv(const enum AMPBLAS_ORDER order,
                   const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N,
                   const int KL, const int KU, const float alpha,
                   const float *A, const int lda, const float *X,
                   const int incX, const float beta, float *Y, const int incY);
void ampblas_strmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const float *A, const int lda, 
                   float *X, const int incX);
void ampblas_stbmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const int K, const float *A, const int lda, 
                   float *X, const int incX);
void ampblas_stpmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const float *Ap, float *X, const int incX);
void ampblas_strsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const float *A, const int lda, float *X,
                   const int incX);
void ampblas_stbsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const int K, const float *A, const int lda,
                   float *X, const int incX);
void ampblas_stpsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const float *Ap, float *X, const int incX);

void ampblas_dgemv(const enum AMPBLAS_ORDER order,
                   const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N,
                   const double alpha, const double *A, const int lda,
                   const double *X, const int incX, const double beta,
                   double *Y, const int incY);
void ampblas_dgbmv(const enum AMPBLAS_ORDER order,
                   const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N,
                   const int KL, const int KU, const double alpha,
                   const double *A, const int lda, const double *X,
                   const int incX, const double beta, double *Y, const int incY);
void ampblas_dtrmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const double *A, const int lda, 
                   double *X, const int incX);
void ampblas_dtbmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const int K, const double *A, const int lda, 
                   double *X, const int incX);
void ampblas_dtpmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const double *Ap, double *X, const int incX);
void ampblas_dtrsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const double *A, const int lda, double *X,
                   const int incX);
void ampblas_dtbsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const int K, const double *A, const int lda,
                   double *X, const int incX);
void ampblas_dtpsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const double *Ap, double *X, const int incX);

void ampblas_cgemv(const enum AMPBLAS_ORDER order,
                   const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N,
                   const void *alpha, const void *A, const int lda,
                   const void *X, const int incX, const void *beta,
                   void *Y, const int incY);
void ampblas_cgbmv(const enum AMPBLAS_ORDER order,
                   const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N,
                   const int KL, const int KU, const void *alpha,
                   const void *A, const int lda, const void *X,
                   const int incX, const void *beta, void *Y, const int incY);
void ampblas_ctrmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const void *A, const int lda, 
                   void *X, const int incX);
void ampblas_ctbmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const int K, const void *A, const int lda, 
                   void *X, const int incX);
void ampblas_ctpmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const void *Ap, void *X, const int incX);
void ampblas_ctrsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const void *A, const int lda, void *X,
                   const int incX);
void ampblas_ctbsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const int K, const void *A, const int lda,
                   void *X, const int incX);
void ampblas_ctpsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const void *Ap, void *X, const int incX);

void ampblas_zgemv(const enum AMPBLAS_ORDER order,
                   const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N,
                   const void *alpha, const void *A, const int lda,
                   const void *X, const int incX, const void *beta,
                   void *Y, const int incY);
void ampblas_zgbmv(const enum AMPBLAS_ORDER order,
                   const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N,
                   const int KL, const int KU, const void *alpha,
                   const void *A, const int lda, const void *X,
                   const int incX, const void *beta, void *Y, const int incY);
void ampblas_ztrmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const void *A, const int lda, 
                   void *X, const int incX);
void ampblas_ztbmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const int K, const void *A, const int lda, 
                   void *X, const int incX);
void ampblas_ztpmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const void *Ap, void *X, const int incX);
void ampblas_ztrsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const void *A, const int lda, void *X,
                   const int incX);
void ampblas_ztbsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const int K, const void *A, const int lda,
                   void *X, const int incX);
void ampblas_ztpsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag,
                   const int N, const void *Ap, void *X, const int incX);


// 
// Routines with S and D prefixes only
//
void ampblas_ssymv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const float alpha, const float *A,
                   const int lda, const float *X, const int incX,
                   const float beta, float *Y, const int incY);
void ampblas_ssbmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const int K, const float alpha, const float *A,
                   const int lda, const float *X, const int incX,
                   const float beta, float *Y, const int incY);
void ampblas_sspmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const float alpha, const float *Ap,
                   const float *X, const int incX,
                   const float beta, float *Y, const int incY);
void ampblas_sger(const enum AMPBLAS_ORDER order, const int M, const int N,
                  const float alpha, const float *X, const int incX,
                  const float *Y, const int incY, float *A, const int lda);
void ampblas_ssyr(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                  const int N, const float alpha, const float *X,
                  const int incX, float *A, const int lda);
void ampblas_sspr(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                  const int N, const float alpha, const float *X,
                  const int incX, float *Ap);
void ampblas_ssyr2(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const float alpha, const float *X,
                   const int incX, const float *Y, const int incY, float *A,
                   const int lda);
void ampblas_sspr2(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const float alpha, const float *X,
                   const int incX, const float *Y, const int incY, float *A);

void ampblas_dsymv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const double alpha, const double *A,
                   const int lda, const double *X, const int incX,
                   const double beta, double *Y, const int incY);
void ampblas_dsbmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const int K, const double alpha, const double *A,
                   const int lda, const double *X, const int incX,
                   const double beta, double *Y, const int incY);
void ampblas_dspmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const double alpha, const double *Ap,
                   const double *X, const int incX,
                   const double beta, double *Y, const int incY);
void ampblas_dger(const enum AMPBLAS_ORDER order, const int M, const int N,
                  const double alpha, const double *X, const int incX,
                  const double *Y, const int incY, double *A, const int lda);
void ampblas_dsyr(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                  const int N, const double alpha, const double *X,
                  const int incX, double *A, const int lda);
void ampblas_dspr(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                  const int N, const double alpha, const double *X,
                  const int incX, double *Ap);
void ampblas_dsyr2(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const double alpha, const double *X,
                   const int incX, const double *Y, const int incY, double *A,
                   const int lda);
void ampblas_dspr2(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const double alpha, const double *X,
                   const int incX, const double *Y, const int incY, double *A);


// 
// Routines with C and Z prefixes only
//
void ampblas_chemv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const void *alpha, const void *A,
                   const int lda, const void *X, const int incX,
                   const void *beta, void *Y, const int incY);
void ampblas_chbmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const int K, const void *alpha, const void *A,
                   const int lda, const void *X, const int incX,
                   const void *beta, void *Y, const int incY);
void ampblas_chpmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const void *alpha, const void *Ap,
                   const void *X, const int incX,
                   const void *beta, void *Y, const int incY);
void ampblas_cgeru(const enum AMPBLAS_ORDER order, const int M, const int N,
                   const void *alpha, const void *X, const int incX,
                   const void *Y, const int incY, void *A, const int lda);
void ampblas_cgerc(const enum AMPBLAS_ORDER order, const int M, const int N,
                   const void *alpha, const void *X, const int incX,
                   const void *Y, const int incY, void *A, const int lda);
void ampblas_cher(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                  const int N, const float alpha, const void *X, const int incX,
                  void *A, const int lda);
void ampblas_chpr(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                  const int N, const float alpha, const void *X,
                  const int incX, void *A);
void ampblas_cher2(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const int N,
                   const void *alpha, const void *X, const int incX,
                   const void *Y, const int incY, void *A, const int lda);
void ampblas_chpr2(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const int N,
                   const void *alpha, const void *X, const int incX,
                   const void *Y, const int incY, void *Ap);

void ampblas_zhemv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const void *alpha, const void *A,
                   const int lda, const void *X, const int incX,
                   const void *beta, void *Y, const int incY);
void ampblas_zhbmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const int K, const void *alpha, const void *A,
                   const int lda, const void *X, const int incX,
                   const void *beta, void *Y, const int incY);
void ampblas_zhpmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                   const int N, const void *alpha, const void *Ap,
                   const void *X, const int incX,
                   const void *beta, void *Y, const int incY);
void ampblas_zgeru(const enum AMPBLAS_ORDER order, const int M, const int N,
                   const void *alpha, const void *X, const int incX,
                   const void *Y, const int incY, void *A, const int lda);
void ampblas_zgerc(const enum AMPBLAS_ORDER order, const int M, const int N,
                   const void *alpha, const void *X, const int incX,
                   const void *Y, const int incY, void *A, const int lda);
void ampblas_zher(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                  const int N, const double alpha, const void *X, const int incX,
                  void *A, const int lda);
void ampblas_zhpr(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo,
                  const int N, const double alpha, const void *X,
                  const int incX, void *A);
void ampblas_zher2(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const int N,
                   const void *alpha, const void *X, const int incX,
                   const void *Y, const int incY, void *A, const int lda);
void ampblas_zhpr2(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const int N,
                   const void *alpha, const void *X, const int incX,
                   const void *Y, const int incY, void *Ap);


//----------------------------------------------------------------------------
// Prototypes for level 3 BLAS
//----------------------------------------------------------------------------

// 
// Routines with standard 4 prefixes (S, D, C, Z)
//
void ampblas_sgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const float alpha, const float *A,
                   const int lda, const float *B, const int ldb,
                   const float beta, float *C, const int ldc);
void ampblas_ssymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const int M, const int N,
                   const float alpha, const float *A, const int lda,
                   const float *B, const int ldb, const float beta,
                   float *C, const int ldc);
void ampblas_ssyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K,
                   const float alpha, const float *A, const int lda,
                   const float beta, float *C, const int ldc);
void ampblas_ssyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo,
                    const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K,
                    const float alpha, const float *A, const int lda,
                    const float *B, const int ldb, const float beta,
                    float *C, const int ldc);
void ampblas_strmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_DIAG Diag, const int M, const int N,
                   const float alpha, const float *A, const int lda,
                   float *B, const int ldb);
void ampblas_strsm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_DIAG Diag, const int M, const int N,
                   const float alpha, const float *A, const int lda,
                   float *B, const int ldb);

void ampblas_dgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const double alpha, const double *A,
                   const int lda, const double *B, const int ldb,
                   const double beta, double *C, const int ldc);
void ampblas_dsymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const int M, const int N,
                   const double alpha, const double *A, const int lda,
                   const double *B, const int ldb, const double beta,
                   double *C, const int ldc);
void ampblas_dsyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K,
                   const double alpha, const double *A, const int lda,
                   const double beta, double *C, const int ldc);
void ampblas_dsyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo,
                    const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K,
                    const double alpha, const double *A, const int lda,
                    const double *B, const int ldb, const double beta,
                    double *C, const int ldc);
void ampblas_dtrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_DIAG Diag, const int M, const int N,
                   const double alpha, const double *A, const int lda,
                   double *B, const int ldb);
void ampblas_dtrsm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_DIAG Diag, const int M, const int N,
                   const double alpha, const double *A, const int lda,
                   double *B, const int ldb);

void ampblas_cgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const void *alpha, const void *A,
                   const int lda, const void *B, const int ldb,
                   const void *beta, void *C, const int ldc);
void ampblas_csymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const int M, const int N,
                   const void *alpha, const void *A, const int lda,
                   const void *B, const int ldb, const void *beta,
                   void *C, const int ldc);
void ampblas_csyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K,
                   const void *alpha, const void *A, const int lda,
                   const void *beta, void *C, const int ldc);
void ampblas_csyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo,
                    const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K,
                    const void *alpha, const void *A, const int lda,
                    const void *B, const int ldb, const void *beta,
                    void *C, const int ldc);
void ampblas_ctrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_DIAG Diag, const int M, const int N,
                   const void *alpha, const void *A, const int lda,
                   void *B, const int ldb);
void ampblas_ctrsm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_DIAG Diag, const int M, const int N,
                   const void *alpha, const void *A, const int lda,
                   void *B, const int ldb);

void ampblas_zgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const void *alpha, const void *A,
                   const int lda, const void *B, const int ldb,
                   const void *beta, void *C, const int ldc);
void ampblas_zsymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const int M, const int N,
                   const void *alpha, const void *A, const int lda,
                   const void *B, const int ldb, const void *beta,
                   void *C, const int ldc);
void ampblas_zsyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K,
                   const void *alpha, const void *A, const int lda,
                   const void *beta, void *C, const int ldc);
void ampblas_zsyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo,
                    const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K,
                    const void *alpha, const void *A, const int lda,
                    const void *B, const int ldb, const void *beta,
                    void *C, const int ldc);
void ampblas_ztrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_DIAG Diag, const int M, const int N,
                   const void *alpha, const void *A, const int lda,
                   void *B, const int ldb);
void ampblas_ztrsm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_DIAG Diag, const int M, const int N,
                   const void *alpha, const void *A, const int lda,
                   void *B, const int ldb);


// 
// Routines with prefixes C and Z only
//
void ampblas_chemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const int M, const int N,
                   const void *alpha, const void *A, const int lda,
                   const void *B, const int ldb, const void *beta,
                   void *C, const int ldc);
void ampblas_cherk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K,
                   const float alpha, const void *A, const int lda,
                   const float beta, void *C, const int ldc);
void ampblas_cher2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo,
                    const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K,
                    const void *alpha, const void *A, const int lda,
                    const void *B, const int ldb, const float beta,
                    void *C, const int ldc);

void ampblas_zhemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side,
                   const enum AMPBLAS_UPLO Uplo, const int M, const int N,
                   const void *alpha, const void *A, const int lda,
                   const void *B, const int ldb, const void *beta,
                   void *C, const int ldc);
void ampblas_zherk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo,
                   const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K,
                   const double alpha, const void *A, const int lda,
                   const double beta, void *C, const int ldc);
void ampblas_zher2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo,
                    const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K,
                    const void *alpha, const void *A, const int lda,
                    const void *B, const int ldb, const double beta,
                    void *C, const int ldc);
void ampblas_xerbla(const char *srname, int * info);

#ifdef __cplusplus
}
#endif

#endif //AMPCBLAS_H
