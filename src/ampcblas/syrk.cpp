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
 * syrk.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/syrk.h"

extern "C" {

void ampblas_ssyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const float alpha, const float *A, const int lda, const float beta, float *C, const int ldc)
{
    AMPBLAS_CHECKED_CALL( ampblas::syrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc) );
}

void ampblas_dsyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const double alpha, const double *A, const int lda, const double beta, double *C, const int ldc)
{
    AMPBLAS_CHECKED_CALL( ampblas::syrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc) );
}

void ampblas_csyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const ampblas_fcomplex* alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex* beta, ampblas_fcomplex *C, const int ldc)
{
    const fcomplex calpha = *ampblas_cast(alpha);
    const fcomplex cbeta  = *ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampblas::syrk(Order, Uplo, Trans, N, K, calpha, ampblas_cast(A), lda, cbeta, ampblas_cast(C), ldc) );
}

void ampblas_zsyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const ampblas_dcomplex* alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex* beta, ampblas_dcomplex *C, const int ldc)
{
    const dcomplex zalpha = *ampblas_cast(alpha);
    const dcomplex zbeta  = *ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampblas::syrk(Order, Uplo, Trans, N, K, zalpha, ampblas_cast(A), lda, zbeta, ampblas_cast(C), ldc) );
}

// void ampblas_cherk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const float* alpha, const ampblas_fcomplex *A, const int lda, const float* beta, ampblas_fcomplex *C, const int ldc)
// {
// }

// void ampblas_zherk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const double* alpha, const ampblas_dcomplex *A, const int lda, const double* beta, ampblas_dcomplex *C, const int ldc)
// {
// }

} // extern "C"