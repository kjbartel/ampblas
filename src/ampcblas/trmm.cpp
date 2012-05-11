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
 * trmm.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/trmm.h"

extern "C" {

void ampblas_strmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const float alpha, const float *A, const int lda, float *B, const int ldb)
{
    AMPBLAS_CHECKED_CALL( ampblas::trmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb) );
}
void ampblas_dtrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const double alpha, const double *A, const int lda, double *B, const int ldb)
{
    AMPBLAS_CHECKED_CALL( ampblas::trmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb) );
}

void ampblas_ctrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const ampblas_fcomplex *alpha, const ampblas_fcomplex *A, const int lda, ampblas_fcomplex *B, const int ldb)
{
    const fcomplex calpha = *ampblas_cast(alpha);
    AMPBLAS_CHECKED_CALL( ampblas::trmm(Order, Side, Uplo, TransA, Diag, M, N, calpha, ampblas_cast(A), lda, ampblas_cast(B), ldb) );
}
void ampblas_ztrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const ampblas_dcomplex *alpha, const ampblas_dcomplex *A, const int lda, ampblas_dcomplex *B, const int ldb)
{
    const dcomplex zalpha = *ampblas_cast(alpha);
    AMPBLAS_CHECKED_CALL( ampblas::trmm(Order, Side, Uplo, TransA, Diag, M, N, zalpha, ampblas_cast(A), lda, ampblas_cast(B), ldb) );
}

} // extern "C"
