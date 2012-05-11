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
 * symv.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/symv.h"

extern "C" {

void ampblas_ssymv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const int N, const float alpha, const float *A, const int lda, const float *X, const int incX, const float beta, float *Y, const int incY)
{
    AMPBLAS_CHECKED_CALL( ampblas::symv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY) );
}

void ampblas_dsymv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const int N, const double alpha, const double *A, const int lda, const double *X, const int incX, const double beta, double *Y, const int incY)
{
    AMPBLAS_CHECKED_CALL( ampblas::symv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY) );
}

// void ampblas_shemv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const int N, const ampblas_fcomplex* alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex *X, const int incX, const ampblas_fcomplex* beta, ampblas_fcomplex *Y, const int incY)
// {
// }

// void ampblas_dhemv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const int N, const ampblas_dcomplex* alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex *X, const int incX, const ampblas_dcomplex* beta, ampblas_dcomplex *Y, const int incY)
// {
// }

} // extern "C"
