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
 * ger.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/ger.h"

extern "C" {

void ampblas_sger(const enum AMPBLAS_ORDER order, const int M, const int N, const float alpha, const float *X, const int incX, const float *Y, const int incY, float *A, const int lda)
{
    AMPBLAS_CHECKED_CALL( ampblas::ger<float,ampblas::_detail::noop>(order, M, N, alpha, X, incX, Y, incY, A, lda) );
}

void ampblas_dger(const enum AMPBLAS_ORDER order, const int M, const int N, const double alpha, const double *X, const int incX, const double *Y, const int incY, double *A, const int lda)
{
	AMPBLAS_CHECKED_CALL( ampblas::ger<double,ampblas::_detail::noop>(order, M, N, alpha, X, incX, Y, incY, A, lda) );
}

void ampblas_cgeru(const enum AMPBLAS_ORDER order, const int M, const int N, const ampblas_fcomplex* alpha, const ampblas_fcomplex *X, const int incX, const ampblas_fcomplex *Y, const int incY, ampblas_fcomplex *A, const int lda)
{
    const fcomplex calpha = *ampblas_cast(alpha);
    AMPBLAS_CHECKED_CALL( ampblas::ger<fcomplex,ampblas::_detail::noop>(order, M, N, calpha, ampblas_cast(X), incX, ampblas_cast(Y), incY, ampblas_cast(A), lda) );
}

void ampblas_zgeru(const enum AMPBLAS_ORDER order, const int M, const int N, const ampblas_dcomplex* alpha, const ampblas_dcomplex *X, const int incX, const ampblas_dcomplex *Y, const int incY, ampblas_dcomplex *A, const int lda)
{
    const dcomplex zalpha = *ampblas_cast(alpha);
	AMPBLAS_CHECKED_CALL( ampblas::ger<dcomplex,ampblas::_detail::noop>(order, M, N, zalpha, ampblas_cast(X), incX, ampblas_cast(Y), incY, ampblas_cast(A), lda) );
}

// void ampblas_cgerc(const enum AMPBLAS_ORDER order, const int M, const int N, const ampblas_fcomplex* alpha, const ampblas_fcomplex *X, const int incX, const ampblas_fcomplex *Y, const int incY, ampblas_fcomplex *A, const int lda)
// {
//     const fcomplex calpha = *ampblas_cast(alpha);
//     AMPBLAS_CHECKED_CALL( ampblas::ger<fcomplex,ampblas::_detail::conj>(order, M, N, calpha, ampblas_cast(X), incX, ampblas_cast(Y), incY, ampblas_cast(A), lda) );
// }

// void ampblas_zgerc(const enum AMPBLAS_ORDER order, const int M, const int N, const ampblas_dcomplex* alpha, const ampblas_dcomplex *X, const int incX, const ampblas_dcomplex *Y, const int incY, ampblas_dcomplex *A, const int lda)
// {
//     const dcomplex zalpha = *ampblas_cast(alpha);
// 	   AMPBLAS_CHECKED_CALL( ampblas::ger<dcomplex,ampblas::_detail::conj>(order, M, N, zalpha, ampblas_cast(X), incX, ampblas_cast(Y), incY, ampblas_cast(A), lda) );
// }

} // extern "C"
