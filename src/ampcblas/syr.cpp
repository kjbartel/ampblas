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
 * syr.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/syr.h"

extern "C" {

void ampblas_ssyr(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO uplo, const int N, const float alpha, const float *X, const int incX, float *A, const int lda)
{
    AMPBLAS_CHECKED_CALL( ampblas::syr<ampblas::_detail::noop>(order, uplo, N, alpha, X, incX, A, lda) );
}

void ampblas_dsyr(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO uplo, const int N, const double alpha, const double *X, const int incX, double *A, const int lda)
{
	AMPBLAS_CHECKED_CALL( ampblas::syr<ampblas::_detail::noop>(order, uplo, N, alpha, X, incX, A, lda) );
}

void ampblas_cher(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO uplo, const int N, const float alpha, const ampblas_fcomplex *X, const int incX, ampblas_fcomplex *A, const int lda)
{
    AMPBLAS_CHECKED_CALL( ampblas::syr<ampblas::_detail::conjugate>(order, uplo, N, alpha, ampblas_cast(X), incX, ampblas_cast(A), lda) );
}
 
void ampblas_zher(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO uplo, const int N, const double alpha, const ampblas_dcomplex *X, const int incX, ampblas_dcomplex *A, const int lda)
{
    AMPBLAS_CHECKED_CALL( ampblas::syr<ampblas::_detail::conjugate>(order, uplo, N, alpha, ampblas_cast(X), incX, ampblas_cast(A), lda) );
}

} // extern "C"