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
 * trmv.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/trmv.h"

extern "C" {

void ampblas_strmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const float *A, const int lda, float *X, const int incX)
{
    AMPBLAS_CHECKED_CALL(ampblas::trmv(order, Uplo, TransA, Diag, N, A, lda, X, incX));
}

void ampblas_dtrmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const double *A, const int lda, double *X, const int incX)
{
    AMPBLAS_CHECKED_CALL(ampblas::trmv(order, Uplo, TransA, Diag, N, A, lda, X, incX));
}

} // extern "C"
