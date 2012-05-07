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
 * symm.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/symm.h"

extern "C" {

void ampblas_ssymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc)
{
    AMPBLAS_CHECKED_CALL( ampblas::symm<float>(Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc) );
}

void ampblas_dsymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc)
{
    AMPBLAS_CHECKED_CALL( ampblas::symm<double>(Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc) );
}

} // extern "C"
