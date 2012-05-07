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
 * axpy.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/axpy.h"

extern "C" {

void ampblas_saxpy(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY)
{
    AMPBLAS_CHECKED_CALL( ampblas::axpy(N, alpha, X, incX, Y, incY) );
}

void ampblas_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY)
{
	AMPBLAS_CHECKED_CALL( ampblas::axpy(N, alpha, X, incX, Y, incY) );
}

void ampblas_caxpy(const int N, const ampblas_fcomplex *alpha, const ampblas_fcomplex *X, const int incX, ampblas_fcomplex *Y, const int incY)
{
	fcomplex falpha =*(fcomplex*)(alpha);
	AMPBLAS_CHECKED_CALL( ampblas::axpy(N, falpha, (fcomplex*)X, incX, (fcomplex*)Y, incY) );
}

void ampblas_zaxpy(const int N, const ampblas_dcomplex *alpha, const ampblas_dcomplex *X,const int incX, ampblas_dcomplex *Y, const int incY)
{
    dcomplex dalpha =*(dcomplex*)(alpha);
	AMPBLAS_CHECKED_CALL( ampblas::axpy(N, dalpha, (dcomplex*)X, incX, (dcomplex*)Y, incY) );
}

} // extern "C"