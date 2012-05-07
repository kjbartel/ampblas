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
 * scal.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/scal.h"

extern "C" {

void ampblas_sscal(const int N, const float alpha, float *X, const int incX)
{
    AMPBLAS_CHECKED_CALL( ampblas::scal(N, alpha, X, incX) );
}

void ampblas_dscal(const int N, const double alpha, double *X, const int incX)
{
	AMPBLAS_CHECKED_CALL( ampblas::scal(N, alpha, X, incX) );
}

void ampblas_cscal(const int N, const ampblas_fcomplex *alpha, ampblas_fcomplex *X, const int incX)
{
	const fcomplex falpha =*(fcomplex*)(alpha);
	AMPBLAS_CHECKED_CALL( ampblas::scal(N, falpha, (fcomplex*)X, incX) );
}

void ampblas_zscal(const int N, const ampblas_dcomplex *alpha, ampblas_dcomplex *X, const int incX)
{
    dcomplex dalpha =*(dcomplex*)(alpha);
	AMPBLAS_CHECKED_CALL( ampblas::scal(N, dalpha, (dcomplex*)X, incX) );
}

void ampblas_csscal(const int N, const float alpha, ampblas_fcomplex *X, const int incX)
{
    AMPBLAS_CHECKED_CALL( ampblas::scal<fcomplex>(N, alpha, (fcomplex*)X, incX) );
}

void ampblas_zdscal(const int N, const double alpha, ampblas_dcomplex *X, const int incX)
{
	AMPBLAS_CHECKED_CALL( ampblas::scal<dcomplex>(N, alpha, (dcomplex*)X, incX) );
}

} // extern "C"