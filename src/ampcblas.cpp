/* 
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
 * BLAS levels 1,2,3 library implementation files.
 *
 * This file contains AMP CBLAS wrappers to AMP C++ BLAS implementaion. 
 *
 *---------------------------------------------------------------------------*/
#include <assert.h>
#include "ampcblas.h"
#include "ampblas.h"

// 
// AMP CBLAS AXPY implementation file.
// 
extern "C" void ampblas_saxpy(const int N, const float alpha, const float *X,
                              const int incX, float *Y, const int incY)
{
    // TODO: catch exceoptions and convert to ampblas_result error code, and queriable 
    // by calling get_last_err_code, get_last_err_message
    ampblas::axpy<float>(N, alpha, X, incX, Y, incY);
}

void ampblas_daxpy(const int N, const double alpha, const double *X,
                   const int incX, double *Y, const int incY)
{
	ampblas::axpy<double>(N, alpha, X, incX, Y, incY);
}

#pragma warning ( push )
#pragma warning ( disable : 4100 ) // 'N' : unreferenced formal parameter

void ampblas_caxpy(const int N, const void *alpha, const void *X,
                   const int incX, void *Y, const int incY)
{
	assert(0); // not yet implemented
}


void ampblas_zaxpy(const int N, const void *alpha, const void *X,
                   const int incX, void *Y, const int incY)
{
	assert(0); // not yet implemented
}
#pragma warning ( pop )
