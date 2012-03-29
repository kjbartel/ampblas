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
 * Generic wrappers to BLAS levels 1,2,3 library header for AMP CBLAS.
 *
 *---------------------------------------------------------------------------*/
#ifndef AMPXBLAS_H
#define AMPXBLAS_H

#include "ampcblas.h"

//----------------------------------------------------------------------------
// Prototypes for level 1 BLAS routines
//----------------------------------------------------------------------------

// 
// Routines with standard 4 prefixes (s, d, c, z)
//
// TODO: add other routines
template<typename value_type> 
void ampblas_xaxpy(const int N, const value_type alpha, const value_type *X,
                   const int incX, value_type *Y, const int incY);

template<> 
inline void ampblas_xaxpy<float>(const int N, const float alpha, const float *X,
				                 const int incX, float *Y, const int incY)
{
	ampblas_saxpy(N, alpha, X, incX, Y, incY);
}

template<> 
inline void ampblas_xaxpy<double>(const int N, const double alpha, const double *X,
							      const int incX, double *Y, const int incY)
{
	ampblas_daxpy(N, alpha, X, incX, Y, incY);
}

#endif //AMPXBLAS_H
