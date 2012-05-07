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
 * dot.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/dot.h"

extern "C" {

// float ampblas_sdsdot(const int N, const float alpha, const float *X, const int incX, const float *Y, const int incY) 
// {
// }

double ampblas_dsdot(const int N, const float *X, const int incX, const float *Y, const int incY)
{
    double ret = 0;
	AMPBLAS_CHECKED_CALL( ret = ampblas::dot<float,double,ampblas::_detail::noop>(N,X,incX,Y,incY) );
    return ret;
}

float ampblas_sdot(const int N, const float  *X, const int incX, const float  *Y, const int incY)
{
    float ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampblas::dot<float,float,ampblas::_detail::noop>(N,X,incX,Y,incY) );
    return ret;
}

double ampblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY)
{   
    double ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampblas::dot<double,double,ampblas::_detail::noop>(N,X,incX,Y,incY) );
    return ret;
}

} // extern "C"
