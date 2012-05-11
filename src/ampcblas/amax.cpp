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
 * amax.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/amax.h"

extern "C" {

int ampblas_isamax(const int N, const float *X, const int incX)
{
    int ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampblas::amax<int>(N, X, incX) );
    return ret;
}

int ampblas_idamax(const int N, const double *X, const int incX)
{
    int ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampblas::amax<int>(N, X, incX) );
    return ret;
}

int ampblas_icamax(const int N, const ampblas_fcomplex *X, const int incX)
{
    int ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampblas::amax<int>(N, ampblas_cast(X), incX) );
    return ret;
}

int ampblas_izamax(const int N, const ampblas_dcomplex *X, const int incX)
{
    int ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampblas::amax<int>(N, ampblas_cast(X), incX) );
    return ret;
}

} // extern "C"
