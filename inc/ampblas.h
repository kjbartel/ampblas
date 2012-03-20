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
 * BLAS levels 1,2,3 library header for C++ AMP.
 *
 * This file contains C++ template BLAS APIs for generic data types.
 *
 *---------------------------------------------------------------------------*/
#ifndef AMPBLAS_H
#define AMPBLAS_H

#ifdef __cplusplus
#include <amp.h>
#include "ampblas_defs.h"
#include "ampblas_runtime.h"

namespace ampblas
{
// Generic AXPY algorithm on any multi-dimensional container and scalar type
template <int rank, typename alpha_type, typename x_type, typename y_type>
void axpy(const concurrency::extent<rank>& e, alpha_type&& alpha, x_type&& X, y_type&& Y)
{
	concurrency::parallel_for_each(get_current_accelerator_view(), e, [=] (concurrency::index<rank> idx) restrict(amp) {
		Y[idx] += alpha * X[idx];
	});
}

// Generic AXPY algorithm for AMPBLAS arrays of scalar type T
template <typename T>
void axpy(const int N, const T alpha, const T *X,
          const int incX, T *Y, const int incY)
{
	concurrency::array_view<T> avX = get_array_view(X, N);
	concurrency::array_view<T> avY = get_array_view(Y, N);

	auto avX1 = make_stride_view(avX, incX, make_extent(N));
	auto avY1 = make_stride_view(avY, incY, make_extent(N));
    
	ampblas::axpy(make_extent(N), alpha, avX1, avY1);
}

} // namespace ampblas
#endif // __cplusplus
#endif //AMPBLAS_H

