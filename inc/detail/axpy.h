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
 * axpy.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {

//-------------------------------------------------------------------------
// AXPY
//  a scalar alpha times a container X plus a container Y.
//-------------------------------------------------------------------------

template <typename alpha_type, typename x_type, typename y_type>
void axpy(const concurrency::accelerator_view& av, const alpha_type& alpha, const x_type& x, y_type& y)
{
    concurrency::parallel_for_each(
        av, 
        x.extent, 
        [=] (concurrency::index<1> idx) restrict(amp) 
        {
            y[idx] += alpha * x[idx];
        }
    );
}

// Generic AXPY algorithm for AMPBLAS arrays of type T
template <typename value_type>
void axpy(int n, value_type alpha, const value_type *x, int incx, value_type *y, int incy)
{
	// quick return
	if (n <= 0 || alpha == value_type())
        return;

    // check arguments
    if (x == nullptr)
		argument_error("axpy", 3);
    if (y == nullptr)
		argument_error("axpy", 5);

    auto x_vec = make_vector_view(n, x, incx);
    auto y_vec = make_vector_view(n, y, incy);

    axpy(get_current_accelerator_view(), alpha, x_vec, y_vec); 
}

} // namespace ampblas
