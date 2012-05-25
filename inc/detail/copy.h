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
 * copy.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {

//-------------------------------------------------------------------------
// COPY
//   copy a container to another container. The two containers cannot be 
//   overlpped.
//-------------------------------------------------------------------------

template <typename x_type, typename y_type>
void copy(const concurrency::accelerator_view& av, const x_type& x, y_type& y)
{
    _detail::copy(av, x.extent, x, y);
}

// Generic COPY algorithm for AMPBLAS arrays of type T
template <typename value_type>
void copy(int n, const value_type *x, int incx, value_type *y, int incy)
{
	// quick return
	if (n <= 0)
		return;

    // check arguments
    if (x == nullptr)
		argument_error("copy", 2);
    if (y == nullptr)
        argument_error("copy", 4);

    auto x_vec = make_vector_view(n, x, incx);
    auto y_vec = make_vector_view(n, y, incy);

	copy(get_current_accelerator_view(), x_vec, y_vec);
}

} // namespace ampblas
