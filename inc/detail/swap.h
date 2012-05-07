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
 * swap.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {

//-------------------------------------------------------------------------
// SWAP
//   The input buffers or containers cannot overlap with each other. Otherwise,
// runtime will throw an ampblas_exception when the buffers are bound. 
//-------------------------------------------------------------------------

template <typename value_type>
void swap(int n, value_type *x, int incx, value_type *y, int incy)
{
	// quick return
	if (n <= 0 || x == y) 
		return;
 
    // check arguments
    if (x == nullptr)
		argument_error("swap", 2);
	if (y == nullptr)
		argument_error("swap", 3);

    auto x_vec = make_vector_view(n,x,incx);
    auto y_vec = make_vector_view(n,y,incy);
    _detail::swap(make_extent(n), x_vec, y_vec);
}

} // namespace ampblas
