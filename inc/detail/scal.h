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
 * scal.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {

//-------------------------------------------------------------------------
// SCAL
//-------------------------------------------------------------------------

// Generic SCAL algorithm for AMPBLAS arrays of type value_type
template <typename value_type>
void scal(int n, value_type alpha, value_type *x, int incx)
{
	// quick return
	if (n <= 0) 
        return;

    // check arguments
    if (x == nullptr)
		argument_error("scal", 3);

    auto x_vec = make_vector_view(n,x,incx);

    _detail::scale(make_extent(n), alpha, x_vec);
}

} // namespace ampblas
