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
 * amax.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {
namespace _detail {

//
// amax_helper
//   Functor for AMAX reduction 
//

template<typename ret_type, typename value_type, typename x_type, typename functor>
struct amax_helper
{
    amax_helper(const value_type& value, const functor& max_op) restrict(cpu, amp) 
        : init_value(value), op(max_op) 
    {
    }

    // gets the maximum of the absolute values of lhs and X[idx], and stores in lhs
    void local_reduce(value_type& lhs, int idx, const x_type& X) const restrict(cpu, amp)
    {
        value_type temp(idx+1, _detail::abs(X[ concurrency::index<1>(idx) ]));
        lhs = _detail::max(lhs, temp);
    }

    // finds the maximum in a container and returns its position
    ret_type global_reduce(const std::vector<value_type>& vec) const
    {
         return std::max_element(vec.begin(), vec.end())->idx;
    }

    value_type init_value;
    functor op;
};

} // namespace _detail

//-------------------------------------------------------------------------
// AMAX
//   Finds the index of element having maximum absolute value in a container
//-------------------------------------------------------------------------

template <typename int_type, typename x_type>
int_type amax(int n, const x_type& X)
{
    typedef typename x_type::value_type T;
    typedef typename indexed_type<int_type,T> U;

    // static and const for view in parallel section 
    static const unsigned int tile_size = 64;
    static const unsigned int max_tiles = 64;

    U x0 = U(1, _detail::abs(X[concurrency::index<1>(0)]));
    auto func = _detail::amax_helper<int_type, U, x_type, _detail::maximum<U>>(x0, _detail::maximum<U>());

    // call generic 1D reduction
    return _detail::reduce<tile_size, max_tiles, int_type, U, x_type>(n, X, func);
}

template <typename index_type, typename value_type>
index_type amax(const int n, const value_type* x, const int incx)
{
	// Fortran indexing
	if (n < 1 || incx <= 0)
		return 1;

	if (x == nullptr)
		argument_error("amax", 2);

    auto x_vec = make_vector_view(n, x, incx);
    return amax<index_type>(n, x_vec);
} 

} // namespace ampblas
