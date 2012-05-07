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
 * dot.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {
namespace _detail {

template<typename ret_type, typename value_type, typename x_type, typename functor>
struct dot_helper
{
    dot_helper(const value_type& value, const functor& sum_op) restrict(cpu, amp) 
        : init_value(value), op(sum_op) 
    {}

    // x_type must be a pair of the two vector types
    void local_reduce(ret_type& lhs, int idx, const x_type& X) const restrict(cpu, amp)
    {
        lhs += value_type(X.first[concurrency::index<1>(idx)]) * value_type(X.second[concurrency::index<1>(idx)]);
    }

    // returns the summation of all values in a container
    ret_type global_reduce(const std::vector<value_type>& vec) const
    {
         return std::accumulate(vec.begin(), vec.end(), init_value);
    }

    ret_type init_value;
    functor op;
};

} // namespace _detail

//-------------------------------------------------------------------------
// DOT
//   computes the dot product of two 1D arrays.
//-------------------------------------------------------------------------

template <typename ret_type, typename operation, typename array_type>
ret_type dot(int n, const array_type& X, const array_type& Y)
{
    typedef typename array_type::value_type T;

    // tuning sizes
    static const unsigned int tile_size = 128;
    static const unsigned int max_tiles = 64;

    auto func = _detail::dot_helper<ret_type, ret_type, std::pair<array_type,array_type>, _detail::sum<ret_type>>(ret_type(), _detail::sum<ret_type>());

    // call generic 1D reduction
    return _detail::reduce<tile_size, max_tiles, ret_type, ret_type>(n, std::make_pair(X,Y), func);
}

// Generic NRM2 algorithm for AMPBLAS arrays of type T
template <typename value_type, typename accumulation_type, typename operation>
accumulation_type dot(int n, const value_type *x, int incx, const value_type *y, int incy)
{
	// quick return
    if (x <= 0) 
        return accumulation_type();
 
    if (x == nullptr)
		argument_error("dot", 2);

    if (y == nullptr)
        argument_error("dot", 4);

    auto x_vec = make_vector_view(n, x, incx);
    auto y_vec = make_vector_view(n, y, incy);

    return dot<accumulation_type,operation>(n, x_vec, y_vec);
}

} // namespace ampblas