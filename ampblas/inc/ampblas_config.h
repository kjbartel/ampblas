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
 * ampblas_config.h
 *
 * Common header for all ampblas implementation files. Contains shared headers
 * and utilites used for the implementation of all kernels.
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPBLAS_CONFIG_H
#define AMPBLAS_CONFIG_H

#include <numeric>
#include <algorithm>
#include <amp.h>
#include "ampblas_defs.h"
#include "ampblas_complex.h"

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace ampblas {

// The functions in the _detail namespace are used internally for other BLAS 
// functions internally. 
namespace _detail
{

// extent helpers
inline concurrency::extent<1> make_extent(int n) restrict(cpu, amp)
{
	return concurrency::extent<1>(n);
}

inline concurrency::extent<2> make_extent(int m, int n) restrict(cpu, amp)
{
	return concurrency::extent<2>(n,m);
}

//
// indexed_type
//   A wrapper to represent the value and its position of an element in a container. 
//   One usage is to find the index of the maximum value in a container. 
//
template <typename index_type, typename value_type>
struct indexed_type
{
    indexed_type() restrict(cpu, amp)
        : idx(1), val(value_type())  {}

    indexed_type(index_type idx, const value_type& val) restrict(cpu, amp)
        : idx(idx), val(val)  {}

    bool operator>(const indexed_type<index_type,value_type>& rhs) const restrict(cpu, amp) 
    {
        return val > rhs.val; 
    }

    bool operator<(const indexed_type<index_type,value_type>& rhs) const restrict(cpu, amp) 
    {
        return val < rhs.val; 
    }

    bool operator==(const indexed_type<index_type,value_type>& rhs) const restrict(cpu, amp) 
    {
        return val == rhs.val; 
    }

    index_type idx;
    value_type val;
};

// Generic fill algorithm on any multi-dimensional container
template <int rank, typename value_type, typename x_type>
void fill(const concurrency::accelerator_view& av, const concurrency::extent<rank>& e, value_type&& value, x_type&& x)
{
    concurrency::parallel_for_each(av, e, [=] (concurrency::index<rank> idx) restrict(amp) 
    {
        x[idx] = value;
    });
}

// Triangular fill for rank-2 containers
template <typename value_type>
void fill(const concurrency::accelerator_view& av, enum class uplo uplo, const value_type& value, const concurrency::array_view<value_type,2>& x)
{
    if (uplo == uplo::upper)
    {
        concurrency::parallel_for_each(av, x.extent, [=] (concurrency::index<2> idx) restrict(amp) 
        {
            auto i = idx[1];
            auto j = idx[0];
            if ( i <= j )
                x[idx] = value;
        });
    }
    else
    {
        concurrency::parallel_for_each(av, x.extent, [=] (concurrency::index<2> idx) restrict(amp) 
        {
            auto i = idx[1];
            auto j = idx[0];
            if ( i >= j )
                x[idx] = value;
        });
    }
}

// Generic scale algorithm on any multi-dimensional container
template <int rank, typename value_type, typename x_type>
void scale(const concurrency::accelerator_view& av, const concurrency::extent<rank>& e, const value_type& value, x_type&& x)
{
    concurrency::parallel_for_each(
        av,
        e, 
        [=] (concurrency::index<rank> idx) restrict(amp) 
        {
            x[idx] *= value;
        }
    );
}

// Generic swap algorithm on any multi-dimensional container
template <int rank, typename x_type, typename y_type>
void swap(const concurrency::accelerator_view& av, const concurrency::extent<rank>& e, x_type&& x, y_type&& y)
{
    concurrency::parallel_for_each(
        av, 
        e, 
        [=] (concurrency::index<rank> idx) restrict(amp) 
        {
            auto tmp = y[idx];
            y[idx] = x[idx];
            x[idx] = tmp;
        }
    );
}

// Triangular fill for rank-2 containers
template <typename scalar_type, typename value_type>
void scale(const concurrency::accelerator_view& av, enum class uplo uplo, const scalar_type& value, const concurrency::array_view<value_type,2>& x)
{
    if ( uplo == uplo::upper )
        concurrency::parallel_for_each(av, x.extent, [=] (concurrency::index<2> idx) restrict(amp) 
        {
            auto i = idx[1];
            auto j = idx[0];
            if ( i <= j )
                x[idx] = value*x[idx];
        });
    else
        concurrency::parallel_for_each(av, x.extent, [=] (concurrency::index<2> idx) restrict(amp) 
        {
            auto i = idx[1];
            auto j = idx[0];
            if ( i >= j )
                x[idx] = value*x[idx];
        });
}

// eneric copy algorithm on any multi-dimensional container
template <int rank, typename x_type, typename y_type>
void copy(const concurrency::accelerator_view& av, const concurrency::extent<rank>& e, x_type&& x, y_type&& y)
{
    concurrency::parallel_for_each(av, e, 
        [=] (concurrency::index<rank> idx) restrict(amp) 
        {
            y[idx] = x[idx];
        }
    );
}

template <typename T>
inline T abs(const T& val) restrict(cpu, amp)
{
    return val >= 0 ? val: -val;
}

// some routines use this form of abs() for complex values
template <typename value_type>
inline value_type abs_1(const value_type& val) restrict(cpu, amp)
{
    return abs(val);
}

template <typename value_type>
inline value_type abs_1(const complex<value_type>& val) restrict(cpu, amp)
{
    return abs(val.real()) + abs(val.imag());
}

template<typename T>
inline const T& max(const T& a, const T& b) restrict(cpu, amp) 
{
    return a > b ? a : b;
}

template<typename T>
inline const T& min(const T& a, const T& b) restrict(cpu, amp) 
{
    return a < b ? a : b;
}

// returns a value whose absolute value matches that of a, but whose sign bit matches that of b.
template <typename T>
inline T copysign(const T& a, const T& b) restrict(cpu, amp) 
{
    T x = _detail::abs(a);
    return (b >= 0 ? x : -x);
}

template <typename T>
struct maximum
{
    const T& operator()(const T& lhs, const T& rhs) const restrict (cpu, amp) 
    { 
        return _detail::max(lhs, rhs);
    }
};

template <typename T>
struct sum 
{
    T operator()(const T& lhs, const T& rhs) const restrict (cpu, amp) 
    { 
        return lhs + rhs; 
    }
};

template <typename value_type>
struct subtract
{
    value_type subtrahend;

    subtract(const value_type& subtrahend) restrict (cpu,amp)
        : subtrahend(subtrahend) 
    {
    }

    inline void operator()(value_type& minuend) const restrict (cpu,amp)
    {
        minuend -= subtrahend;
    }
};


struct noop
{
    template <typename value_type>
    static inline const value_type op(const value_type& val) restrict (cpu, amp)
    {
        return val;
    }
};

struct conjugate
{
    template <typename value_type>
    static inline const value_type op(const value_type& val) restrict (cpu, amp)
    {
        // noop for real types
        return val;
    }

    template <typename value_type>
    static inline const complex<value_type> op(const complex<value_type>& val) restrict (cpu, amp)
    {
        complex<value_type> ret(val.real(), -val.imag());
        return ret;
    }
};

template <typename value_type>
inline void only_real(const value_type&) restrict(cpu, amp)
{
    // noop for real
}

template <typename value_type>
inline void only_real(complex<value_type>& val) restrict(cpu, amp)
{
    val.imag(value_type());
}

//
// bounds checked opeators
// 
template <bool enabled, typename value_type>
inline value_type guarded_read(const concurrency::array_view<value_type,2>& A, const concurrency::index<2>& idx) restrict(cpu,amp)
{
    return (enabled && A.extent.contains(idx)) ? A[idx] : value_type();
}

template <bool enabled, typename value_type>
inline void guarded_write(const concurrency::array_view<value_type,2>& A, const concurrency::index<2>& idx, const value_type& val) restrict(cpu,amp)
{
    if (enabled && A.extent.contains(idx))
        A[idx] = val;
}

template <bool enabled, typename value_type, typename operation>
inline void guarded_update(const concurrency::array_view<value_type,2>& A, const concurrency::index<2>& idx, const operation& op) restrict(cpu,amp)
{
    if (enabled && A.extent.contains(idx))
        op(A[idx]);
}

inline bool is_diag(const concurrency::index<2>& idx) restrict(cpu,amp)
{
    return idx[0] == idx[1];
}

template <typename x_type, typename y_type>
void require_matched_extent(const x_type& /*x*/, const y_type& /*y*/)
{
    // TODO: what should this actually do if not matched?
}

template <typename T, unsigned int tile_size, typename functor>
void tile_local_reduction(T* const mem, concurrency::tiled_index<tile_size> tid, const functor& op) restrict(amp)
{
    // local index
    unsigned int local = tid.local[0];

    // unrolled for performance
    if (tile_size >= 1024) { if (local < 512) { mem[0] = op(mem[0], mem[512]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=  512) { if (local < 256) { mem[0] = op(mem[0], mem[256]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=  256) { if (local < 128) { mem[0] = op(mem[0], mem[128]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=  128) { if (local <  64) { mem[0] = op(mem[0], mem[ 64]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=   64) { if (local <  32) { mem[0] = op(mem[0], mem[ 32]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=   32) { if (local <  16) { mem[0] = op(mem[0], mem[ 16]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=   16) { if (local <   8) { mem[0] = op(mem[0], mem[  8]); } tid.barrier.wait_with_tile_static_memory_fence(); }   
    if (tile_size >=    8) { if (local <   4) { mem[0] = op(mem[0], mem[  4]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=    4) { if (local <   2) { mem[0] = op(mem[0], mem[  2]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=    2) { if (local <   1) { mem[0] = op(mem[0], mem[  1]); } tid.barrier.wait_with_tile_static_memory_fence(); }
}

// Generic reduction of an 1D container with the reduction operation specified by a helper functor
template <unsigned int tile_size, unsigned int max_tiles, typename ret_type, typename elm_type, typename x_type, typename functor>
ret_type reduce(const concurrency::accelerator_view& av, int n, const x_type& X, const functor& reduce_helper)
{
    // runtime sizes
    unsigned int tile_count = (n+tile_size-1) / tile_size;
    tile_count = std::min(tile_count, max_tiles);   

    // simultaneous live threads
    const unsigned int thread_count = tile_count * tile_size;

    // global buffer (return type)
    concurrency::array<elm_type,1> global_buffer(tile_count);
    concurrency::array_view<elm_type,1> global_buffer_view(global_buffer);

    // configuration
    concurrency::extent<1> extent(thread_count);

    concurrency::parallel_for_each(
        av,
        extent.tile<tile_size>(),
        [=] (concurrency::tiled_index<tile_size> tid) restrict(amp)
    {
        // shared tile buffer
        tile_static elm_type local_buffer[tile_size];

        // indexes
        int idx = tid.global[0];

        // this threads's shared memory pointer
        elm_type& smem = local_buffer[ tid.local[0] ];

        // initialize local buffer
        smem = reduce_helper.init_value;

        // fold data into local buffer
        while (idx < n)
        {
            // reduction of smem and X[idx] with results stored in smem
            reduce_helper.local_reduce(smem, idx, X);

            // next chunk
            idx += thread_count;
        }

        // synchronize
        tid.barrier.wait_with_tile_static_memory_fence();

        // reduce all values in this tile
        _detail::tile_local_reduction<elm_type,tile_size>(&smem, tid, reduce_helper.op);

        // only 1 thread per tile does the inter tile communication
        if (tid.local[0] == 0)
        {
            // write to global buffer in this tiles
            global_buffer_view[ tid.tile[0] ] = smem;
        }
    });

    // 2nd pass reduction
    std::vector<elm_type> host_buffer(global_buffer);
    return reduce_helper.global_reduce(host_buffer);
}

} // namespace _detail
} // namespace ampblas

#endif // AMPBLAS_CONFIG_H
