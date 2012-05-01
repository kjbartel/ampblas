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
#include <numeric>
#include <algorithm>
#include <amp.h>
#include "ampblas_defs.h"
#include "ampblas_complex.h"
#include "ampblas_runtime.h"

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace ampblas
{

//----------------------------------------------------------------------------
// AMPBLAS error checking
//----------------------------------------------------------------------------
inline void argument_error(std::string fname, int info)
{
	ampblas_xerbla(fname.c_str(), &info);
	throw ampblas_exception(AMPBLAS_INVALID_ARG); 
}

inline void not_yet_implemented()
{
	throw ampblas_exception(AMPBLAS_NOT_SUPPORTED_FEATURE);
}

// The functions in the _detail namespace are used internally for other BLAS 
// functions internally. 
namespace _detail
{

// Generic fill algorithm on any multi-dimensional container
template <int rank, typename value_type, typename x_type>
inline void fill(const concurrency::extent<rank>& e, value_type&& value, x_type&& x)
{
    concurrency::parallel_for_each(get_current_accelerator_view(), e, [=] (concurrency::index<rank> idx) restrict(amp) 
    {
        x[idx] = value;
    });
}

// Generic scale algorithm on any multi-dimensional container
template <int rank, typename value_type, typename x_type>
inline void scale(const concurrency::extent<rank>& e, value_type&& value, x_type&& x)
{
    concurrency::parallel_for_each(get_current_accelerator_view(), e, [=] (concurrency::index<rank> idx) restrict(amp) 
    {
        x[idx] *= value;
    });
}

// Generic swap algorithm on any multi-dimensional container
template <int rank, typename x_type, typename y_type>
inline void swap(const concurrency::extent<rank>& e, x_type&& x, y_type&& y)
{
    concurrency::parallel_for_each(get_current_accelerator_view(), e, [=] (concurrency::index<rank> idx) restrict(amp) 
    {
        auto tmp = y[idx];
        y[idx] = x[idx];
        x[idx] = tmp;
    });
}

// Generic copy algorithm on any multi-dimensional container
template <int rank, typename x_type, typename y_type>
inline void copy(const concurrency::extent<rank>& e, x_type&& x, y_type&& y)
{
    concurrency::parallel_for_each(get_current_accelerator_view(), e, [=] (concurrency::index<rank> idx) restrict(amp) 
    {
        y[idx] = x[idx];
    });
}

template <typename T>
inline T abs(const T& val) restrict(cpu, amp)
{
    return val >= 0 ? val: -val;
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

struct noop
{
    void operator()() const restrict (cpu, amp) {}
};

//
// asum_helper
//   Functor for ASUM reduction 
//
template<typename ret_type, typename value_type, typename x_type, typename functor>
struct asum_helper
{
    asum_helper(const value_type& value, const functor& sum_op) restrict(cpu, amp)
        : init_value(value), op(sum_op) 
    {
    }

    // computes the sum of lhs and the absolute value of X[idx] and stores results in lhs
    void local_reduce(value_type& lhs, int idx, const x_type& X) const restrict(cpu, amp)
    {
        lhs += _detail::abs(X[ concurrency::index<1>(idx) ]);
    }

    // reduction of container vec
    ret_type global_reduce(const std::vector<value_type>& vec) const
    {
         return std::accumulate(vec.begin(), vec.end(), init_value);
    }

    value_type init_value;
    functor op;
};

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

//
// nrm2_helper
//   Functor for NRM2 reduction 
//
template<typename ret_type, typename value_type, typename x_type, typename functor>
struct nrm2_helper
{
    nrm2_helper(const value_type& value, const functor& sum_op) restrict(cpu, amp) 
        : init_value(value), op(sum_op) 
    {
    }

    // computes the euclidean norm of lhs and the absolute value of X[idx] and stores results in lhs
    void local_reduce(value_type& lhs, int idx, const x_type& X) const restrict(cpu, amp)
    {
        value_type temp = X[ concurrency::index<1>(idx) ];
        lhs += (temp * temp);
    }

    // returns the square of the summation of all values in a container
    ret_type global_reduce(const std::vector<value_type>& vec) const
    {
         return std::sqrt(std::accumulate(vec.begin(), vec.end(), init_value));
    }

    value_type init_value;
    functor op;
};


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
template<unsigned int tile_size, 
         unsigned int max_tiles, 
         typename ret_type, 
         typename elm_type,
         typename x_type, 
         typename functor>
ret_type reduce(int n, const x_type& X, const functor& reduce_helper)
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

    concurrency::parallel_for_each (
        get_current_accelerator_view(), 
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

//=========================================================================
// BLAS 1
//=========================================================================

//-------------------------------------------------------------------------
// ASUM
//  computes the sum of the absolute values in a container.
//-------------------------------------------------------------------------

template <typename x_type>
typename x_type::value_type asum(int n, const x_type& X)
{
    typedef typename x_type::value_type T;

    // static and const for view in parallel section 
    static const unsigned int tile_size = 128;
    static const unsigned int max_tiles = 64;

    auto func = _detail::asum_helper<T, T, x_type, _detail::sum<T>>(T(), _detail::sum<T>());

    // call generic 1D reduction
    return _detail::reduce<tile_size, max_tiles, T, T>(n, X, func);
}

template <typename value_type>
value_type asum(const int n, const value_type* x, const int incx)
{
    // quick return
	if (n == 0 || incx <= 0)
		return value_type();

	// argument check
	if (x == nullptr)
		argument_error("asum", 2);

    auto x_vec = make_vector_view(n, x, incx);

    return asum(n, x_vec);
}

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

//-------------------------------------------------------------------------
// AXPY
//  a scalar alpha times a container X plus a container Y.
//-------------------------------------------------------------------------

// Generic AXPY algorithm on any multi-dimensional container
template <int rank, typename alpha_type, typename x_type, typename y_type>
void axpy(const concurrency::extent<rank>& e, alpha_type&& alpha, x_type&& X, y_type&& Y)
{
    concurrency::parallel_for_each(get_current_accelerator_view(), e, [=] (concurrency::index<rank> idx) restrict(amp) 
    {
        Y[idx] += alpha * X[idx];
    });
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

    axpy(make_extent(n), alpha, x_vec, y_vec); 
}

//-------------------------------------------------------------------------
// COPY
//   copy a container to another container. The two containers cannot be 
//   overlpped.
//-------------------------------------------------------------------------

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

	_detail::copy(x_vec.extent, x_vec, y_vec);
}

//-------------------------------------------------------------------------
// DOT
//   computes the dot product of two 1D arrays.
//-------------------------------------------------------------------------

template <typename ret_type, typename operation, typename array_type>
ret_type dot(int n, const array_type& X, const array_type& Y)
{
    // static and const for view in parallel section 
    static const unsigned int tile_size = 128;
    static const unsigned int max_tiles = 64;

    // runtime sizes
    unsigned int tile_count = (n+tile_size-1) / tile_size;
    tile_count = std::min(tile_count, max_tiles);   

    // simultaneous live threads
    const unsigned int thread_count = tile_count * tile_size;

    // global buffer (return type)
    concurrency::array<ret_type,1> global_buffer(tile_count);
    concurrency::array_view<ret_type,1> global_buffer_view(global_buffer);
    global_buffer_view.discard_data();

    // configuration
    concurrency::extent<1> extent(thread_count);

    concurrency::parallel_for_each (
        get_current_accelerator_view(), 
        extent.tile<tile_size>(),
        [=] (concurrency::tiled_index<tile_size> tid) restrict(amp)
    {
        // shared tile buffer
        tile_static ret_type local_buffer[ tile_size ];

        // index
        int idx = tid.global[0];

        // this threads's shared memory pointer
        ret_type& smem = local_buffer[ tid.local[0] ];

        // zero out local buffer
        smem = ret_type();

        // fold data into local buffer
        while (idx < n)
        {
            ret_type tempX = static_cast<ret_type>(X[ concurrency::index<1>(idx) ]);
            ret_type tempY = static_cast<ret_type>(Y[ concurrency::index<1>(idx) ]);

            smem += (tempX * tempY);

            // next chunk
            idx += thread_count;
        }

        // synchronize
        tid.barrier.wait_with_tile_static_memory_fence();

        // reduce all values in this tile
        _detail::tile_local_reduction<ret_type,tile_size>(&smem, tid, _detail::sum<ret_type>());

        // only 1 thread per tile does the inter tile communication
        if (tid.local[0] == 0)
        {
            // write to global buffer int this tiles
            global_buffer_view[ tid.tile[0] ] = smem;
        }
    });

    // 2nd pass accumulation
    std::vector<ret_type> host_buffer(global_buffer);
    return std::accumulate(host_buffer.begin(), host_buffer.end(), ret_type());
}

// Generic DOT algorithm for two AMPBLAS arrays
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

//-------------------------------------------------------------------------
// NRM2
//   computes the euclidean norm of a 1D container
//-------------------------------------------------------------------------

template <typename x_type>
typename x_type::value_type nrm2(int n, const x_type& X)
{
    typedef typename x_type::value_type T;

    // tuning sizes
    static const unsigned int tile_size = 128;
    static const unsigned int max_tiles = 64;

    auto func = _detail::nrm2_helper<T, T, x_type, _detail::sum<T>>(T(), _detail::sum<T>());

    // call generic 1D reduction
    return _detail::reduce<tile_size, max_tiles, T, T>(n, X, func);
}

// Generic NRM2 algorithm for AMPBLAS arrays of type T
template <typename value_type>
value_type nrm2(int n, const value_type *x, int incx)
{
	// quick return
	if (n <= 0) 
        return value_type();

    // check arguments
    if (x == nullptr)
		argument_error("nrm2", 2);
        
    auto x_vec = make_vector_view(n, x, incx);
    
	return nrm2(n, x_vec);
}

//-------------------------------------------------------------------------
// ROT
//-------------------------------------------------------------------------

template <int rank, typename x_type, typename y_type, typename c_type, typename s_type>
void rot(const concurrency::extent<rank>& e, x_type&& x, y_type&& y, c_type&& c, s_type&& s)
{
    concurrency::parallel_for_each(
        get_current_accelerator_view(), 
        e, 
        [=] (concurrency::index<rank> idx) restrict(amp) 
        {
            auto temp = c * x[idx] + s * y[idx];
            y[idx] = c * y[idx] - s * x[idx];
            x[idx] = temp;
        }
    );
}

template <typename value_type>
void rot(int n, value_type* x, int incx, value_type* y, int incy, value_type c, value_type s)
{
	// quick return
	if (n <= 0)
		return;

	// error check
	if (x == nullptr)
		argument_error("rot", 2);
	if (y == nullptr)
		argument_error("rot", 4);

    auto x_vec = make_vector_view(n, x, incx);
    auto y_vec = make_vector_view(n, y, incy);

    rot(make_extent(n), x_vec, y_vec, c, s); 
}

//-------------------------------------------------------------------------
// ROTG
//-------------------------------------------------------------------------

template <typename T>
void rotg(T& a, T& b, T& c, T& s)
{
    T r, roe, scale, z;
    roe = b;

    if (abs(a) > abs(b))
        roe = a;

    scale = abs(a) + abs(b);

    if (scale == T())
    {
        c = T(1);
        s = 0;
        r = 0;
        z = 0;
    }
    else
    {
        T tmpA = sqrt(a/scale);
        T tmpB = sqrt(b/scale);
        r = scale * (tmpA*tmpA + tmpB*tmpB);
        r = _detail::copysign(T(1), roe) * r;
        c = a/r;
        s = b/r;
        z = T(1);

        if (abs(a) > abs(b))
            z = s;

        if (abs(b) >= abs(a) && c != 0)
            z = T(1) / c;
    }

    a = r;
    b = z;    
}

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

//=============================================================================
// BLAS 2
//=============================================================================

//-------------------------------------------------------------------------
// GEMV
//-------------------------------------------------------------------------

template <typename alpha_type, typename x_vector_type, typename a_matrix_type, typename beta_type, typename y_vector_type>
void gemv(alpha_type alpha, const a_matrix_type& a, const x_vector_type& x, beta_type beta, y_vector_type& y)
{
	concurrency::parallel_for_each(get_current_accelerator_view(), y.extent, [=] (concurrency::index<1> y_idx) restrict(amp)
    {
        alpha_type result = alpha_type();
        
        for (int n = 0; n < x.extent[0]; ++n)
        {
       		concurrency::index<2> a_idx(n, y_idx[0]);
			concurrency::index<1> x_idx(n);

            result += a[a_idx] * x[x_idx];
        }

        y[y_idx] = alpha * result + beta * y[y_idx];
    });
}

template <typename value_type>
void gemv(enum AMPBLAS_ORDER order, enum AMPBLAS_TRANSPOSE transa, int m, int n, value_type alpha, const value_type *a, int lda, const value_type *x, int incx, value_type beta, value_type* y, int incy)
{
	// quick return
	if (m == 0 || n == 0 || (alpha == value_type() && beta == value_type(1)))
		return;

	// error check
	if (order != AmpblasColMajor)
        not_yet_implemented();
	if (transa != AmpblasNoTrans)
		not_yet_implemented();
	if (m < 0)
		argument_error("gemv", 3);
	if (n < 0)
		argument_error("gemv", 4);
	if (a == nullptr)
		argument_error("gemv", 6);
	if (lda < (transa == AmpblasNoTrans ? m : n))
		argument_error("gemv", 7);
	if (x == nullptr)
		argument_error("gemv", 8);
	if (y == nullptr)
		argument_error("gemv", 11);

	auto x_vec = make_vector_view(n, x, incx);
    auto y_vec = make_vector_view(m, y, incy);
    auto a_mat = make_matrix_view(m, n, a, lda);

	if (alpha == value_type())
	{
		if (beta == value_type())
			_detail::fill(y_vec.extent, value_type(), y_vec);
		else
			_detail::scale(y_vec.extent, beta, y_vec);

		return;
	}

	gemv(alpha, a_mat, x_vec, beta, y_vec); 
}


//-------------------------------------------------------------------------
// GER
//   performs the rank 1 operation
//
//     A := alpha*X*transpose(Y) + A,
//
//  where alpha is a scalar, X is an M element vector, Y is an N element
//  vector and A is an M by N matrix.
//-------------------------------------------------------------------------

template <typename alpha_type, typename x_vector_type, typename y_vector_type, typename a_matrix_type>
void ger(alpha_type alpha, const x_vector_type& x, const y_vector_type& y, a_matrix_type& a )
{
    concurrency::parallel_for_each ( 
        get_current_accelerator_view(), 
        a.extent,
        [=] (concurrency::index<2> idx_a) restrict(amp)
        {
            concurrency::index<1> idx_x(idx_a[1]);
            concurrency::index<1> idx_y(idx_a[0]);

            a[idx_a] += alpha * x[idx_x] * y[idx_y] ;
        }
    );
}

template <typename value_type, typename trans_op>
void ger(enum AMPBLAS_ORDER order, int m, int n, value_type alpha, const value_type *x, int incx, const value_type *y, int incy, value_type *a, int lda)
{
	// recursive order adjustment
	if (order == AmpblasRowMajor)
    {
		ger<value_type,trans_op>(AmpblasColMajor, n, m, alpha, y, incy, x, incx, a, lda);
        return;
    }

	// quick return
	if (m == 0 || n == 0 || alpha == value_type())
		return;

	// argument check
	if (m < 0)
		argument_error("ger", 2);
	if (n < 0)
		argument_error("ger", 3);
	if (x == nullptr)
		argument_error("ger", 5);
	if (y == nullptr)
		argument_error("ger", 7);
	if (a == nullptr)
		argument_error("ger", 9);
	if (lda < (order == AmpblasColMajor ? m : n))
		argument_error("ger", 10);

	// create views
	auto x_vec = make_vector_view(m, x, incx);
    auto y_vec = make_vector_view(n, y, incy);
    auto a_mat = make_matrix_view(m, n, a, lda);

	// call generic implementation
	ger(alpha, x_vec, y_vec, a_mat);
}

//-------------------------------------------------------------------------
// SYMV
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// SYR
//-------------------------------------------------------------------------

template <enum AMPBLAS_UPLO uplo, typename alpha_type, typename x_vector_type, typename a_matrix_type>
void syr(alpha_type alpha, const x_vector_type& x, a_matrix_type& a )
{
    concurrency::parallel_for_each (
        get_current_accelerator_view(),
        a.extent,
        [=] (concurrency::index<2> idx_a) restrict(amp)
        {
            concurrency::index<1> idx_x(idx_a[1]); // "i"
            concurrency::index<1> idx_xt(idx_a[0]); // "j"

            if ( uplo == AmpblasUpper && idx_a[0] >= idx_a[1] ||
                 uplo == AmpblasLower && idx_a[1] >= idx_a[0]
               )
                a[idx_a] += alpha * x[idx_x] * x[idx_xt];
        }
    );
}

template <typename value_type>
void syr(enum AMPBLAS_ORDER order, enum AMPBLAS_UPLO uplo, int n, value_type alpha, const value_type *x, int incx, value_type *a, int lda)
{
	// recursive order adjustment
	if (order == AmpblasRowMajor) // todo: implement row major
    {
        enum AMPBLAS_UPLO opposite = uplo == AmpblasUpper ? AmpblasLower : AmpblasUpper;
		syr<value_type>(AmpblasRowMajor, opposite, n, alpha, x, incx, a, lda);
        return;
    }

	// quick return
	if (n == 0 || alpha == value_type())
		return;

	// argument check
	if (n < 0)
		argument_error("syr", 3);
	if (x == nullptr)
		argument_error("syr", 5);
	if (a == nullptr)
		argument_error("syr", 7);
	if (lda < n)
		argument_error("syr", 8);

	// create views
	auto x_vec = make_vector_view(n, x, incx);
    auto a_mat = make_matrix_view(n, n, a, lda);

	// call generic implementation
	if ( uplo == AmpblasUpper )
	    syr<AmpblasUpper>(alpha, x_vec, a_mat);
	else
	    syr<AmpblasLower>(alpha, x_vec, a_mat);
}

//-------------------------------------------------------------------------
// TRMV
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// TRSV
//-------------------------------------------------------------------------

//=============================================================================
// BLAS 3
//=============================================================================

//-------------------------------------------------------------------------
// GEMM
//-------------------------------------------------------------------------

// Generic GEMM algorithm on AMP array_views of type value_type
template <typename alpha_type, typename a_matrix_type, typename b_matrix_type, typename beta_type, typename c_matrix_type>
void gemm(enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_TRANSPOSE transb, alpha_type alpha, const a_matrix_type& a, const b_matrix_type& b, beta_type beta, c_matrix_type& c)
{
    // matrix a is assumed column-major. a.extent<2> = [ a_col, a_lda ]
	int k_max = (transa == AmpblasNoTrans ? a.extent[0] : a.extent[1]);

	concurrency::parallel_for_each(get_current_accelerator_view(), c.extent, [=] (concurrency::index<2> c_idx) restrict(amp)
	{
		alpha_type result = alpha_type();

		for (int k = 0; k < k_max; ++k)
		{
			concurrency::index<2> a_idx = (transa == AmpblasNoTrans ? concurrency::index<2>(k, c_idx[1]) : concurrency::index<2>(c_idx[1], k));
			concurrency::index<2> b_idx = (transb == AmpblasNoTrans ? concurrency::index<2>(c_idx[0], k) : concurrency::index<2>(k, c_idx[0]));

			result += a[a_idx] * b[b_idx];
		}

		c[c_idx] = alpha * result + beta * c[c_idx];
	});
}

template <typename value_type>
void gemm(enum AMPBLAS_ORDER order, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_TRANSPOSE transb, int m, int n, int k, value_type alpha, const value_type *a, int lda, const value_type *b, int ldb, value_type beta, value_type *c, int ldc) 
{
	// recursive order adjustment 
	if (order == AmpblasRowMajor)
        gemm(AmpblasColMajor, transb, transa, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);

    // quick return
    if ((m == 0 || n == 0 || alpha == value_type() || k == 0) && beta == value_type(1))
        return;

	// error check
	if (m < 0)		       
		argument_error("GEMM", 4);
	if (n < 0)        
		argument_error("GEMM", 5);
	if (k < 0)        
		argument_error("GEMM", 6);
	if (a == nullptr) 
		argument_error("GEMM", 8);
	if (lda < ((order == AmpblasRowMajor && transa == AmpblasNoTrans || order == AmpblasColMajor && transa == AmpblasTrans) ? k : m))
		argument_error("GEMM", 9);
	if (b == nullptr) 
		argument_error("GEMM", 10);
	if (ldb < ((order == AmpblasRowMajor && transb == AmpblasNoTrans || order == AmpblasColMajor && transb == AmpblasTrans) ? n : k)) 
		argument_error("GEMM", 11);
	if (c == nullptr) 
		argument_error("GEMM", 13);
	if (ldc < (order == AmpblasRowMajor ? n : m)) 
		argument_error("GEMM", 14);

	auto a_row = (transa == AmpblasNoTrans ? m : k);
	auto a_col = (transa == AmpblasNoTrans ? k : m);   
	auto b_row = (transb == AmpblasNoTrans ? k : n);
	auto b_col = (transb == AmpblasNoTrans ? n : k);
  
	auto a_mat = make_matrix_view(a_row, a_col, a, lda);
	auto b_mat = make_matrix_view(b_row, b_col, b, ldb);
	auto c_mat = make_matrix_view(m, n, c, ldc);

	if (alpha == value_type())
	{
		if (beta == value_type())
			_detail::fill(c_mat.extent, value_type(), c_mat);
		else
			_detail::scale(c_mat.extent, beta, c_mat);

		return;
	}

    gemm(transa, transb, alpha, a_mat, b_mat, beta, c_mat);
}

//-------------------------------------------------------------------------
// SYMM
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// SYRK
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// SYR2K
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// TRSM
// (Incomplete!)
//-------------------------------------------------------------------------

template <unsigned int tile_size, typename value_type>
void trsm_left(enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, value_type alpha, const concurrency::array_view<value_type,2>& a, concurrency::array_view<value_type,2> b) 
{
    // TODO: support transposes
    // TODO: proper wait() types
    // TODO: combine pathes with better loop parameterization

    // runtime sizes
    unsigned int m = b.extent[1];
    unsigned int n = b.extent[0];
    unsigned int tiles = (n+tile_size-1)/tile_size;

    // bounds checking not implemented yet
    assert(m % tile_size == 0 && n % tile_size == 0);

    // configuration
    auto e = make_extent(tile_size, tile_size*tiles);
	
    concurrency::parallel_for_each ( 
		get_current_accelerator_view(), 
        e.tile<tile_size,tile_size>(),
        [=] (concurrency::tiled_index<tile_size,tile_size> tid) restrict(amp)
        {
            // shared memory buffers
            tile_static value_type a_tile[tile_size][tile_size];
            tile_static value_type b_tile[tile_size][tile_size];

            // local indexes
            const int col = tid.local[0];
            const int row = tid.local[1];

            // per-thread common alias
            value_type& a_local = a_tile[row][col];
            value_type& b_local = b_tile[row][col];

            // global j index
            unsigned int j = tid.tile_origin[0];

            // lower + no trans <==> upper + trans 
            if ((uplo == AmpblasLower) ^ (transa != AmpblasNoTrans))
            {
                // loop down by tiles
                for (unsigned int i=0; i<m; i+=tile_size)
                {
                    // read tile at A(i,i) into local A
                    a_local = a(concurrency::index<2>(i+row, i+col));

                    // read tile at B(i,j) into local B
                    b_local = b(concurrency::index<2>(j+row, i+col));
                    tid.barrier.wait();

                    // solve X(i,j) = B(i,j) \ A(i,i)
                    if (col == 0)
                    {
                        // TODO: consider making this a function
                        unsigned int jj = row;

                        // loop down shared block
                        for (unsigned int ii=0; ii<tile_size; ii++)
                        {
                            // elimation scalar
                            value_type alpha = b_tile[jj][ii];
                            if (diag == AmpblasNonUnit)
                                alpha /= a_tile[ii][ii];

                            b_tile[jj][ii] = alpha;

                            // apply
                            for (unsigned int kk=ii+1; kk<tile_size; kk++)
                                b_tile[jj][kk] -= alpha * a_tile[ii][kk];
                        }
                    }

                    // wait for local solve
                    tid.barrier.wait();

                    // write B(i,j)
                    b(concurrency::index<2>(j+row, i+col)) = alpha * b_local;

                    // apply B(k,j) -= B(i,j) * A(k,i) 
                    for (unsigned int k=i+tile_size; k<m; k+=tile_size)
                    {   
                        // read tile at A(k,i) into local A
                        a_local = a(concurrency::index<2>(i+row, k+col));
                        tid.barrier.wait();

                        // accumulate
                        value_type sum = value_type();

                        // TODO: unrollable?
                        for (int l=0; l<tile_size; l++)
                            sum += a_tile[l][col] * b_tile[row][l];

                        // update
                        b(concurrency::index<2>(j+row, k+col)) -= sum;

                        // wait for a to finish being read
                        tid.barrier.wait();
                    }
                }
            }

            // lower + trans <==> upper + no trans 
            else
            {
               // TODO
            }
        }
    );
}

template <unsigned int tile_size, typename alpha_type, typename a_matrix_type, typename b_matrix_type>
void trsm(enum AMPBLAS_SIDE side, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, alpha_type alpha, const a_matrix_type& a, b_matrix_type& b)
{
	if (side == AmpblasLeft)
    {
        // assumed lower for now
        trsm_left<tile_size>(uplo, transa, diag, alpha, a, b);
	}
	else
	{
	}

}

template <typename value_type>
void trsm(enum AMPBLAS_ORDER /*order*/, enum AMPBLAS_SIDE side, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, int m, int n, value_type alpha, const value_type *a, int lda, value_type *b, int ldb)
{
	// tuning sizes
    static const unsigned int tile_size = 16;

	// recursive order adjustment 
	// TODO:

    // quick return
	// TODO: 

	// derived parameters
	int k = (side == AmpblasLeft ? m : n);

	// error check
	// TODO:
  
	// amp data
	auto a_mat = make_matrix_view(k, k, a, lda);
	auto b_mat = make_matrix_view(m, n, b, ldb);

	// special paths
	// TODO:

	// implementation
	trsm<tile_size>(side, uplo, transa, diag, alpha, a_mat, b_mat);
}

//-------------------------------------------------------------------------
// TRMM
//-------------------------------------------------------------------------


} // namespace ampblas
#endif // __cplusplus
#endif //AMPBLAS_H

