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
// The functions in the _detail namespace are used internally for other BLAS 
// functions internally. 
namespace _detail
{
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

//-------------------------------------------------------------------------
// SCAL
//-------------------------------------------------------------------------

// Generic SCAL algorithm on AMP array_view of type T
template <int rank, typename alpha_type, typename value_type>
void scal__1(alpha_type&& alpha, concurrency::array_view<value_type, rank>& av)
{
    concurrency::parallel_for_each(get_current_accelerator_view(), av.extent, [=] (concurrency::index<rank> idx) restrict(amp) 
    {
        av[idx] = alpha * av[idx];
    });
}

// Generic SCAL algorithm for AMPBLAS matrix of type T 
// All arguments are valid
template <typename T>
void scal_impl__1(const enum AMPBLAS_ORDER Order, 
                  const int M, const int N, const T alpha, 
                  const T  *C, const int ldc)
{
    assert(M >= 0 && N >= 0 && C != nullptr && ldc >= (Order == CblasRowMajor ? N : M));

    if (M == 0 || N == 0 || alpha == T(1.0)) 
    {
        return;
    }

    int row = M;
    int col = N;

    if (Order == CblasColMajor)
    {
        std::swap(row, col);
    }

    auto avC = get_array_view(C, row*ldc).view_as(concurrency::extent<2>(row, ldc)).section(concurrency::extent<2>(row, col));
    scal__1<2>(alpha, avC);
}

// Generic SCAL algorithm for AMPBLAS matrix of type T
template <typename T>
void scal(const enum AMPBLAS_ORDER Order, 
          const int M, const int N, const T alpha, 
          const T *C,  const int ldc)
{
    if (M < 0 || N < 0 || C == nullptr || ldc < (Order == CblasRowMajor ? N : M))
    {
        throw ampblas_exception("Invalid argument in scal", AMPBLAS_INVALID_ARG);
    }

    scal_impl(Order, M, N, alpha, C, ldc);
}

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

template <typename T>
T asum(const int N, const T* X, const int incX)
{
    concurrency::array_view<T> avX = get_array_view(X, N*abs(incX));

    if (incX == 1)
    {
        return asum(N, avX);
    }
    else
    {
        auto avX1 = make_stride_view(avX, incX, make_extent(N));
        return asum(N, avX1);
    }
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

template <typename T>
int amax(const int N, const T* X, const int incX)
{
    concurrency::array_view<T> avX = get_array_view(X, N*abs(incX));

    if (incX == 1)
    {
        return amax<int>(N, avX);
    }
    else
    {
        auto avX1 = make_stride_view(avX, incX, make_extent(N));
        return amax<int>(N, avX1);
    }
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
template <typename T>
void axpy(const int N, const T alpha, const T *X, const int incX, T *Y, const int incY)
{
    // check arguments
    if (X == nullptr)
    {
        throw ampblas_exception("The 3rd argument in axpy is invalid", AMPBLAS_INVALID_ARG);
    }

    if (Y == nullptr)
    {
        throw ampblas_exception("The 5th argument in axpy is invalid", AMPBLAS_INVALID_ARG);
    }

    if (N <= 0 || alpha == T())
    {
        return;
    }

    concurrency::array_view<T> avX = get_array_view(X, N*abs(incX));
    concurrency::array_view<T> avY = get_array_view(Y, N*abs(incY));

    if (incX == 1 && incY == 1)
    {
        axpy(make_extent(N), alpha, avX, avY);
    }
    else
    {
        auto avX1 = make_stride_view(avX, incX, make_extent(N));
        auto avY1 = make_stride_view(avY, incY, make_extent(N));

        axpy(make_extent(N), alpha, avX1, avY1);
    }
}

//-------------------------------------------------------------------------
// COPY
//   copy a container to another container. The two containers cannot be 
//   overlpped.
//-------------------------------------------------------------------------

// Generic COPY algorithm on any multi-dimensional container
template <int rank, typename x_type, typename y_type>
void copy(const concurrency::extent<rank>& e, x_type&& X, y_type&& Y)
{
    concurrency::parallel_for_each(get_current_accelerator_view(), e, [=] (concurrency::index<rank> idx) restrict(amp) 
    {
        Y[idx] = X[idx];
    });
}

// Generic COPY algorithm for AMPBLAS arrays of type T
template <typename T>
void copy(const int N, const T *X, const int incX, T *Y, const int incY)
{
    // check arguments
    if (X == nullptr)
    {
        throw ampblas_exception("The 2nd argument in copy is invalid", AMPBLAS_INVALID_ARG);
    }

    if (Y == nullptr)
    {
        throw ampblas_exception("The 4th argument in copy is invalid", AMPBLAS_INVALID_ARG);
    }

    if (N <= 0 || X == Y) 
    {
        return;
    }

    concurrency::array_view<T> avX = get_array_view(X, N*abs(incX));
    concurrency::array_view<T> avY = get_array_view(Y, N*abs(incY));

    if (incX == 1 && incY == 1)
    {
        avX.copy_to(avY);
    }
    else
    {
        auto avX1 = make_stride_view(avX, incX, make_extent(N));
        auto avY1 = make_stride_view(avY, incY, make_extent(N));

        copy(make_extent(N), avX1, avY1);
    }
}

//-------------------------------------------------------------------------
// DOT
//   computes the dot product of two 1D arrays.
//-------------------------------------------------------------------------

template <typename array_type, typename ret_type, typename Operator>
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

// Generic DOT algorithm for AMPBLAS arrays of type T
template <typename T, typename U, typename Operator>
U dot(const int N, const T* X, const int incX, const T* Y, const int incY)
{
    // check arguments
    if (N <= 0) 
        return T();

    if (X == nullptr)
        throw ampblas_exception("The 2rd argument in dot is invalid", AMPBLAS_INVALID_ARG);

    if (Y == nullptr)
        throw ampblas_exception("The 4th argument in dot is invalid", AMPBLAS_INVALID_ARG);

    auto avX = get_array_view(X, N*abs(incX));
    auto avY = get_array_view(Y, N*abs(incY));

    auto svvX = make_stride_view(avX, incX, make_extent(N));
    auto svvY = make_stride_view(avY, incY, make_extent(N));

    return dot< stride_view<concurrency::array_view<T>>, U, Operator >(N, svvX, svvY);
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

template <typename VectorType, typename MatrixType>
void ger(int M, int N, typename VectorType::value_type alpha, VectorType& X, VectorType& Y, MatrixType& A)
{
    // configuration
    concurrency::extent<2> extent(N, M);

    concurrency::parallel_for_each (
        get_current_accelerator_view(), 
        extent,
        [=] (concurrency::index<2> idx) restrict(amp)
    {
        A[ idx ] += alpha * X[ idx[1] ] * Y[ idx[0] ] ;
    });
}

template <typename T, typename Operator>
void ger(enum AMPBLAS_ORDER order, int M, int N, T alpha, const T* X, int incX, const T* Y, int incY, T* A, int ldA)
{
    // TODO: NYI
    assert(order == CblasColMajor);

    auto avX = get_array_view(X, M * abs(incX));
    auto avY = get_array_view(Y, N * abs(incY));

    // TODO: functionize this
    auto avA = get_array_view(A, ldA*N).view_as(concurrency::extent<2>(N,M)).section(concurrency::extent<2>(N,M));

    ger(M, N, alpha, avX, avY, avA);
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
template <typename T>
T nrm2(const int N, const T *X, const int incX)
{
    // check arguments
    if (X == nullptr)
    {
        throw ampblas_exception("The 2nd argument in nrm2 is invalid", AMPBLAS_INVALID_ARG);
    }

    if (N <= 0) 
    {
        return T();
    }

    auto avX = get_array_view(X, N*abs(incX));
    auto svvX = make_stride_view(avX, incX, make_extent(N));

    return nrm2(N, svvX);
}

//-------------------------------------------------------------------------
// ROT
//-------------------------------------------------------------------------

template <typename VectorType>
void rot(int N, VectorType& X, VectorType& Y, typename VectorType::value_type c, typename VectorType::value_type s)
{
    typedef typename VectorType::value_type T;

    concurrency::extent<1> extent(N);

    concurrency::parallel_for_each(
        get_current_accelerator_view(), 
        extent, 
        [=] (concurrency::index<1> idx) restrict(amp) 
    {
        T temp = c * X[idx] + s * Y[idx];
        Y[idx] = c * Y[idx] - s * X[idx];
        X[idx] = temp;
    });
}

template <typename T>
void rot(int N, T* X, int incX, T* Y, int incY, T c, T s)
{
    concurrency::array_view<T> avX = get_array_view(X, N*abs(incX));
    concurrency::array_view<T> avY = get_array_view(Y, N*abs(incY));

    rot(N, avX, avY, c, s); 
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

// Generic SCAL algorithm on any multi-dimensional container
template <int rank, typename alpha_type, typename x_type>
void scal(const concurrency::extent<rank>& e, alpha_type&& alpha, x_type&& X)
{
    concurrency::parallel_for_each(get_current_accelerator_view(), e, [=] (concurrency::index<rank> idx) restrict(amp) 
    {
        X[idx] *= alpha;
    });
}

// Generic SCAL algorithm for AMPBLAS arrays of type T
template <typename T>
void scal(const int N, const T alpha, T *X, const int incX)
{
    // check arguments
    if (X == nullptr)
    {
        throw ampblas_exception("The 3rd argument in copy is invalid", AMPBLAS_INVALID_ARG);
    }

    if (N <= 0) 
    {
        return;
    }

    concurrency::array_view<T> avX = get_array_view(X, N*abs(incX));

    if (incX == 1)
    {
        scal(make_extent(N), alpha, avX);
    }
    else
    {
        auto avX1 = make_stride_view(avX, incX, make_extent(N));

        scal(make_extent(N), alpha, avX1);
    }
}

//-------------------------------------------------------------------------
// SWAP
//   The input buffers or containers cannot overlap with each other. Otherwise,
// runtime will throw an ampblas_exception when the buffers are bound. 
//-------------------------------------------------------------------------

// Generic SWAP algorithm on any multi-dimensional container
template <int rank, typename x_type, typename y_type>
void swap(const concurrency::extent<rank>& e, x_type&& X, y_type&& Y)
{
    concurrency::parallel_for_each(get_current_accelerator_view(), e, [=] (concurrency::index<rank> idx) restrict(amp) 
    {
        auto tmp = Y[idx];
        Y[idx] = X[idx];
        X[idx] = tmp;
    });
}

// Generic SWAP algorithm for AMPBLAS arrays of type T
template <typename T>
void swap(const int N, T *X, const int incX, T *Y, const int incY)
{
    // check arguments
    if (X == nullptr)
    {
        throw ampblas_exception("The 2nd argument in swap is invalid", AMPBLAS_INVALID_ARG);
    }

    if (Y == nullptr)
    {
        throw ampblas_exception("The 4th argument in swap is invalid", AMPBLAS_INVALID_ARG);
    }

    if (N <= 0 || X == Y) 
    {
        return;
    }

    concurrency::array_view<T> avX = get_array_view(X, N*abs(incX));
    concurrency::array_view<T> avY = get_array_view(Y, N*abs(incY));

    if (incX == 1 && incY == 1)
    {
        swap(make_extent(N), avX, avY);
    }
    else
    {
        auto avX1 = make_stride_view(avX, incX, make_extent(N));
        auto avY1 = make_stride_view(avY, incY, make_extent(N));

        swap(make_extent(N), avX1, avY1);
    }
}

//=============================================================================
// BLAS 3
//=============================================================================

//-------------------------------------------------------------------------
// GEMM
//-------------------------------------------------------------------------

// All kernels are implemented assuming RowMajor ordering, but are used to 
// execute the corresponding ColumnMajor gemms. Here is the mapping:
//
//   RowMajor [type]: <--> Tranposed Equation: <--> ColumnMajor [type]: 
//   ----------------      -------------------      -------------------
//   C = A B     [NN]      C^T = B^T A^T            C = B A     [NN]
//   C = A^T B   [TN]      C^T = B^T A              C = B A^T   [NT]
//   C = A B^T   [NT]      C^T = B A^T              C = B^T A   [TN]
//   C = A^T B^T [TT]      C^T = B A                C = B^T A^T [TT]
//

// Generic GEMM algorithm on AMP array_views of type value_type
template <typename alpha_type, typename beta_type, typename value_type>
void gemm(enum AMPBLAS_TRANSPOSE TransA, 
    enum AMPBLAS_TRANSPOSE TransB, 
    alpha_type&& alpha, 
    beta_type&& beta, 
    const concurrency::array_view<value_type, 2>& avA, 
    const concurrency::array_view<value_type, 2>& avB, 
    concurrency::array_view<value_type, 2>& avC)
{
    concurrency::parallel_for_each(get_current_accelerator_view(), avC.extent, [=] (concurrency::index<2> idx) restrict(amp)
    {
        value_type result = value_type();

        for(int k = 0; k < avA.extent[1]; ++k)
        {
            concurrency::index<2> idxA = (TransA == CblasNoTrans ? concurrency::index<2>(idx[0], k) : concurrency::index<2>(k, idx[0]));
            concurrency::index<2> idxB = (TransB == CblasNoTrans ? concurrency::index<2>(k, idx[1]) : concurrency::index<2>(idx[1], k));

            result += avA[idxA] * avB[idxB];
        }

        avC[idx] = alpha * result + beta * avC[idx];
    });
}

// Matrices A, B and C are all RowMajor 
// TODO: handle conjugate transpose 
template<typename T>
void gemm_impl(const enum AMPBLAS_TRANSPOSE TransA, 
    const enum AMPBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K, 
    const T alpha, const T *A, const int lda, 
    const T *B, const int ldb,
    const T beta, T *C, const int ldc) 
{
    auto row_a = (TransA == CblasNoTrans ? M : K);
    auto col_a = (TransA == CblasNoTrans ? K : M);
    auto row_b = (TransB == CblasNoTrans ? K : N);
    auto col_b = (TransB == CblasNoTrans ? N : K);

    auto avA = get_array_view(A, row_a*lda).view_as(concurrency::extent<2>(row_a, lda)).section(concurrency::extent<2>(row_a, col_a));
    auto avB = get_array_view(B, row_b*ldb).view_as(concurrency::extent<2>(row_b, ldb)).section(concurrency::extent<2>(row_b, col_b));
    auto avC = get_array_view(C, M*ldc).view_as(concurrency::extent<2>(M, ldc)).section(concurrency::extent<2>(M, N));

    gemm(TransA, TransB, alpha, beta, avA, avB, avC);
}

template<typename T>
void gemm_impl(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA,
    const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N,
    const int K, const T alpha, const T *A,
    const int lda, const T *B, const int ldb,
    const T beta, T *C, const int ldc) 
{
    // Quick return
    if (M == 0 || N == 0 || (alpha == T() || K == 0) && beta == T(1.0))
    {
        return;
    }

    if (alpha == T())
    {
        _detail::scal_impl__1(Order, M, N, beta, C, ldc);
        return;
    }

    // Normal operation
    if (Order == CblasRowMajor)
    {
        gemm_impl(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    else // CblasColMajor
    {
        gemm_impl(TransB, TransA, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
    }
}

// Generic GEMM algorithm for AMPBLAS arrays of type T
template<typename T>
void gemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA,
    const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N,
    const int K, const T alpha, const T *A,
    const int lda, const T *B, const int ldb,
    const T beta, T *C, const int ldc)
{
    if (M < 0 || 
        N < 0 || 
        K < 0 || 
        A == nullptr || 
        B == nullptr || 
        C == nullptr ||
        lda < ((Order == CblasRowMajor && TransA == CblasNoTrans ||
        Order == CblasColMajor && TransA == CblasTrans) ? K : M) ||
        ldb < ((Order == CblasRowMajor && TransB == CblasNoTrans ||
        Order == CblasColMajor && TransB == CblasTrans) ? N : K) ||
        ldc < (Order == CblasRowMajor ? N : M))
    {
        throw ampblas_exception("Invalid argument in gemm", AMPBLAS_INVALID_ARG);
    }

    gemm_impl(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}


} // namespace ampblas
#endif // __cplusplus
#endif //AMPBLAS_H

