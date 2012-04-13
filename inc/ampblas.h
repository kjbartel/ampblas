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
#include "ampblas_complex.h"
#include "ampblas_runtime.h"

namespace ampblas
{
// The functions in the _detail namespace are used internally for other BLAS 
// functions internally. 
namespace _detail
{
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
} // namespace _detail

//-------------------------------------------------------------------------
// AXPY
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
void axpy(const int N, const T alpha, const T *X,
          const int incX, T *Y, const int incY)
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

    concurrency::array_view<T> avX = get_array_view(X, N);
    concurrency::array_view<T> avY = get_array_view(Y, N);

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

    concurrency::array_view<T> avX = get_array_view(X, N);
    concurrency::array_view<T> avY = get_array_view(Y, N);

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
// SWAP
//   The input buffers or containers can overlap with each other. Otherwise,
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

    concurrency::array_view<T> avX = get_array_view(X, N);
    concurrency::array_view<T> avY = get_array_view(Y, N);

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

    concurrency::array_view<T> avX = get_array_view(X, N);

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

