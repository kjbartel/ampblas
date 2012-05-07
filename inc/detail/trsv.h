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
 * trsv.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {
namespace _detail {

template <int tile_size, typename value_type, typename x_vector_type> 
void trsv_l(enum AMPBLAS_TRANSPOSE transa, const enum AMPBLAS_DIAG diag, const concurrency::array_view<const value_type,2>& a, x_vector_type& x)
{
    const int n = x.extent[0];

    concurrency::parallel_for_each(get_current_accelerator_view(), make_extent(tile_size).tile<tile_size>(), [=] (concurrency::tiled_index<tile_size> tid) restrict(amp)
    {
        for (int j=0; j<n; j++)
        {
            if (diag == AmpblasNonUnit)
            {
                if (tid.local[0] == 0)
                    x[concurrency::index<1>(j)] /= a[concurrency::index<2>(j,j)];

                tid.barrier.wait();
            }

            value_type alpha = x[concurrency::index<1>(j)];

            for (int i=tid.local[0]+j+1; i<n; i+=tile_size)
                x[concurrency::index<1>(i)] -= alpha * a[transa == AmpblasNoTrans ? concurrency::index<2>(j,i) : concurrency::index<2>(i,j)];

            tid.barrier.wait();
        }
    });
}

template <int tile_size, typename value_type, typename x_vector_type> 
void trsv_u(enum AMPBLAS_TRANSPOSE transa, const enum AMPBLAS_DIAG diag, const concurrency::array_view<const value_type,2>& a, x_vector_type& x)
{
    int n = x.extent[0];

    // compiler work around
    const int dummy = 1;

    concurrency::parallel_for_each(get_current_accelerator_view(), make_extent(tile_size).tile<tile_size>(), [=] (concurrency::tiled_index<tile_size> tid) restrict(amp)
    {
        for (int jj=n-1; jj>=0; jj--)
        {
            // compiler work around
            int j = dummy ? jj : 0;

            if (diag==AmpblasNonUnit)
            {
                if (tid.local[0] == 0)
                    x[concurrency::index<1>(j)] /= a[concurrency::index<2>(j,j)];

                tid.barrier.wait();
            }

            value_type alpha = x[concurrency::index<1>(j)];

            for (int i=tid.local[0]; i<j; i+=tile_size)
                x[concurrency::index<1>(i)] -= alpha * a[transa == AmpblasNoTrans ? concurrency::index<2>(j,i) : concurrency::index<2>(i,j)];
            tid.barrier.wait();
        }
    });
}

} // namespace _detail

//-------------------------------------------------------------------------
// TRSV
//  The current implementation of this routine has minimial parallelism and
//  should only be used as a building block to enable larger routines.
//-------------------------------------------------------------------------

template <typename value_type, typename x_vector_type> 
void trsv(enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, const concurrency::array_view<const value_type,2>& a, x_vector_type& x)
{
    // tuning parameters
    const int tile_size = 256;

    if ((uplo == AmpblasLower) ^ (transa != AmpblasNoTrans))
    {
        // lower + no trans <==> upper + trans
        _detail::trsv_l<tile_size>(transa, diag, a, x);
    }
    else
    {
        // upper + no trans <==> lower + trans
        _detail::trsv_u<tile_size>(transa, diag, a, x);
    }
}

template <typename value_type>
void trsv(enum AMPBLAS_ORDER order, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, int n, const value_type* a, int lda, value_type* x, int incx)
{
    // recursive order adjustment
	if (order == AmpblasRowMajor)
    {
        trsv(AmpblasColMajor, uplo == AmpblasUpper ? AmpblasLower : AmpblasUpper, transa == AmpblasNoTrans ? AmpblasTrans : transa, diag, n, a, lda, x, incx);
        return;
    }

	// quick return
	if (n == 0)
		return;

	// argument check
	if (n < 0)
        argument_error("trsv", 5);
    if (a == nullptr)
        argument_error("trsv", 6);
    if (lda < n)
        argument_error("trsv", 7);
    if (x == nullptr)
        argument_error("trsv", 7);

	// create views
	auto x_vec = make_vector_view(n, x, incx);
    auto a_mat = make_matrix_view(n, n, a, lda);

    // forward to tuning routine
    trsv(uplo, transa, diag, a_mat, x_vec);
}

} // namespace ampblas
