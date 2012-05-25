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
 * trmm.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {
namespace _detail {

template <int tile_size, typename value_type>
void trmm(const concurrency::accelerator_view& av, enum AMPBLAS_SIDE side, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, int m, int n, value_type alpha, const concurrency::array_view<const value_type,2>& a_mat, const concurrency::array_view<const value_type,2>& b_mat, const concurrency::array_view<value_type,2>& c_mat )
{
    // pad() has undesirable functionality - pads even when unnecessary
    // auto e = c_mat.extent.tile<16,16>().pad();

    int tiles_m = (m+tile_size-1)/tile_size;
    int tiles_n = (n+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size*tiles_m, tile_size*tiles_n);

    if (side == AmpblasLeft)
    {
        if ((uplo == AmpblasLower) ^ (transa != AmpblasNoTrans))
        {
            concurrency::parallel_for_each (
                av,
                e.tile<tile_size,tile_size>(),
                [=] (concurrency::tiled_index<tile_size,tile_size> idx_b) restrict(amp)
                {
                    tile_static value_type a[tile_size][tile_size]; // "a" tile
                    tile_static value_type b[tile_size][tile_size]; // "b" tile

                    const int i = idx_b.local[1];
                    const int j = idx_b.local[0];
                    const int tile_i = idx_b.tile[1];
                    const int tile_i_origin = idx_b.tile_origin[1];
                    const int global_i = idx_b.global[1];
                    const int global_j = idx_b.global[0];

                    value_type& a_local = (transa == AmpblasNoTrans ? a[i][j] : a[j][i]);

                    value_type out = value_type();

                    auto tile_origin = 0;
                    for ( auto tile=0; tile<=tile_i; ++tile, tile_origin+=tile_size )
                    {
                        if ( tile == tile_i )
                        {
                            // diagonal tile, treat specially
                            auto a_idx = concurrency::index<2>(tile_origin+j,tile_origin+i);
                            if ( ( transa == AmpblasNoTrans && i >= j ) || ( transa != AmpblasNoTrans && i <= j ) )
                                a_local = ( diag == AmpblasUnit && i==j ) ? value_type(1) : (_detail::guarded_read<true>(a_mat,a_idx));
                            else
                                a_local = value_type();
                        }
                        else
                        {
                            // off diagonal, load
                            if ( transa == AmpblasNoTrans )
                            {
                                auto a_idx = concurrency::index<2>(tile_origin+j,global_i);
                                a_local = _detail::guarded_read<true>(a_mat,a_idx);
                            }
                            else
                            {
                                auto a_idx = concurrency::index<2>(tile_i_origin+j,tile_origin+i);
                                a_local = _detail::guarded_read<true>(a_mat,a_idx);
                            }
                        }
                        auto b_idx = concurrency::index<2>(global_j, tile_origin+i);
                        b[i][j] = _detail::guarded_read<true>(b_mat,b_idx);

                        idx_b.barrier.wait_with_tile_static_memory_fence();

                        int end = _detail::min(tile_size,m-tile_origin);
                        for ( int k=0; k<end; ++k )
                            out += alpha*a[i][k]*b[k][j];

                        idx_b.barrier.wait_with_tile_static_memory_fence();
                    }
                    if ( global_i<m && global_j<n )
                    {
                        c_mat[idx_b] = out;
                    }
                }
            );
        }
        else // upper + notrans or lower + trans
        {
            concurrency::parallel_for_each (
                av,
                e.tile<tile_size,tile_size>(),
                [=] (concurrency::tiled_index<tile_size,tile_size> idx_b) restrict(amp)
                {
                    tile_static value_type a[tile_size][tile_size]; // "a" tile
                    tile_static value_type b[tile_size][tile_size]; // "b" tile

                    const int i = idx_b.local[1];
                    const int j = idx_b.local[0];
                    const int tile_i = idx_b.tile[1];
                    const int tile_i_origin = idx_b.tile_origin[1];
                    const int global_i = idx_b.global[1];
                    const int global_j = idx_b.global[0];

                    value_type& a_local = (transa == AmpblasNoTrans ? a[i][j] : a[j][i]);

                    value_type out = value_type();

                    auto tile_origin = tile_i*tile_size;
                    for ( auto tile=tile_i; tile<tiles_m; ++tile, tile_origin+=tile_size )
                    {
                        if ( tile == tile_i )
                        {
                            // diagonal tile, treat specially
                            auto a_idx = concurrency::index<2>(tile_origin+j,tile_origin+i);
                            if ( ( transa == AmpblasNoTrans && i <= j ) || ( transa != AmpblasNoTrans && i >= j ) )
                                a_local = ( diag == AmpblasUnit && i==j ) ? value_type(1) : guarded_read<true>(a_mat,a_idx);
                            else
                                a_local = value_type();
                        }
                        else
                        {
                            // off diagonal, load
                            if ( transa == AmpblasNoTrans )
                            {
                                auto a_idx = concurrency::index<2>(tile_origin+j,global_i);
                                a_local = guarded_read<true>(a_mat,a_idx);
                            }
                            else
                            {
                                auto a_idx = concurrency::index<2>(tile_i_origin+j,tile_origin+i);
                                a_local = guarded_read<true>(a_mat,a_idx);
                            }
                        }
                        auto b_idx = concurrency::index<2>(global_j, tile_origin+i);
                        b[i][j] = guarded_read<true>(b_mat,b_idx);

                        idx_b.barrier.wait_with_tile_static_memory_fence();

                        int end = _detail::min(tile_size,m-tile_origin);
                        for ( int k=0; k<end; ++k )
                            out += alpha*a[i][k]*b[k][j];

                        idx_b.barrier.wait_with_tile_static_memory_fence();
                    }
                    if ( global_i<m && global_j<n )
                    {
                        c_mat[idx_b] = out;
                    }
                }
            );
        }
    }
    else // right
    {
        if ((uplo == AmpblasLower) ^ (transa != AmpblasNoTrans))
        {
            concurrency::parallel_for_each (
                av,
                e.tile<tile_size,tile_size>(),
                [=] (concurrency::tiled_index<tile_size,tile_size> idx_b) restrict(amp)
                {
                    tile_static value_type a[tile_size][tile_size]; // "a" tile
                    tile_static value_type b[tile_size][tile_size]; // "b" tile

                    const int i = idx_b.local[1];
                    const int j = idx_b.local[0];
                    const int tile_j = idx_b.tile[0];
                    const int tile_j_origin = idx_b.tile_origin[0];
                    const int global_i = idx_b.global[1];
                    const int global_j = idx_b.global[0];

                    value_type& a_local = (transa == AmpblasNoTrans ? a[i][j] : a[j][i]);

                    value_type out = value_type();

                    auto tile_origin = tile_j*tile_size;
                    for ( auto tile=tile_j; tile<tiles_n; ++tile, tile_origin+=tile_size )
                    {
                        if ( tile == tile_j )
                        {
                            // diagonal tile, treat specially
                            auto a_idx = concurrency::index<2>(tile_origin+j,tile_origin+i);
                            if ( ( transa == AmpblasNoTrans && i >= j ) || ( transa != AmpblasNoTrans && i <= j ) )
                                a_local = ( diag == AmpblasUnit && i==j ) ? value_type(1) : guarded_read<true>(a_mat,a_idx);
                            else
                                a_local = value_type();
                        }
                        else
                        {
                            // off diagonal, load
                            if ( transa == AmpblasNoTrans )
                            {
                                auto a_idx = concurrency::index<2>(global_j,tile_origin+i);
                                a_local = guarded_read<true>(a_mat,a_idx);
                            }
                            else
                            {
                                auto a_idx = concurrency::index<2>(tile_origin+j,tile_j_origin+i);
                                a_local = guarded_read<true>(a_mat,a_idx);
                            }
                        }
                        auto b_idx = concurrency::index<2>(tile_origin+j,global_i);
                        b[i][j] = guarded_read<true>(b_mat,b_idx);

                        idx_b.barrier.wait_with_tile_static_memory_fence();

                        int end = _detail::min(tile_size,n-tile_origin);
                        for ( int k=0; k<end; ++k )
                            out += alpha*b[i][k]*a[k][j];

                        idx_b.barrier.wait_with_tile_static_memory_fence();
                    }

                    guarded_write<true>(c_mat, idx_b, out);
                }
            );
        }
        else // upper + notrans or lower + trans
        {
            concurrency::parallel_for_each (
                av,
                e.tile<tile_size,tile_size>(),
                [=] (concurrency::tiled_index<tile_size,tile_size> idx_b) restrict(amp)
                {
                    tile_static value_type a[tile_size][tile_size]; // "a" tile
                    tile_static value_type b[tile_size][tile_size]; // "b" tile

                    const int i = idx_b.local[1];
                    const int j = idx_b.local[0];
                    const int tile_j = idx_b.tile[0];
                    const int tile_j_origin = idx_b.tile_origin[0];
                    const int global_i = idx_b.global[1];
                    const int global_j = idx_b.global[0];

                    value_type& a_local = (transa == AmpblasNoTrans ? a[i][j] : a[j][i]);

                    value_type out = value_type();

                    auto tile_origin = 0;
                    for ( auto tile=0; tile<=tile_j; ++tile, tile_origin+=tile_size )
                    {
                        if ( tile == tile_j )
                        {
                            // diagonal tile, treat specially
                            auto a_idx = concurrency::index<2>(tile_origin+j,tile_origin+i);
                            if ( ( transa == AmpblasNoTrans && i <= j ) || ( transa != AmpblasNoTrans && i >= j ) )
                                a_local = ( diag == AmpblasUnit && i==j ) ? value_type(1) : _detail::guarded_read<true>(a_mat,a_idx);
                            else
                                a_local = value_type();
                        }
                        else
                        {
                            // off diagonal, load
                            if (transa == AmpblasNoTrans)
                            {
                                auto a_idx = concurrency::index<2>(global_j,tile_origin+i);
                                a_local = _detail::guarded_read<true>(a_mat,a_idx);
                            }
                            else
                            {
                                auto a_idx = concurrency::index<2>(tile_origin+j,tile_j_origin+i);
                                a_local = _detail::guarded_read<true>(a_mat,a_idx);
                            }
                        }
                        auto b_idx = concurrency::index<2>(tile_origin+j,global_i);
                        b[i][j] = _detail::guarded_read<true>(b_mat,b_idx);

                        idx_b.barrier.wait_with_tile_static_memory_fence();

                        int end = _detail::min(tile_size,n-tile_origin);
                        for ( int k=0; k<end; ++k )
                            out += alpha*b[i][k]*a[k][j];

                        idx_b.barrier.wait_with_tile_static_memory_fence();
                    }

                    guarded_write<true>(c_mat, idx_b, out);
                }
            );
        }
    }
}

} // namespace _detail

template <typename value_type>
void trmm(const concurrency::accelerator_view& av, enum AMPBLAS_SIDE side, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, int m, int n, value_type alpha, const concurrency::array_view<const value_type,2>& a, const concurrency::array_view<const value_type,2>& b, const concurrency::array_view<value_type,2>& c)
{
    // tuning parameters
    const int tile_size = 16;

    // call implementation
    _detail::trmm<tile_size>(av, side, uplo, transa, diag, m, n, alpha, a, b, c);
}

template <typename value_type>
void trmm(enum AMPBLAS_ORDER order, enum AMPBLAS_SIDE side, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, int m, int n, value_type alpha, const value_type* a, int lda, value_type* b, int ldb)
{
    // recursive order adjustment
    if (order == AmpblasRowMajor) 
    {
        trmm(AmpblasColMajor, side == AmpblasLeft ? AmpblasRight : AmpblasLeft, uplo == AmpblasUpper ? AmpblasLower : AmpblasUpper, transa, diag, m, n, alpha, a, lda, b, ldb);
        return;
    }

    // quick return
    if (m == 0 && n == 0) 
        return;

    // derived parameters
    int nrowa = (side == AmpblasLeft ? m : n);

    // argument check
    if (m < 0)
        argument_error("trmm", 6);
    if (n < 0)
        argument_error("trmm", 7);
    if (a == nullptr)
        argument_error("trmm", 9);
    if (lda < nrowa)
        argument_error("trmm", 10);
    if (b == nullptr)
        argument_error("trmm", 11);
    if (ldb < m)
        argument_error("trmm", 12);

    // create views
    auto a_mat = make_matrix_view(nrowa, nrowa, a, lda);
    auto b_mat = make_matrix_view(m, n, b, ldb);
    auto b_mat_const = make_matrix_view(m, n, const_cast<const value_type*>(b), ldb);

    // fill with zeros if alpha is zero
    if (alpha == value_type())
    {
        _detail::fill(get_current_accelerator_view(), b_mat.extent, value_type(), b_mat);
        return;
    }

    // workspace
    concurrency::array<value_type,2> c(n,m); 
    concurrency::array_view<value_type,2> c_mat(c);
    c_mat.discard_data();

    // forward to tuning routine
    trmm(get_current_accelerator_view(), side, uplo, transa, diag, m, n, alpha, a_mat, b_mat_const, c_mat);

    // copy workspace to answer
    copy(c_mat, b_mat);
}

} // namespace ampblas
