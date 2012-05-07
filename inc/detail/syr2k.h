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
 * syr2k.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {
namespace _detail {

template <int tile_size, typename alpha_type, typename a_value_type, typename b_value_type, typename beta_type, typename c_value_type>
void syr2k(enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE trans, int n, int k, alpha_type alpha, const concurrency::array_view<const a_value_type,2>& a_mat, const concurrency::array_view<const b_value_type,2>& b_mat, beta_type beta, const concurrency::array_view<c_value_type,2>& c_mat)
{
    typedef a_value_type value_type;

    // pad() has undesirable functionality - pads even when unnecessary
    // auto e = c_mat.extent.tile<16,16>().pad();

    int tiles = (n+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size*tiles,tile_size*tiles);

    concurrency::parallel_for_each (
        get_current_accelerator_view(),
        e.tile<tile_size,tile_size>(),
        [=] (concurrency::tiled_index<tile_size,tile_size> idx_c) restrict(amp)
        {
            tile_static value_type at[tile_size+1][tile_size]; // "a" tile
            tile_static value_type att[tile_size+1][tile_size]; // "a" transpose tile

            auto i = idx_c.local[1];
            auto j = idx_c.local[0];
            auto tile_i = idx_c.tile[1];
            auto tile_j = idx_c.tile[0];
            auto tile_i_origin = idx_c.tile_origin[1];
            auto tile_j_origin = idx_c.tile_origin[0];
            auto global_i = idx_c.global[1];
            auto global_j = idx_c.global[0];

            // quick return path for unnecessary tiles
            // skips too early for operations with just a beta component, but those shouldn't be handled by this routine
            if ( (uplo==AmpblasUpper && tile_j < tile_i) || (uplo==AmpblasLower && tile_i < tile_j) ) return;

            bool notrans = trans == AmpblasNoTrans;
            value_type& at_local = notrans ? at[i][j] : at[j][i];
            value_type& att_local = notrans ? att[i][j] : att[j][i];

            value_type out=value_type(0);
            for ( auto ii=0; ii < k; ii += tile_size )
            {
                auto a_idx = notrans?concurrency::index<2>(i+ii, tile_i_origin+j):concurrency::index<2>(tile_i_origin+i, ii+j);
                auto bt_idx = notrans?concurrency::index<2>(i+ii, tile_j_origin+j):concurrency::index<2>(tile_j_origin+i, ii+j);

                at_local  = _detail::guarded_read<true>(a_mat,a_idx);
                att_local = _detail::guarded_read<true>(b_mat,bt_idx);

                idx_c.barrier.wait_with_tile_static_memory_fence();

                // multiply matrices
                int end = _detail::min(tile_size,k-ii);
                if ( tile_i == tile_j ) // shortcut for diagonal tiles
                    for ( auto kk=0; kk<end; ++kk )
                    {
                        out += alpha*(at[kk][i]*att[kk][j]+at[kk][j]*att[kk][i]);
                    }
                else
                    for ( auto kk=0; kk<end; ++kk )
                    {
                        out += alpha*at[kk][i]*att[kk][j];
                    }

                idx_c.barrier.wait_with_tile_static_memory_fence();
                if ( tile_i == tile_j ) continue; // diagonal tiles skip some memory time

                // swap matrices, repeat
                at_local  = _detail::guarded_read<true>(b_mat,a_idx);
                att_local = _detail::guarded_read<true>(a_mat,bt_idx);

                idx_c.barrier.wait_with_tile_static_memory_fence();

                // multiply matrices
                for ( auto kk=0; kk<end; ++kk )
                {
                    out += alpha*at[kk][i]*att[kk][j];
                }
            }
            if ( (uplo==AmpblasUpper && global_j >= global_i) || (uplo==AmpblasLower && global_i >= global_j) && global_i<n && global_j<n )
            {
                if ( beta != value_type() )
                    out += beta*c_mat[idx_c];
                c_mat[idx_c] = out;
            }
        }
    );
}

} // namespace _detail

template <typename alpha_type, typename a_value_type, typename b_value_type, typename beta_type, typename c_value_type>
void syr2k(enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE trans, int n, int k, alpha_type alpha, const concurrency::array_view<const a_value_type,2>& a_mat, const concurrency::array_view<const b_value_type,2>& b_mat, beta_type beta, const concurrency::array_view<c_value_type,2>& c_mat )
{
    // tuning parameters
    const int tile_size = 16;

    // call main routine
    _detail::syr2k<tile_size>(uplo, trans, n, k, alpha, a_mat, b_mat, beta, c_mat);
}

template <typename value_type>
void syr2k(enum AMPBLAS_ORDER order, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE trans, int n, int k, value_type alpha, const value_type* a, int lda, const value_type* b, int ldb, value_type beta, value_type* c, int ldc)
{
    // recursive order adjustment
    if (order == AmpblasRowMajor) 
    {
        syr2k(AmpblasColMajor, uplo == AmpblasLower ? AmpblasUpper : AmpblasLower, trans == AmpblasNoTrans ? AmpblasTrans : trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }

    // quick return
    if (n == 0 || ((alpha == value_type() || k == 0) && beta == value_type(1)))
        return;

    // derived parameters
    int nrowa = (trans == AmpblasNoTrans ? n : k);
    int ka = (trans == AmpblasNoTrans ? k : n);

    // argument check
    if (n < 0)
        argument_error("syr2k", 4);
    if (k < 0)
        argument_error("syr2k", 5);
    if (a == nullptr)
        argument_error("syr2k", 7);
    if (lda < nrowa)
        argument_error("syr2k", 8);
    if (a == nullptr)
        argument_error("syr2k", 9);
    if (ldb < nrowa)
        argument_error("syr2k", 10);
    if (c == nullptr)
        argument_error("syr2k", 12);
    if (ldc < n)
        argument_error("syr2k", 13);

    // create views
    auto a_mat = make_matrix_view(nrowa, ka, a, lda);
    auto b_mat = make_matrix_view(nrowa, ka, b, ldb);
    auto c_mat = make_matrix_view(n, n, c, ldc);

    // use triangular scale or fill if alpha is zero and beta is not 1
    if ( alpha == value_type() )
    {
        if ( beta == value_type() )
            _detail::fill_triangle(uplo,value_type(),c_mat);
        else
            _detail::scale(uplo,beta,c_mat);
        return;
    }

    syr2k(uplo, trans, n, k, alpha, a_mat, b_mat, beta, c_mat);
}


} // namespace ampblas
