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
 * syrk.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {
namespace _detail {

template <typename trans_op, int tile_size, typename alpha_type, typename a_value_type, typename beta_type, typename c_value_type>
void syrk(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, alpha_type alpha, const concurrency::array_view<const a_value_type,2>& a_mat, beta_type beta, const concurrency::array_view<c_value_type,2>& c_mat )
{
    typedef a_value_type value_type;

    // pad() has undesirable functionality - pads even when unnecessary
    // auto e = c_mat.extent.tile<16,16>().pad();

    const int n = c_mat.extent[0];
    const int k = (trans == transpose::no_trans ? a_mat.extent[0] : a_mat.extent[1]);
    const int tiles = (n+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size*tiles,tile_size*tiles);

    concurrency::parallel_for_each(
        av,
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
            if ( (uplo==uplo::upper && tile_j < tile_i) || (uplo==uplo::lower && tile_i < tile_j) ) 
                return;

            bool notrans = trans == transpose::no_trans;
            value_type& at_local = notrans ? at[i][j] : at[j][i];
            value_type& att_local = notrans ? att[i][j] : att[j][i];

            value_type out = value_type(0);
            for ( auto ii=0; ii < k; ii += tile_size )
            {
                auto a_idx = notrans ? concurrency::index<2>(i+ii, tile_i_origin+j) : concurrency::index<2>(i+tile_i_origin, ii+j);
                auto at_idx = notrans ? concurrency::index<2>(i+ii, tile_j_origin+j) : concurrency::index<2>(i+tile_j_origin, ii+j);
                auto v = _detail::guarded_read<true>(a_mat,a_idx);

                at_local = v;
                att_local = tile_i==tile_j ? v : (_detail::guarded_read<true>(a_mat,at_idx));

                // apply transpose operation
                if (trans == transpose::no_trans)
                    att_local = trans_op::op(att_local);
                else
                    at_local = trans_op::op(at_local);

                idx_c.barrier.wait_with_tile_static_memory_fence();

                // multiply matrices
                int end = _detail::min(tile_size,k-ii);
                for ( auto kk=0; kk<end; ++kk )
                    out += alpha*at[kk][i]*att[kk][j];

                idx_c.barrier.wait_with_tile_static_memory_fence();
            }
            if ( (uplo==uplo::upper && global_j >= global_i) || (uplo==uplo::lower && global_i >= global_j) && global_i<n && global_j<n )
            {
                auto c_val = c_mat[idx_c];

                if ( global_i == global_j )
                {
                    _detail::only_real(c_val);
                }

                if ( beta != beta_type() )
                    out += beta*c_val;

                c_mat[idx_c] = out;
            }
        }
    );
}

} // namespace _detail

template <typename trans_op, typename alpha_type, typename a_value_type, typename beta_type, typename c_value_type>
void syrk(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, alpha_type alpha, const concurrency::array_view<const a_value_type,2>& a_mat, beta_type beta, const concurrency::array_view<c_value_type,2>& c_mat)
{
    // tuning parameters
    const int tile_size = 16;

    // main routine
    _detail::syrk<trans_op, tile_size>(av, uplo, trans, alpha, a_mat, beta, c_mat);
}

template <typename alpha_type, typename a_value_type, typename beta_type, typename c_value_type>
void syrk(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, alpha_type alpha, const concurrency::array_view<const a_value_type,2>& a_mat, beta_type beta, const concurrency::array_view<c_value_type,2>& c_mat)
{
    syrk<_detail::noop>(av, uplo, trans, alpha, a_mat, beta, c_mat);
}

template <typename alpha_type, typename a_value_type, typename beta_type, typename c_value_type>
void herk(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, int n, int k, alpha_type alpha, const concurrency::array_view<const a_value_type,2>& a_mat, beta_type beta, const concurrency::array_view<c_value_type,2>& c_mat)
{
    syrk<_detail::conjugate>(av, uplo, trans, alpha, a_mat, beta, c_mat);
}

template <typename scalar_type, typename value_type>
void syrk(enum AMPBLAS_ORDER order, enum class uplo uplo, enum class transpose trans, int n, int k, scalar_type alpha, const value_type* a, int lda, scalar_type beta, value_type* c, int ldc)
{
    syrk<_detail::noop>(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <typename scalar_type, typename value_type>
void herk(enum AMPBLAS_ORDER order, enum class uplo uplo, enum class transpose trans, int n, int k, scalar_type alpha, const value_type* a, int lda, scalar_type beta, value_type* c, int ldc)
{
    syrk<_detail::conjugate>(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

} // namespace ampblas
