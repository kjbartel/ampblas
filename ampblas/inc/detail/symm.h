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
 * symm.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {
namespace _detail {

template <int tile_size, typename trans_op, typename alpha_type, typename a_value_type, typename b_value_type, typename beta_type, typename c_value_type>
void symm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, int m, int n, alpha_type alpha, const concurrency::array_view<const a_value_type,2>& a_mat, const concurrency::array_view<const b_value_type,2>& b_mat, beta_type beta, const concurrency::array_view<c_value_type,2>& c_mat )
{
    typedef a_value_type value_type;

    // pad() has undesirable functionality - pads even when unnecessary
    // auto e = c_mat.extent.tile<16,16>().pad();

    int tiles_m = (m+tile_size-1)/tile_size;
    int tiles_n = (n+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size*tiles_m,tile_size*tiles_n);

    if ( side == side::left )
        concurrency::parallel_for_each (
            av,
            e.tile<tile_size,tile_size>(),
            [=] (concurrency::tiled_index<tile_size,tile_size> idx_c) restrict(amp)
            {
                tile_static value_type at[tile_size+1][tile_size]; // "a" tile
                tile_static value_type bt[tile_size][tile_size]; // "b" tile

                auto i = idx_c.local[1];
                auto j = idx_c.local[0];
                auto tile_i = idx_c.tile[1];
                auto tile_i_origin = idx_c.tile_origin[1];
                auto global_i = idx_c.global[1];
                auto global_j = idx_c.global[0];

                int tile_origin = 0;
                value_type out=value_type(0);
                for ( int tile=0; tile < tiles_m; ++tile, tile_origin += tile_size )
                {
                    // depending on A's symmetry (uplo), need to adjust load coordinates
                    concurrency::index<2> a_idx;
                    if ( tile_i == tile )
                    {
                        // diagonal tile, need to load half and fill missing symmetry
                        if ( (uplo == uplo::upper && i <= j) || (uplo == uplo::lower && i >= j) )
                        {
                            a_idx = concurrency::index<2>(tile_origin+j,tile_origin+i);
                            // data is present - read it
                            auto v = a_mat[a_idx];
                            at[i][j] = v;

                            // fill in the missing other triangle as well
                            if ( i != j ) at[j][i] = v;
                        }
                    }
                    else if ( (uplo == uplo::upper && tile_i < tile) || (uplo == uplo::lower && tile_i > tile) )
                    {
                        // simple case, tile is fully present - read it
                        a_idx = concurrency::index<2>(tile_origin+j,global_i);
                        at[i][j] = a_mat[a_idx];
                    }
                    else
                    {
                        // need to grab the transpose tile - and transpose it as we read
                        a_idx = concurrency::index<2>(tile_i_origin+j,tile_origin+i);
                        at[j][i] = trans_op::op(a_mat[a_idx]);
                    }

                    auto b_idx = concurrency::index<2>(global_j,tile_origin+i);
                    bt[i][j] = b_mat[b_idx];

                    idx_c.barrier.wait();

                    // multiply matrices
                    int end = _detail::min(tile_size,m-tile_origin);
                    for ( auto kk=0; kk<end; ++kk )
                    {
                        out += alpha*at[i][kk]*bt[kk][j];
                    }

                    idx_c.barrier.wait();

                }
                if ( global_i<m && global_j<n )
                {
                    if ( beta != value_type() )
                        out += beta*c_mat[idx_c];
                    c_mat[idx_c] = out;
                }
            }
        );
    else
        concurrency::parallel_for_each (
            av,
            e.tile<tile_size,tile_size>(),
            [=] (concurrency::tiled_index<tile_size,tile_size> idx_c) restrict(amp)
            {
                tile_static value_type at[tile_size+1][tile_size]; // "a" tile
                tile_static value_type bt[tile_size][tile_size]; // "b" tile

                auto i = idx_c.local[1];
                auto j = idx_c.local[0];
                auto tile_j = idx_c.tile[0];
                auto tile_j_origin = idx_c.tile_origin[0];
                auto global_i = idx_c.global[1];
                auto global_j = idx_c.global[0];

                int tile_origin = 0;
                value_type out=value_type(0);
                for ( int tile=0; tile < tiles_n; ++tile, tile_origin += tile_size )
                {
                    // depending on A's symmetry (uplo), need to adjust load coordinates
                    concurrency::index<2> a_idx;
                    if ( tile_j == tile )
                    {
                        // diagonal tile, need to load half and fill missing symmetry
                        if ( (uplo == uplo::upper && i <= j) || (uplo == uplo::lower && i >= j) )
                        {
                            a_idx = concurrency::index<2>(tile_origin+j,tile_origin+i);
                            // data is present - read it
                            auto v = a_mat[a_idx];
                            at[i][j] = v;

                            // fill in the missing other triangle as well
                            if ( i != j ) at[j][i] = v;
                        }
                    }
                    else if ( (uplo == uplo::upper && tile_j > tile) || (uplo == uplo::lower && tile_j < tile) )
                    {
                        // simple case, tile is fully present - read it
                        a_idx = concurrency::index<2>(global_j,tile_origin+i);
                        at[i][j] = a_mat[a_idx];
                    }
                    else
                    {
                        // need to grab the transpose tile - and transpose it as we read
                        a_idx = concurrency::index<2>(tile_origin+j,tile_j_origin+i);
                        at[j][i] = trans_op::op(a_mat[a_idx]);
                    }

                    auto b_idx = concurrency::index<2>(tile_origin+j,global_i);
                    bt[i][j] = b_mat[b_idx];

                    idx_c.barrier.wait();

                    // multiply matrices
                    int end = _detail::min(tile_size,n-tile_origin);
                    for ( auto kk=0; kk<end; ++kk )
                    {
                        out += alpha*bt[i][kk]*at[kk][j];
                    }

                    idx_c.barrier.wait();

                }
                if ( global_i<m && global_j<n )
                {
                    if ( beta != value_type() )
                        out += beta*c_mat[idx_c];
                    c_mat[idx_c] = out;
                }
            }
        );
}

} // namespace _detail

template <typename trans_op, typename alpha_type, typename a_value_type, typename b_value_type, typename beta_type, typename c_value_type>
void symm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, int m, int n, alpha_type alpha, const concurrency::array_view<const a_value_type,2>& a_mat, const concurrency::array_view<const b_value_type,2>& b_mat, beta_type beta, const concurrency::array_view<c_value_type,2>& c_mat )
{
    // tuning parameters
    const int tile_size = 16;

    // main routine
    _detail::symm<tile_size,trans_op>(av, side, uplo, m, n, alpha, a_mat, b_mat, beta, c_mat);
}

} // namespace ampblas
