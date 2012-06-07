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
 * gemm.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {
namespace _detail {

// Generic GEMM algorithm on AMP array_views of type value_type
template <int tile_size, bool guarded, enum class transpose transa, enum class transpose transb, typename value_type>
void gemm(const concurrency::accelerator_view& av, value_type alpha, const concurrency::array_view<const value_type,2>& a, const concurrency::array_view<const value_type,2>& b, value_type beta, const concurrency::array_view<value_type,2>& c)
{
    int k_max = (transa == transpose::no_trans ? a.extent[0] : a.extent[1]);

    // configuration
    const int n = c.extent[0];
    const int m = c.extent[1];
    const int tiles_m = (m+tile_size-1)/tile_size;
    const int tiles_n = (n+tile_size-1)/tile_size;
    auto e = make_extent(tile_size*tiles_m, tile_size*tiles_n);

    concurrency::parallel_for_each(
        av,
        e.tile<tile_size,tile_size>(), 
        [=] (concurrency::tiled_index<tile_size,tile_size> tid) restrict(amp)
	{
        // shared memory buffers
        tile_static value_type a_tile[tile_size][tile_size];
        tile_static value_type b_tile[tile_size][tile_size];

        // constant offsets
        const int col = tid.local[0];
        const int row = tid.local[1];

        // common memory locations
        value_type& a_local = (transa == transpose::no_trans ? a_tile[row][col] : a_tile[col][row]);
        value_type& b_local = (transb == transpose::no_trans ? b_tile[row][col] : b_tile[col][row]);

        // tile location
        const int j = tid.tile_origin[0];
        const int i = tid.tile_origin[1];

        value_type sum = value_type(); 

        // k loop
        for (int k=0; k<k_max; k+=tile_size)
        {
            auto a_idx = (transa == transpose::no_trans ? concurrency::index<2>(k+row, i+col) : concurrency::index<2>(i+row, k+col));
            auto b_idx = (transb == transpose::no_trans ? concurrency::index<2>(j+row, k+col) : concurrency::index<2>(k+row, j+col));
            
            // load a & b
            a_local = guarded_read<guarded>(a, a_idx);
            b_local = guarded_read<guarded>(b, b_idx);

            // apply transpose operations
            if (transa == transpose::conj_trans)
                a_local = conjugate::op(a_local);
            if (transb == transpose::conj_trans)
                b_local = conjugate::op(b_local);

            // wait for reads
            tid.barrier.wait_with_tile_static_memory_fence();

            // intrablock accumulation
            for (int l=0; l<tile_size; l++)
                sum += a_tile[l][col] * b_tile[row][l];
            
            // wait for access 
            tid.barrier.wait_with_tile_static_memory_fence();
        }

        // write results
        const concurrency::index<2> c_idx(j+row, i+col);

        // apply alpha & beta
        value_type c_val = guarded_read<guarded>(c, c_idx);
        c_val = alpha*sum + beta*c_val;

        // final write
        guarded_write<guarded>(c, c_idx, c_val);
	});

    // testing...
    const_cast<concurrency::accelerator_view&>(av).wait();
}

} // namespace _detail

// Generic GEMM algorithm on AMP array_views of type value_type
template <typename value_type>
void gemm(const concurrency::accelerator_view& av, enum class transpose transa, enum class transpose transb, value_type alpha, const concurrency::array_view<const value_type,2>& a, const concurrency::array_view<const value_type,2>& b, value_type beta, const concurrency::array_view<value_type,2>& c)
{
    // tuning parameters
    const int tile_size = 16;
    const bool guarded = true;

    // main routine
    if (transa == transpose::no_trans)
    {
        if (transb == transpose::no_trans)
        {
            // GEMM_NN
            _detail::gemm<tile_size, guarded, transpose::no_trans, transpose::no_trans>(av, alpha, a, b, beta, c);
        }
        else if (transb == transpose::trans)
        {
            // GEMM_NT
            _detail::gemm<tile_size, guarded, transpose::no_trans, transpose::trans>(av, alpha, a, b, beta, c);
        }
        else if (transb == transpose::conj_trans)
        {
            // GEMM_NC
            _detail::gemm<tile_size, guarded, transpose::no_trans, transpose::conj_trans>(av, alpha, a, b, beta, c);
        }
    }
    else if (transa == transpose::trans)
    {
        if (transb == transpose::no_trans)
        {
            // GEMM_TN
            _detail::gemm<tile_size, guarded, transpose::trans, transpose::no_trans>(av, alpha, a, b, beta, c);
        }
        else if (transb == transpose::trans)
        {
            // GEMM_TT
            _detail::gemm<tile_size, guarded, transpose::trans, transpose::trans>(av, alpha, a, b, beta, c);
        }
        else if (transb == transpose::conj_trans)
        {
            // GEMM_TC
            _detail::gemm<tile_size, guarded, transpose::trans, transpose::conj_trans>(av, alpha, a, b, beta, c);
        }
    }
    else if (transa == transpose::conj_trans)
    {
    if (transb == transpose::no_trans)
        {
            // GEMM_CN
            _detail::gemm<tile_size, guarded, transpose::conj_trans, transpose::no_trans>(av, alpha, a, b, beta, c);
        }
        else if (transb == transpose::trans)
        {
            // GEMM_CT
            _detail::gemm<tile_size, guarded, transpose::conj_trans, transpose::trans>(av, alpha, a, b, beta, c);
        }
        else if (transb == transpose::conj_trans)
        {
            // GEMM_CC
            _detail::gemm<tile_size, guarded, transpose::conj_trans, transpose::conj_trans>(av, alpha, a, b, beta, c);
        }
    }
}

} // namespace ampblas
