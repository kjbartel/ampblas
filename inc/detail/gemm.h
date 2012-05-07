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
template <int tile_size, bool guarded, typename value_type>
void gemm(enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_TRANSPOSE transb, value_type alpha, const concurrency::array_view<const value_type,2>& a, const concurrency::array_view<const value_type,2>& b, value_type beta, const concurrency::array_view<value_type,2>& c)
{
    int k_max = (transa == AmpblasNoTrans ? a.extent[0] : a.extent[1]);

    // configuration
    const int n = c.extent[0];
    const int m = c.extent[1];
    const int tiles_m = (m+tile_size-1)/tile_size;
    const int tiles_n = (n+tile_size-1)/tile_size;
    auto e = make_extent(tile_size*tiles_m, tile_size*tiles_n);

    concurrency::parallel_for_each(get_current_accelerator_view(), e.tile<tile_size,tile_size>(), [=] (concurrency::tiled_index<tile_size,tile_size> tid) restrict(amp)
	{
        // shared memory buffers
        tile_static value_type a_tile[tile_size][tile_size];
        tile_static value_type b_tile[tile_size][tile_size];

        // constant offsets
        const int col = tid.local[0];
        const int row = tid.local[1];

        // common memory locations
        value_type& a_local = (transa == AmpblasNoTrans ? a_tile[row][col] : a_tile[col][row]);
        value_type& b_local = (transb == AmpblasNoTrans ? b_tile[row][col] : b_tile[col][row]);

        // tile location
        const int j = tid.tile_origin[0];
        const int i = tid.tile_origin[1];

        value_type sum = value_type(); 
        
        // k loop
        for (int k=0; k<k_max; k+=tile_size)
        {
            auto a_idx = (transa == AmpblasNoTrans ? concurrency::index<2>(k+row, i+col) : concurrency::index<2>(i+row, k+col));
            auto b_idx = (transb == AmpblasNoTrans ? concurrency::index<2>(j+row, k+col) : concurrency::index<2>(k+row, j+col));
            
            // load a & b
            a_local = _detail::guarded_read<guarded>(a, a_idx);
            b_local = _detail::guarded_read<guarded>(b, b_idx);

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
        value_type c_val = _detail::guarded_read<guarded>(c, c_idx);
        c_val = alpha*sum + beta*c_val;

        // final write
        _detail::guarded_write<true>(c, c_idx, c_val);

	});
}

} // namespace _detail

// Generic GEMM algorithm on AMP array_views of type value_type
template <typename value_type>
void gemm(enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_TRANSPOSE transb, value_type alpha, const concurrency::array_view<const value_type,2>& a, const concurrency::array_view<const value_type,2>& b, value_type beta, const concurrency::array_view<value_type,2>& c)
{
    // tuning parameters
    const int tile_size = 16;
    const bool guarded = true;

    // main routine
    _detail::gemm<tile_size,guarded>(transa, transb, alpha, a, b, beta, c);
}

// Generic GEMM algorithm
template <typename value_type>
void gemm(enum AMPBLAS_ORDER order, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_TRANSPOSE transb, int m, int n, int k, value_type alpha, const value_type *a, int lda, const value_type *b, int ldb, value_type beta, value_type *c, int ldc) 
{
	// recursive order adjustment 
	if (order == AmpblasRowMajor)
    {
        gemm(AmpblasColMajor, transb, transa, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
        return;
    }

    // quick return
    if ((m == 0 || n == 0 || alpha == value_type() || k == 0) && beta == value_type(1))
        return;

    // derived parameters
    auto a_row = (transa == AmpblasNoTrans ? m : k);
	auto a_col = (transa == AmpblasNoTrans ? k : m);   
	auto b_row = (transb == AmpblasNoTrans ? k : n);
	auto b_col = (transb == AmpblasNoTrans ? n : k);

	// error check
	if (m < 0)		       
		argument_error("gemm", 4);
	if (n < 0)        
		argument_error("gemm", 5);
	if (k < 0)        
		argument_error("gemm", 6);
	if (a == nullptr) 
		argument_error("gemm", 8);
	if (lda < a_row)
		argument_error("gemm", 9);
	if (b == nullptr) 
		argument_error("gemm", 10);
	if (ldb < b_row) 
		argument_error("gemm", 11);
	if (c == nullptr) 
		argument_error("gemm", 13);
	if (ldc < m) 
		argument_error("gemm", 14);
  
    // create views
	auto a_mat = make_matrix_view(a_row, a_col, a, lda);
	auto b_mat = make_matrix_view(b_row, b_col, b, ldb);
	auto c_mat = make_matrix_view(m, n, c, ldc);

    // special cases
	if (alpha == value_type())
	{
		if (beta == value_type())
			_detail::fill(c_mat.extent, value_type(), c_mat);
		else
			_detail::scale(c_mat.extent, beta, c_mat);

		return;
	}

    // forward to tuning routine
    gemm(transa, transb, alpha, a_mat, b_mat, beta, c_mat);
}

} // namespace ampblas
