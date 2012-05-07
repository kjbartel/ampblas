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
 * trsm.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {
namespace _detail {

template <int tile_size, bool guarded, typename value_type>
void trsm_ll(enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, value_type alpha, const concurrency::array_view<const value_type,2>& a, const concurrency::array_view<value_type,2>& b) 
{
    // runtime sizes
    int m = b.extent[1];
    int n = b.extent[0];
    int tiles = (n+tile_size-1)/tile_size;

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
        // transpose read pattern still allows for coalesced global memory access
        value_type& a_local = (transa == AmpblasNoTrans ? a_tile[row][col] : a_tile[col][row]);
        value_type& b_local = b_tile[row][col];

        // global j index
        const int j = tid.tile_origin[0];

        // loop down by tiles
        for (int i=0; i<m; i+=tile_size)
        {
            // read tile at A(i,i) into local A
            a_local = _detail::guarded_read<guarded>(a, concurrency::index<2>(i+row, i+col));

            // read tile at B(i,j) into local B
            b_local = _detail::guarded_read<guarded>(b, concurrency::index<2>(j+row, i+col));
            tid.barrier.wait();

            // solve X(i,j) = B(i,j) \ A(i,i)
            if (col == 0)
            {
                int jj = row;

                // loop down shared block
                for (int ii=0; ii<tile_size; ii++)
                {
                    // elimation scalar
                    value_type temp = b_tile[jj][ii];
                    if (diag == AmpblasNonUnit)
                        temp /= (a_tile[ii][ii] == value_type() ? value_type(1) : a_tile[ii][ii]);

                    // apply
                    for (unsigned int kk=ii+1; kk<tile_size; kk++)
                        b_tile[jj][kk] -= temp * a_tile[ii][kk];

                    b_tile[jj][ii] = temp;
                }
            }

            // wait for local solve
            tid.barrier.wait();

            // apply B(k,j) -= B(i,j) * A(k,i) 
            for (int k=i+tile_size; k<m; k+=tile_size)
            {   
                // read tile at A(k,i) into local A
                a_local = _detail::guarded_read<guarded>(a, transa == AmpblasNoTrans ? concurrency::index<2>(i+row, k+col) : concurrency::index<2>(k+row, i+col));
                tid.barrier.wait();

                // accumulate
                value_type sum = value_type();

                // TODO: explictly unrollable?
                for (int l=0; l<tile_size; l++)
                    sum += a_tile[l][col] * b_tile[row][l];

                // update
                _detail::guarded_update<guarded>(b, concurrency::index<2>(j+row, k+col), _detail::subtract<value_type>(sum));

                // wait for a to finish being read
                tid.barrier.wait();
            }

            // write B(i,j)
            _detail::guarded_write<guarded>(b, concurrency::index<2>(j+row, i+col), alpha*b_local);
        }
    });
}

template <int tile_size, bool guarded, typename value_type>
void trsm_lu(enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, value_type alpha, const concurrency::array_view<const value_type,2>& a, const concurrency::array_view<value_type,2>& b) 
{
    // runtime sizes
    int m = b.extent[1];
    int n = b.extent[0];
    int tiles = (n+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size, tile_size*tiles);

    // compiler work around
    const int dummy = 1;

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
        // transpose read pattern still allows for coalesced global memory access
        value_type& a_local = (transa == AmpblasNoTrans ? a_tile[row][col] : a_tile[col][row]);
        value_type& b_local = b_tile[row][col];

        // global j index
        const int j = tid.tile_origin[0];

        // loop up by tiles
        for (int i_ = (m-1) & (-tile_size); i_>=0; i_-=tile_size)
        {
            // compiler work around
            int i = dummy ? i_ : 0;            

            // read tile at A(i,i) into local A
            a_local = _detail::guarded_read<guarded>(a, concurrency::index<2>(i+row, i+col));

            // read tile at B(i,j) into local B
            b_local = _detail::guarded_read<guarded>(b, concurrency::index<2>(j+row, i+col));
            tid.barrier.wait_with_tile_static_memory_fence();

            // solve X(i,j) = B(i,j) \ A(i,i)
            if (col == 0)
            {
                int jj = row;

                // loop down shared block
                for (int ii=_detail::min(tile_size-1,m-1-i); ii>=0; ii--)
                {
                    // elimation scalar
                    value_type temp = b_tile[jj][ii];

                    if (diag == AmpblasNonUnit)
                        temp /= (a_tile[ii][ii] == value_type() ? value_type(1) : a_tile[ii][ii]);

                    // apply
                    for (int kk=0; kk<ii; kk++)
                        b_tile[jj][kk] -= temp * a_tile[ii][kk];

                    b_tile[jj][ii] = temp;
                }
            }

            // wait for local solve
            tid.barrier.wait_with_tile_static_memory_fence();

            // write B(i,j)
            _detail::guarded_write<guarded>(b, concurrency::index<2>(j+row, i+col), alpha*b_local);

            // apply B(k,j) -= B(i,j) * A(k,i) 
            for (int k_=0; k_<i; k_+=tile_size)
            {   
                // compiler workaround
                int k = dummy ? k_ : 0;

                // read tile at A(k,i) into local A
                a_local = _detail::guarded_read<guarded>(a, transa == AmpblasNoTrans ? concurrency::index<2>(i+row, k+col) : concurrency::index<2>(k+row, i+col));
                tid.barrier.wait_with_tile_static_memory_fence();

                // accumulate
                value_type sum = value_type();

                // TODO: explictly unrollable?
                for (int l=0; l<tile_size; l++)
                    sum += a_tile[l][col] * b_tile[row][l];

                // update
                _detail::guarded_update<guarded>(b, concurrency::index<2>(j+row, k+col), _detail::subtract<value_type>(sum));
                tid.barrier.wait_with_tile_static_memory_fence();
            }
        }
    });
}

template <int tile_size, bool guarded, typename value_type>
void trsm_rl(enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, value_type alpha, const concurrency::array_view<const value_type,2>& a, const concurrency::array_view<value_type,2>& b) 
{
    // runtime sizes
    int m = b.extent[1];
    int n = b.extent[0];
    int tiles = (m+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size, tile_size*tiles);

    // compiler work around
    const int dummy = 1;

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
        // transpose read pattern still allows for coalesced global memory access
        value_type& a_local = (transa == AmpblasNoTrans ? a_tile[row][col] : a_tile[col][row]);
        value_type& b_local = b_tile[row][col];
        
        // global i index
        const int i = tid.tile_origin[0];

        // loop right to left across tiles
        for (int j_=(n-1) & (-tile_size); j_>=0; j_-=tile_size)
        {
            // compiler work around
            int j = dummy ? j_ : 0;

            // read tile at A(j,j) into local A
            a_local = _detail::guarded_read<guarded>(a, concurrency::index<2>(j+row, j+col));

            // read tile at B(i,j) into local B
            b_local = _detail::guarded_read<guarded>(b, concurrency::index<2>(j+row, i+col));
            tid.barrier.wait_with_tile_static_memory_fence();

            // solve A(j,j) * X(i,j) = B(i,j)
            if (col == 0)
            {
                int ii = row;

                // loop down shared block
                for (int jj=_detail::min(tile_size-1,n-1-j); jj>=0; jj--)
                {
                    // elimation scalar
                    value_type temp = b_tile[jj][ii];

                    if (diag == AmpblasNonUnit)
                        temp /= (a_tile[jj][jj] == value_type() ? value_type(1) : a_tile[jj][jj]);

                    // apply
                    for (int kk=0; kk<jj; kk++)
                        b_tile[kk][ii] -= temp * a_tile[kk][jj];

                    b_tile[jj][ii] = temp;
                }
            }

            // wait for local solve
            tid.barrier.wait_with_tile_static_memory_fence();

            // write B(i,j)
            _detail::guarded_write<guarded>(b, concurrency::index<2>(j+row, i+col), alpha*b_local);
            tid.barrier.wait_with_tile_static_memory_fence();

            // apply B(i,k) -= A(j,k) * B(i,j)
            for (int k_=0; k_<j; k_+=tile_size)
            {   
                // compiler work around
                int k = dummy ? k_ : 0;

                // read tile at A(k,i) into local A
                a_local = _detail::guarded_read<guarded>(a, transa == AmpblasNoTrans ? concurrency::index<2>(k+row, j+col) : concurrency::index<2>(j+row, k+col));
                tid.barrier.wait();

                // accumulate
                value_type sum = value_type();

                // TODO: explictly unrollable?
                for (int l=0; l<tile_size; l++)
                    sum += b_tile[l][col] * a_tile[row][l];

                // update
                _detail::guarded_update<guarded>(b, concurrency::index<2>(k+row, i+col), _detail::subtract<value_type>(sum));

                // wait for a to finish being read
                tid.barrier.wait_with_tile_static_memory_fence();
            }
        }

    });
}

template <int tile_size, bool guarded, typename value_type>
void trsm_ru(enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, value_type alpha, const concurrency::array_view<const value_type,2>& a, const concurrency::array_view<value_type,2>& b) 
{
    // runtime sizes
    int m = b.extent[1];
    int n = b.extent[0];
    int tiles = (m+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size, tile_size*tiles);

    concurrency::parallel_for_each( 
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
        // transpose read pattern still allows for coalesced global memory access
        value_type& a_local = (transa == AmpblasNoTrans ? a_tile[row][col] : a_tile[col][row]);
        value_type& b_local = b_tile[row][col];

        // global i index
        const int i = tid.tile_origin[0];

        // loop right to left across tiles
        for (int j=0; j<n; j+=tile_size)
        {
            // read tile at A(j,j) into local A
            a_local = _detail::guarded_read<guarded>(a, concurrency::index<2>(j+row, j+col));

            // read tile at B(i,j) into local B
            b_local = _detail::guarded_read<guarded>(b, concurrency::index<2>(j+row, i+col));
            tid.barrier.wait_with_tile_static_memory_fence();

            // solve A(j,j) * X(i,j) = B(i,j)
            if (col == 0)
            {
                int ii = row;

                // loop down shared block
                for (int jj=0; jj<_detail::min(tile_size,n-j);jj++)
                {
                    // elimation scalar
                    value_type temp = b_tile[jj][ii];

                    if (diag == AmpblasNonUnit)
                        temp /= (a_tile[jj][jj] == value_type() ? value_type(1) : a_tile[jj][jj]);

                    // apply
                    for (int kk=jj; kk<tile_size; kk++)
                        b_tile[kk][ii] -= temp * a_tile[kk][jj];

                    b_tile[jj][ii] = temp;
                }
            }

            // wait for local solve
            tid.barrier.wait_with_tile_static_memory_fence();

            // write B(i,j)
            _detail::guarded_write<guarded>(b, concurrency::index<2>(j+row, i+col), alpha*b_local);
            tid.barrier.wait_with_tile_static_memory_fence();

            // apply B(i,k) -= A(j,k) * B(i,j) 
            for (int k=j+tile_size; k<n; k+=tile_size)
            {   
                // read tile at A(k,i) into local A
                a_local = _detail::guarded_read<guarded>(a, transa == AmpblasNoTrans ? concurrency::index<2>(k+row, j+col) : concurrency::index<2>(j+row, k+col));
                tid.barrier.wait_with_tile_static_memory_fence();

                // accumulate
                value_type sum = value_type();

                // TODO: explictly unrollable?
                for (int l=0; l<tile_size; l++)
                    sum += b_tile[l][col] * a_tile[row][l];

                // update
                _detail::guarded_update<guarded>(b, concurrency::index<2>(k+row, i+col), _detail::subtract<value_type>(sum));

                // wait for a to finish being read
                tid.barrier.wait_with_tile_static_memory_fence();
            }
        }
    });
}

} // namespace _detail

template <typename value_type>
void trsm(enum AMPBLAS_SIDE side, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, value_type alpha, const concurrency::array_view<const value_type,2>& a, const concurrency::array_view<value_type,2>& b)
{
    // tuning parameters
    const int tile_size = 16;
    const bool guarded = true;

    // select proper kernel based on options
    if (side == AmpblasLeft)
    {
        if ((uplo == AmpblasLower) ^ (transa != AmpblasNoTrans))
        {
            // lower + no trans <==> upper + trans 
            _detail::trsm_ll<tile_size,guarded>(transa, diag, alpha, a, b);
        }
        else
        {
            // upper + no trans <==> lower + trans 
            _detail::trsm_lu<tile_size,guarded>(transa, diag, alpha, a, b);
        }
    }
    else if (side == AmpblasRight)
    {
        if ((uplo == AmpblasLower) ^ (transa != AmpblasNoTrans))
        {
            // lower + no trans <==> upper + trans 
            _detail::trsm_rl<tile_size,guarded>(transa, diag, alpha, a, b);
        }
        else
        {
            // upper + no trans <==> lower + trans
            _detail::trsm_ru<tile_size,guarded>(transa, diag, alpha, a, b);
        }
    }
}

template <typename value_type>
void trsm(enum AMPBLAS_ORDER order, enum AMPBLAS_SIDE side, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, int m, int n, value_type alpha, const value_type *a, int lda, value_type *b, int ldb)
{
	// recursive order adjustment 
    if (order==AmpblasRowMajor)
    {      
        trsm(AmpblasColMajor, (side==AmpblasLeft ? AmpblasRight : AmpblasLeft), (uplo==AmpblasUpper ? AmpblasLower : AmpblasUpper), transa, diag, n, m, alpha, a, lda, b, ldb);
        return;
    }

    // quick return
	if (m==0 || n==0)
        return;

	// derived parameters
	int k = (side==AmpblasLeft ? m : n);

	// error check
	if (m < 0)
        argument_error("trsm", 6);
    if (n < 0)                 
        argument_error("trsm", 7);
    if (a == nullptr)          
        argument_error("trsm", 9);
    if (lda < k)               
        argument_error("trsm", 10);
    if (b == nullptr)          
        argument_error("trsm", 11);
    if (ldb < m)               
        argument_error("trsm", 12);
  
	// amp data
    const concurrency::array_view<const value_type,2> a_mat = make_matrix_view(k, k, a, lda);
	const concurrency::array_view<value_type,2> b_mat = make_matrix_view(m, n, b, ldb);

	// special paths
	if (alpha==value_type())
    {
        _detail::fill(make_extent(m,n), value_type(), b_mat);
        return;
    }

	// implementation
	trsm(side, uplo, transa, diag, alpha, a_mat, b_mat);
}

} // namespace ampblas
