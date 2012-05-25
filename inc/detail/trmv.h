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
 * trmv.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {
namespace _detail {

//-------------------------------------------------------------------------
// TRMV
//  This routine has limited parallelism and should only be used as a
//  building block to enable larger routines.
//-------------------------------------------------------------------------

template <typename value_type, typename x_vector_type, typename y_vector_type> 
void trmv_l(const concurrency::accelerator_view& av, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, const concurrency::array_view<const value_type,2>& a, x_vector_type& x, y_vector_type& y)
{
    concurrency::parallel_for_each(av, y.extent, [=] (concurrency::index<1> y_idx) restrict(amp)
    {
        value_type result = value_type();

        for (int i=0; i<=y_idx[0]; i++)
        {
            concurrency::index<2> a_idx = (transa == AmpblasNoTrans ? concurrency::index<2>(i, y_idx[0]) : concurrency::index<2>(y_idx[0], i));
			concurrency::index<1> x_idx(i);

            result += (diag == AmpblasUnit && _detail::is_diag(a_idx) ? x[x_idx] : a[a_idx] * x[x_idx]);
        }

        y[y_idx] = result;
    });
}

template <typename value_type, typename x_vector_type, typename y_vector_type> 
void trmv_u(const concurrency::accelerator_view& av, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, const concurrency::array_view<const value_type,2>& a, x_vector_type& x, y_vector_type& y)
{
    int n = y.extent[0];

    concurrency::parallel_for_each(av, y.extent, [=] (concurrency::index<1> y_idx) restrict(amp)
    {
        value_type result = value_type();
       
        for (int i=y_idx[0]; i<n; i++)
        {
            concurrency::index<2> a_idx = (transa == AmpblasNoTrans ? concurrency::index<2>(i, y_idx[0]) : concurrency::index<2>(y_idx[0], i));
			concurrency::index<1> x_idx(i);

            result += (diag == AmpblasUnit && _detail::is_diag(a_idx) ? x[x_idx] : a[a_idx] * x[x_idx]);
        }

        y[y_idx] = result;
    });
}

} // namespace _detail

template <typename value_type, typename x_vector_type, typename y_vector_type> 
void trmv(const concurrency::accelerator_view& av, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, const concurrency::array_view<const value_type,2>& a, x_vector_type& x, y_vector_type& y)
{
    if ((uplo == AmpblasLower) ^ (transa != AmpblasNoTrans))
	    _detail::trmv_l(av, transa, diag, a, x, y);
    else
        _detail::trmv_u(av, transa, diag, a, x, y);
}

template <typename value_type>
void trmv(enum AMPBLAS_ORDER order, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, int n, const value_type *a, int lda, value_type *x, int incx)
{
    // recursive order adjustment
	if (order == AmpblasRowMajor)
    {
        trmv(AmpblasColMajor, uplo == AmpblasUpper ? AmpblasLower : AmpblasUpper, transa == AmpblasNoTrans ? AmpblasTrans : transa, diag, n, a, lda, x, incx);
        return;
    }

	// quick return
	if (n == 0)
		return;

	// argument check
	if (n < 0)
        argument_error("trmv", 5);
    if (a == nullptr)
        argument_error("trmv", 6);
    if (lda < n)
        argument_error("trmv", 7);
    if (x == nullptr)
        argument_error("trmv", 7);

    // workspace to enable some more parallelism
    concurrency::array<value_type,1> workspace(n);
    
	// create views
	auto x_vec = make_vector_view(n, x, incx);
    auto a_mat = make_matrix_view(n, n, a, lda);
    concurrency::array_view<value_type,1> y_vec(workspace);

    // call generic implementation
	trmv(get_current_accelerator_view(), uplo, transa, diag, a_mat, x_vec, y_vec);
   
    // copy workspace back to x
    auto x_2d = x_vec.get_base_view().view_as(concurrency::extent<2>(n,std::abs(incx))).section(concurrency::extent<2>(n,1));
    auto y_2d = y_vec.view_as(concurrency::extent<2>(n,1));
    y_2d.copy_to(x_2d);
    x_vec.get_base_view().refresh();
}

} // namespace ampblas
