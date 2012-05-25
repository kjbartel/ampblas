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
 * gemv.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {
namespace _detail {

//-------------------------------------------------------------------------
// GEMV
//-------------------------------------------------------------------------

template <AMPBLAS_TRANSPOSE transa, typename value_type, typename x_vector_type, typename y_vector_type>
void gemv(const concurrency::accelerator_view& av, value_type alpha, const concurrency::array_view<const value_type,2>& a, const x_vector_type& x, value_type beta, y_vector_type& y)
{
	concurrency::parallel_for_each(
        av,
        y.extent, 
        [=] (concurrency::index<1> y_idx) restrict(amp)
        {
            value_type result = value_type();
        
            for (int n = 0; n < x.extent[0]; ++n)
            {
                concurrency::index<2> a_idx = (transa == AmpblasNoTrans ? concurrency::index<2>(n, y_idx[0]) : concurrency::index<2>(y_idx[0], n));
			    concurrency::index<1> x_idx(n);

                auto a_value = a[a_idx];
                if (transa == AmpblasConjTrans)
                    a_value = conjugate::op(a_value);
            
                result += a_value * x[x_idx];
            }

            y[y_idx] = alpha * result + beta * y[y_idx];
        }
    );
}
} // namespace _detail

template <typename value_type, typename x_vector_type, typename y_vector_type>
void gemv(const concurrency::accelerator_view& av, enum AMPBLAS_TRANSPOSE transa, value_type alpha, const concurrency::array_view<const value_type,2>& a, x_vector_type& x, value_type beta, y_vector_type& y)
{
    if (transa == AmpblasNoTrans)
    {
        _detail::gemv<AmpblasNoTrans>(av, alpha, a, x, beta, y);
    }
    else if (transa == AmpblasTrans)
    {
        _detail::gemv<AmpblasTrans>(av, alpha, a, x, beta, y);
    }
    else if (transa == AmpblasConjTrans)
    {
        _detail::gemv<AmpblasConjTrans>(av, alpha, a, x, beta, y);
    }
}

template <typename value_type>
void gemv(enum AMPBLAS_ORDER order, enum AMPBLAS_TRANSPOSE transa, int m, int n, value_type alpha, const value_type *a, int lda, const value_type *x, int incx, value_type beta, value_type* y, int incy)
{
	// quick return
	if (m == 0 || n == 0 || (alpha == value_type() && beta == value_type(1)))
		return;

    // TODO: column major requires a seperate implementation

	// error check
	if (order != AmpblasColMajor)
        not_yet_implemented();
	if (m < 0)
		argument_error("gemv", 3);
	if (n < 0)
		argument_error("gemv", 4);
	if (a == nullptr)
		argument_error("gemv", 6);
	if (lda < m)
		argument_error("gemv", 7);
	if (x == nullptr)
		argument_error("gemv", 8);
	if (y == nullptr)
		argument_error("gemv", 11);

	auto x_vec = make_vector_view((transa == AmpblasNoTrans ? n : m), x, incx);
    auto y_vec = make_vector_view((transa == AmpblasNoTrans ? m : n), y, incy);
    auto a_mat = make_matrix_view(m, n, a, lda);

	if (alpha == value_type())
	{
		if (beta == value_type())
			_detail::fill(get_current_accelerator_view(), y_vec.extent, value_type(), y_vec);
		else
			_detail::scale(get_current_accelerator_view(), y_vec.extent, beta, y_vec);
		return;
	}

	gemv(get_current_accelerator_view(), transa, alpha, a_mat, x_vec, beta, y_vec); 
}

} // namespace ampblas