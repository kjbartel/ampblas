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
 * ger.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {

//-------------------------------------------------------------------------
// GER
//   performs the rank 1 operation
//
//     A := alpha*X*transpose(Y) + A,
//
//  where alpha is a scalar, X is an M element vector, Y is an N element
//  vector and A is an M by N matrix.
//-------------------------------------------------------------------------

template <typename trans_op, typename alpha_type, typename x_vector_type, typename y_vector_type, typename a_value_type>
void ger(const concurrency::accelerator_view& av, alpha_type alpha, const x_vector_type& x, const y_vector_type& y, const concurrency::array_view<a_value_type,2>& a )
{
    concurrency::parallel_for_each ( 
        av,
        a.extent,
        [=] (concurrency::index<2> idx_a) restrict(amp)
        {
            concurrency::index<1> idx_x(idx_a[1]);
            concurrency::index<1> idx_y(idx_a[0]);

            a[idx_a] += alpha * x[idx_x] * trans_op::op(y[idx_y]);
        }
    );
}

template <typename value_type, typename trans_op>
void ger(enum AMPBLAS_ORDER order, int m, int n, value_type alpha, const value_type *x, int incx, const value_type *y, int incy, value_type *a, int lda)
{
    // recursive order adjustment
    if (order == AmpblasRowMajor)
    {
        ger<value_type,trans_op>(AmpblasColMajor, n, m, alpha, y, incy, x, incx, a, lda);
        return;
    }

    // quick return
    if (m == 0 || n == 0 || alpha == value_type())
        return;

    // argument check
    if (m < 0)
        argument_error("ger", 2);
    if (n < 0)
        argument_error("ger", 3);
    if (x == nullptr)
        argument_error("ger", 5);
    if (y == nullptr)
        argument_error("ger", 7);
    if (a == nullptr)
        argument_error("ger", 9);
    if (lda < (order == AmpblasColMajor ? m : n))
        argument_error("ger", 10);

    // create views
    auto x_vec = make_vector_view(m, x, incx);
    auto y_vec = make_vector_view(n, y, incy);
    auto a_mat = make_matrix_view(m, n, a, lda);

    // call generic implementation
    ger<trans_op>(get_current_accelerator_view(), alpha, x_vec, y_vec, a_mat);
}

} // namespace ampblas