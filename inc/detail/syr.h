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
 * syr.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {

//-------------------------------------------------------------------------
// SYR
//-------------------------------------------------------------------------

template <enum AMPBLAS_UPLO uplo, typename trans_op, typename alpha_type, typename x_vector_type, typename a_value_type>
void syr(const concurrency::accelerator_view& av, alpha_type alpha, const x_vector_type& x, const concurrency::array_view<a_value_type,2>& a )
{
    concurrency::parallel_for_each (
        av,
        a.extent,
        [=] (concurrency::index<2> idx_a) restrict(amp)
        {
            concurrency::index<1> idx_x(idx_a[1]); // "i"
            concurrency::index<1> idx_xt(idx_a[0]); // "j"

            if ( uplo == AmpblasUpper && idx_a[0] >= idx_a[1] || uplo == AmpblasLower && idx_a[1] >= idx_a[0] )
            {
                auto a_value = a[idx_a];
                
                // Note that the imaginary parts of the diagonal elements need
                // not be set, they are assumed to be zero, and on exit they
                // are set to zero.
                if (idx_x == idx_xt)
                    _detail::only_real(a_value);

                a[idx_a] = a_value + (alpha * x[idx_x] * trans_op::op(x[idx_xt]));
            }
        }
    );
}

template <typename trans_op, typename alpha_type, typename value_type>
void syr(enum AMPBLAS_ORDER order, enum AMPBLAS_UPLO uplo, int n, alpha_type alpha, const value_type *x, int incx, value_type *a, int lda)
{
    // recursive order adjustment
    if (order == AmpblasRowMajor)
    {
        syr<trans_op>(AmpblasColMajor, uplo == AmpblasUpper ? AmpblasLower : AmpblasUpper, n, alpha, x, incx, a, lda);
        return;
    }

    // quick return
    if (n == 0 || alpha == alpha_type())
        return;

    // argument check
    if (n < 0)
        argument_error("syr", 3);
    if (x == nullptr)
        argument_error("syr", 5);
    if (a == nullptr)
        argument_error("syr", 7);
    if (lda < n)
        argument_error("syr", 8);

    // create views
    auto x_vec = make_vector_view(n, x, incx);
    auto a_mat = make_matrix_view(n, n, a, lda);

    // call generic implementation
    if (uplo == AmpblasUpper)
        syr<AmpblasUpper,trans_op>(get_current_accelerator_view(), alpha, x_vec, a_mat);
    else
        syr<AmpblasLower,trans_op>(get_current_accelerator_view(), alpha, x_vec, a_mat);
}

} // namespace ampblas
