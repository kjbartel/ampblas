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
 * symv.h
 *
 *---------------------------------------------------------------------------*/

#include "../ampblas_config.h"

namespace ampblas {

//-------------------------------------------------------------------------
// SYMV
//   This simpler implementation does not take data duplicity into account.
//   A more advanced solution will use tiles, shared memory, and atomic 
//   operations to account realize that each read of 'a' is two seperate 
//   values.  
//-------------------------------------------------------------------------

template <typename trans_op, typename value_type, typename x_vector_type, typename y_vector_type>
void symv(const concurrency::accelerator_view& av, enum AMPBLAS_UPLO uplo, value_type alpha, const concurrency::array_view<const value_type,2>& a, x_vector_type x, value_type beta, y_vector_type y)
{
    concurrency::parallel_for_each(av, y.extent, [=] (concurrency::index<1> y_idx) restrict(amp)
    {
        value_type result = value_type();
        
        for (int n = 0; n < x.extent[0]; ++n)
        {
            concurrency::index<2> a_idx;
            concurrency::index<1> x_idx(n);

            value_type a_value;

            if (uplo == AmpblasLower)
            {
                if (n > y_idx[0])
                {
                    a_idx = concurrency::index<2>(y_idx[0], n);
                    a_value = a[a_idx];
                }
                else
                {
                    a_idx = concurrency::index<2>(n, y_idx[0]);
                    a_value = trans_op::op(a[a_idx]);
                }
            }
            else 
            {
                if (n < y_idx[0])
                {
                    a_idx = concurrency::index<2>(y_idx[0], n);
                    a_value = a[a_idx];
                }
                else
                {
                    a_idx = concurrency::index<2>(n, y_idx[0]);
                    a_value =  trans_op::op(a[a_idx]);
                }
            }			
            
            result += a_value * x[x_idx];
        }

        y[y_idx] = alpha * result + beta * y[y_idx];

    });
}

template <typename trans_op, typename value_type>
void symv(enum AMPBLAS_ORDER order, enum AMPBLAS_UPLO uplo, int n, value_type alpha, const value_type *a, int lda, const value_type* x, int incx, value_type beta, value_type *y, int incy)
{
    // recursive order adjustment
	if (order == AmpblasRowMajor)
    {
        symv<trans_op>(AmpblasColMajor, uplo == AmpblasLower ? AmpblasUpper : AmpblasLower,  n, alpha, a, lda, x, incx, beta, y, incy);
        return;
    }

	// quick return
	if (n == 0 || (alpha == value_type() && beta == value_type(1)))
		return;

	// argument check
	if (n < 0)
        argument_error("symv", 3);
    if (a == nullptr)
        argument_error("symv", 5);
    if (lda < n)
        argument_error("symv", 6);
    if (x == nullptr)
        argument_error("symv", 7);
    if (y == nullptr)
        argument_error("symv", 9);

	// create views
	auto x_vec = make_vector_view(n, x, incx);
    auto y_vec = make_vector_view(n, y, incy);
    auto a_mat = make_matrix_view(n, n, a, lda);

	// call generic implementation
	symv<trans_op>(get_current_accelerator_view(), uplo, alpha, a_mat, x_vec, beta, y_vec);
}

} // namespace ampblas
