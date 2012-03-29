/* 
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
 * Tests for AMP CBLAS interface ampblas_[s,d,c,z]axpy
 *
 *---------------------------------------------------------------------------*/
#pragma once 

#include <vector>

//------------------------------------------------------------------------------------
// Testing ampblas_xaxpy input arguments
//------------------------------------------------------------------------------------
template<typename T>
bool test_axpy_1()
{
	const int n = 100;
	T x[n], y[n], alpha = 17;
	
    // Initialize buffers 
    for (int i=0; i<n; i++)
	{
		x[i] = (T)i;
		y[i] = (T)i * 10;
	}

    //-------------------------------------------------
    // Testing n = 0, 
    // ampblas_xaxpy should return without modifying y
    //-------------------------------------------------
	RETURN_IF_FAIL(ampblas_bind(x, n * sizeof(T)));
	RETURN_IF_FAIL(ampblas_bind(y, n * sizeof(T)));

    RETURN_IF_KERNEL_FAIL(ampblas_xaxpy(0, alpha, x, 1, y, 1));
	RETURN_IF_FAIL(ampblas_synchronize(y, n * sizeof(T)));

	RETURN_IF_FAIL(ampblas_unbind(x));
	RETURN_IF_FAIL(ampblas_unbind(y));

    // Verify result
	for (int i=0; i<n; i++)
	{
		T actual = y[i];
		T expected = y[i];
		if (actual != expected)
        {
            return false;
        }
	}

    //-------------------------------------------------
    // Testing alpha = 0
    // ampblas_xaxpy should return without modifying y
    //-------------------------------------------------
	RETURN_IF_FAIL(ampblas_bind(x, n * sizeof(T)));
	RETURN_IF_FAIL(ampblas_bind(y, n * sizeof(T)));

    RETURN_IF_KERNEL_FAIL(ampblas_xaxpy(n, (T)0, x, 1, y, 1));
	RETURN_IF_FAIL(ampblas_synchronize(y, n * sizeof(T)));

	RETURN_IF_FAIL(ampblas_unbind(x));
	RETURN_IF_FAIL(ampblas_unbind(y));

    // Verify result
	for (int i=0; i<n; i++)
	{
		T actual = y[i];
		T expected = y[i];
		if (actual != expected)
        {
            return false;
        }
	}

    return true;
}

//------------------------------------------------------------------------------------
// Testing ampblas_xaxpy: y[i*incy] += alpha * x[i*incx], 
// where incx > 0, incy > 0
//------------------------------------------------------------------------------------
template<typename T>
bool test_axpy_2()
{
	const int n = 100;
	T x[n], y[n], alpha = 17;

	for (int i=0; i<n; i++)
	{
		x[i] = (T)i;
		y[i] = (T)i * 10;
	}

    // Bind buffers
	RETURN_IF_FAIL(ampblas_bind(x, n * sizeof(T)));
	RETURN_IF_FAIL(ampblas_bind(y, n * sizeof(T)));

    // Run test
    RETURN_IF_KERNEL_FAIL(ampblas_xaxpy(n, alpha, x, 1, y, 1));
	RETURN_IF_FAIL(ampblas_synchronize(y, n * sizeof(T)));

    // Unbind buffers
	RETURN_IF_FAIL(ampblas_unbind(x));
	RETURN_IF_FAIL(ampblas_unbind(y));

    // Verify result
	for (int i=0; i<n; i++)
	{
		T actual = y[i];
		T expected = static_cast<T>(i * (10 + 17));
		if (actual != expected)
        {
            return false;
        }
	}

    return true;
}

//------------------------------------------------------------------------------------
// Testing ampblas_xaxpy: y[i*incy] += alpha * x[i*incx], 
// using sub-regions of bound buffers
// where incx > 0, incy > 0
//------------------------------------------------------------------------------------
template<typename T>
bool test_axpy_3()
{
	const int n = 100;
    const int offset = 2;

	T x[n], y[n], alpha = 17;
	for (int i=0; i<n; i++)
	{
		x[i] = (T)i;
		y[i] = (T)i * 10;
	}

    // Bind buffers
	RETURN_IF_FAIL(ampblas_bind(x, n * sizeof(T)));
	RETURN_IF_FAIL(ampblas_bind(y, n * sizeof(T)));

    // Run test
    RETURN_IF_KERNEL_FAIL(ampblas_xaxpy(n-offset, alpha, x+offset, 1, y+offset, 1));
	RETURN_IF_FAIL(ampblas_synchronize(y, n * sizeof(T)));

    // Unbind buffers
	RETURN_IF_FAIL(ampblas_unbind(x));
	RETURN_IF_FAIL(ampblas_unbind(y));

    // Verify result
	for (int i=offset; i<n; i++)
	{
		T actual = y[i];
		T expected = static_cast<T>(i * (10 + 17));
		if (actual != expected)
        {
            return false;
        }
	}

    return true;
}

//------------------------------------------------------------------------------------
// Testing ampblas_xaxpy: y[i*incy] += alpha * x[i*incx], 
// where incx < 0, incy < 0
//------------------------------------------------------------------------------------
template<typename T>
bool test_axpy_4()
{
	const int n = 100;
	T x[n], y[n], alpha = 17;

	for (int i=0; i<n; i++)
	{
		x[i] = (T)i;
		y[i] = (T)i * 10;
	}

    // Bind buffers
	RETURN_IF_FAIL(ampblas_bind(x, n * sizeof(T)));
	RETURN_IF_FAIL(ampblas_bind(y, n * sizeof(T)));

    // Run test
    RETURN_IF_KERNEL_FAIL(ampblas_xaxpy(n, alpha, x, -1, y, -1));
	RETURN_IF_FAIL(ampblas_synchronize(y, n * sizeof(T)));

    // Unbind buffers
	RETURN_IF_FAIL(ampblas_unbind(x));
	RETURN_IF_FAIL(ampblas_unbind(y));

    // Verify result
	for (int i=0; i<n; i++)
	{
		T actual = y[i];
		T expected = static_cast<T>(i * (10 + 17));

		if (actual != expected)
        {
            return false;
        }
	}

    return true;
}
