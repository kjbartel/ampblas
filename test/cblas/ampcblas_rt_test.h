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
 * Tests for AMP BLAS runtime
 *
 *---------------------------------------------------------------------------*/
#pragma once 

#include <algorithm>
#include <vector>
#include <amp.h>
using namespace concurrency;

//------------------------------------------------------------------------------------
// Negative testing on AMP BLAS runtime
//------------------------------------------------------------------------------------
template<typename T>
bool test_runtime_1()
{
    bool passed = true; 
	const int n = 99;
    T x[n];
    for (int i=0; i<n; i++)
    {
        x[i] = (T)i;
    }
    
    //-------------------------------------------------
    // Testing ampblas_bind 
    //-------------------------------------------------

    // buffer length is not multiple of int32_t
    auto result = ampblas_bind(x, n * sizeof(int16_t));
    passed &= (result == AMPBLAS_BAD_RESOURCE);

    result = ampblas_bind(x, n * sizeof(T));
    passed &= (result == AMPBLAS_OK);

    // duplicate binding 
    result = ampblas_bind(x, n * sizeof(T));
    passed &= (result == AMPBLAS_BAD_RESOURCE);

    // contained buffer binding 
    result = ampblas_bind(x+2, (n-3) * sizeof(T));
    passed &= (result == AMPBLAS_BAD_RESOURCE);

    // overlapped buffer binding 
    result = ampblas_bind(x+2, n * sizeof(T));
    passed &= (result == AMPBLAS_BAD_RESOURCE);

    // overlapped buffer binding 
    result = ampblas_bind(x-2, n * sizeof(T));
    passed &= (result == AMPBLAS_BAD_RESOURCE);

    //-------------------------------------------------
    // Testing ampblas_unbind
    //-------------------------------------------------
    T y[n];
    for (int i=0; i<n; i++)
    {
        y[i] = (T)i;
    }
	result = ampblas_unbind(y);
    passed &= (result == AMPBLAS_UNBOUND_RESOURCE);

	result = ampblas_unbind(x+2);
    passed &= (result == AMPBLAS_UNBOUND_RESOURCE);

	result = ampblas_unbind(x);
    passed &= (result == AMPBLAS_OK);
    // now no buffer is bound

    //-------------------------------------------------
    // Testing ampblas_synchronize
    //-------------------------------------------------
    result = ampblas_synchronize(x, n * sizeof(T));
    passed &= (result == AMPBLAS_UNBOUND_RESOURCE);
   
    result = ampblas_bind(x, n * sizeof(T));
    passed &= (result == AMPBLAS_OK);

    result = ampblas_synchronize(x+2, (n-2) * sizeof(T));
    passed &= (result == AMPBLAS_OK);

    result = ampblas_synchronize(reinterpret_cast<char*>(x)+2, (n-2) * sizeof(T));
    passed &= (result == AMPBLAS_INVALID_ARG);

	result = ampblas_unbind(x);
    passed &= (result == AMPBLAS_OK);

    return passed;
}

//------------------------------------------------------------------------------------
// Testing set_current_accelerator_view 
//------------------------------------------------------------------------------------
template<typename T>
bool test_runtime_2()
{
    // find all non-emulated accelerators
    std::vector<accelerator> accls = accelerator::get_all();
    accls.erase(std::remove_if(accls.begin(), accls.end(), [](accelerator &accl) {
        return accl.is_emulated;
    }), accls.end());

    // run ampblas_xaxpy on every accelerator
    const int num_accls = static_cast<int>(accls.size());
	const int n = 100;
	T x[n], y[n], alpha = 17;

	for (int i=0; i<n; i++)
	{
		x[i] = (T)i;
		y[i] = (T)i * 10;
	}

	RETURN_IF_FAIL(ampblas_bind(x, n * sizeof(T)));
	RETURN_IF_FAIL(ampblas_bind(y, n * sizeof(T)));

    for (int i=0; i<num_accls; i++)
    {
        accelerator_view accv = accls[i].default_view;
        ampblas_set_current_accelerator_view(&accv);
        RETURN_IF_KERNEL_FAIL(ampblas_xaxpy(n, alpha, x, 1, y, 1));
	    RETURN_IF_FAIL(ampblas_synchronize(y, n * sizeof(T)));
    }

	RETURN_IF_FAIL(ampblas_unbind(x));
	RETURN_IF_FAIL(ampblas_unbind(y));

    // verify result
	for (int i=0; i<n; i++)
	{
		T actual = y[i];
        T expected = static_cast<T>(i * (10 + 17*num_accls));

		if (actual != expected)
        {
            return false;
        }
	}

    return true;
}

