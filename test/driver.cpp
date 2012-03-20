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
 */

#include <assert.h>
#include <stdio.h>
#include "ampcblas.h"

#define  RETURN_IF_FAIL(expr) \
    do { ampblas_result ar = (expr); if (ar != AMPBLAS_OK) return false; } while(0); 

bool test_saxpy_1()
{
	const int n = 100;

	float x[n], y[n], alpha = 17;
	for (int i=0; i<n; i++)
	{
		x[i] = (float)i;
		y[i] = (float)i * 10;
	}

	RETURN_IF_FAIL(ampblas_bind(x, n * sizeof(float)));
	RETURN_IF_FAIL(ampblas_bind(y, n * sizeof(float)));

	ampblas_saxpy(n, alpha, x, 1, y, 1);

	RETURN_IF_FAIL(ampblas_synchronize(y, n * sizeof(float)));

	for (int i=0; i<n; i++)
	{
		float actual = y[i];
		float expected = static_cast<float>(i * (10 + 17));

		if (actual != expected)
        {
			printf("FAIL at position %i, expected=%f, actual=%f\n", i, expected, actual);
            return false;
        }
	}

	RETURN_IF_FAIL(ampblas_unbind(x));
	RETURN_IF_FAIL(ampblas_unbind(y));

    return true;
}

int main()
{
	bool result = test_saxpy_1();

    result ? printf("Test successful\n") : printf("Test fail\n");
    
    return result;
}
