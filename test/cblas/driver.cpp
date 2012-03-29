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

#include <iostream>
#include "ampcblas.h"
#include "ampxblas.h"

#include "ampcblas_test_util.h"
#include "ampcblas_rt_test.h"
#include "ampcblas_xaxpy_test.h"

template<typename test_func>
inline bool run_test(test_func test, const char* test_name)
{
    bool result = test();

    std::cout << test_name << (result ? ": passed\n" : ": failed\n");

    return result;
}

int main()
{
    bool passed = true;

    // Testing runtime
    passed &= run_test(test_runtime_1<float>, "test_runtime_1<float>");
    passed &= run_test(test_runtime_1<double>, "test_runtime_1<double>");
    passed &= run_test(test_runtime_2<float>, "test_runtime_2<float>");
    passed &= run_test(test_runtime_2<double>, "test_runtime_2<double>");

    // Testing axpy
    passed &= run_test(test_axpy_1<float>, "test_axpy_1<float>");
    passed &= run_test(test_axpy_1<double>, "test_axpy_1<double>");
    passed &= run_test(test_axpy_2<float>, "test_axpy_2<float>");
    passed &= run_test(test_axpy_2<double>, "test_axpy_2<double>");
    passed &= run_test(test_axpy_3<float>, "test_axpy_3<float>");
    passed &= run_test(test_axpy_3<double>, "test_axpy_3<double>");
    passed &= run_test(test_axpy_4<float>, "test_axpy_4<float>");
    passed &= run_test(test_axpy_4<double>, "test_axpy_4<double>");
    
    return !passed;
}
