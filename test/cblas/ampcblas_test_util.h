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
 * Common utility routines for AMP CBLAS tests
 *
 *---------------------------------------------------------------------------*/
#pragma once 

#define  RETURN_IF_FAIL(expr) \
    do { ampblas_result ar = (expr); if (ar != AMPBLAS_OK) return false; } while(0) 

#define  RETURN_IF_KERNEL_FAIL(expr) \
    do { (expr); if (ampblas_get_last_error() != AMPBLAS_OK) return false; } while(0)

