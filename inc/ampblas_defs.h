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
 * typedefs for AMPBLAS.
 *
 * This file contains common typedefs for AMP CBLAS and AMP C++ BLAS.
 *
 *---------------------------------------------------------------------------*/
#ifndef AMPBLAS_DEFS_H
#define AMPBLAS_DEFS_H

//----------------------------------------------------------------------------
// Enumerated and derived types
//----------------------------------------------------------------------------
enum AMPBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum AMPBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum AMPBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum AMPBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum AMPBLAS_SIDE {CblasLeft=141, CblasRight=142};

#endif // AMPBLAS_DEFS_H
