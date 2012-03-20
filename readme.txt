AMPBLAS: C++ AMP BLAS Library

This library contains an adaptation of the cblas interface to BLAS for 
C++ AMP. At this point almost all interfaces are not implemented. One 
exception is the ampblas_saxpy which serves as a template for the 
implementation of other routines.

In order to use AMPBLAS you need to:
1) Include inc\ampblas.h in your source
2) Compile the .cpp files in the src subdirectory
3) Link the resulting object files with your sources

You can consult the build_ampblas.cmd for an example of how to build the 
library. You can also use the Visual Studio project file provided.
