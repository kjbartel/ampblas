#pragma once

#include <assert.h>
#include "ampcblas.h"
#include "ampblas_complex.h"

typedef ampblas::complex<float>  fcomplex;
typedef ampblas::complex<double> dcomplex;

#define AMPBLAS_CHECKED_CALL(...) \
    { \
        ampblas_result re = AMPBLAS_OK; \
        try \
        { \
            (__VA_ARGS__);\
        } \
        catch (ampblas::ampblas_exception &e) \
        { \
            re = e.get_error_code(); \
        } \
        catch (concurrency::runtime_exception&) \
        { \
            re = AMPBLAS_AMP_RUNTIME_ERROR; \
        } \
        catch (std::bad_alloc&) \
        { \
            re = AMPBLAS_OUT_OF_MEMORY; \
        } \
        catch (...) \
        { \
            re = AMPBLAS_INTERNAL_ERROR; \
        } \
        ampblas_set_last_error(re); \
    }
