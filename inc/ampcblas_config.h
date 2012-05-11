#pragma once

#include <assert.h>
#include "ampcblas.h"
#include "ampblas_complex.h"
#include "ampblas_ccomplex.h"

typedef ampblas::complex<float>  fcomplex;
typedef ampblas::complex<double> dcomplex;

inline const fcomplex* ampblas_cast(const ampblas_fcomplex* ptr)
{
    return reinterpret_cast<const fcomplex*>(ptr);
}

inline fcomplex* ampblas_cast(ampblas_fcomplex* ptr)
{
    return reinterpret_cast<fcomplex*>(ptr);
}

inline const dcomplex* ampblas_cast(const ampblas_dcomplex* ptr)
{
    return reinterpret_cast<const dcomplex*>(ptr);
}

inline dcomplex* ampblas_cast(ampblas_dcomplex* ptr)
{
    return reinterpret_cast<dcomplex*>(ptr);
}

#define AMPBLAS_CHECKED_CALL(...)           \
{                                           \
    ampblas_result re = AMPBLAS_OK;         \
    try                                     \
    {                                       \
        (__VA_ARGS__);                      \
    }                                       \
    catch (ampblas::ampblas_exception &e)   \
    {                                       \
        re = e.get_error_code();            \
    }                                       \
    catch (concurrency::runtime_exception&) \
    {                                       \
        re = AMPBLAS_AMP_RUNTIME_ERROR;     \
    }                                       \
    catch (std::bad_alloc&)                 \
    {                                       \
        re = AMPBLAS_OUT_OF_MEMORY;         \
    }                                       \
    catch (...)                             \
    {                                       \
        re = AMPBLAS_INTERNAL_ERROR;        \
    }                                       \
    ampblas_set_last_error(re);             \
}
