/*----------------------------------------------------------------------------
 * Copyright � Microsoft Corp.
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
 * AMP CBLAS and AMP C++ BLAS library runtime header.
 *
 * This file contains APIs of data management and data transformation of bound buffer,
 * and APIs of accelerator_view selection and error handling. 
 *
 *---------------------------------------------------------------------------*/
#ifndef AMPBLAS_RUNTIME_H
#define AMPBLAS_RUNTIME_H

//----------------------------------------------------------------------------
// AMPBLAS error codes
//----------------------------------------------------------------------------
enum ampblas_result
{
    AMPBLAS_OK                          = 0,
    AMPBLAS_FAIL                        = 1<<0,
    AMPBLAS_BAD_LEADING_DIM             = 1<<1,
    AMPBLAS_BAD_RESOURCE                = 1<<2,
    AMPBLAS_INVALID_ARG                 = 1<<3,
    AMPBLAS_INVALID_BUFFER_SIZE         = 1<<4,
    AMPBLAS_OUT_OF_MEMORY               = 1<<5,
    AMPBLAS_UNBOUND_RESOURCE            = 1<<6,
    AMPBLAS_AMP_RUNTIME_ERROR           = 1<<7,
    AMPBLAS_NOT_SUPPORTED_FEATURE       = 1<<8,
    AMPBLAS_WIN_ERROR                   = 1<<9, // call ::GetLastError for info
};

//----------------------------------------------------------------------------
// AMPBLAS runtime for C++ BLAS 
//----------------------------------------------------------------------------
#ifdef __cplusplus
#include <exception>
#include <string>
#include <amp.h>

namespace ampblas 
{ 
//----------------------------------------------------------------------------
// ampblas_exception 
//----------------------------------------------------------------------------
class ampblas_exception : public std::exception
{
public:
    ampblas_exception(const char *const& msg, ampblas_result error_code) throw();
    ampblas_exception(const std::string& msg, ampblas_result error_code) throw();
    explicit ampblas_exception(ampblas_result error_code) throw();
    ampblas_exception(const ampblas_exception &other) throw();
    virtual ~ampblas_exception() throw();
    ampblas_result get_error_code() const throw();
    virtual const char *what() const throw();

private:
    ampblas_exception &operator=(const ampblas_exception &);
    std::string err_msg;
    ampblas_result err_code;
};

//----------------------------------------------------------------------------
//
// Data management APIs
//
// bind specifies a region of host memory which should become available
// to manipulation by AMPBLAS routines. A memory region should not be bound more
// than once. Also, binding of overlapping regions is not supported.
//
// To remove a binding call unbind. This function should only be invoked
// on the exact regions which were previously bound.
//
// Bindings are (could be) implemented using array_view and they expose a similar
// contract for sychronizing data to the main copy, discarding current changes,
// and refreshing the contents of the binding after it has been modified directly
// in host memory. The functions synchronize, discard, and 
// refresh, respectively, serve these purposes.
//
// The byte length of the bound buffer needs to be multiple of 4 bytes.
// 
// TODO: consider allowing multiple and concurrent bindings of the same buffer
//
//----------------------------------------------------------------------------
namespace _details
{
void bind(void *buffer_ptr, size_t byte_len);
void unbind(void *buffer_ptr);
void synchronize(void *buffer_ptr, size_t byte_len);
void discard(void *buffer_ptr, size_t byte_len);
void refresh(void *buffer_ptr, size_t byte_len);
concurrency::array_view<int32_t> get_array_view(const void *buffer_ptr, size_t byte_len);
} // nampespace _details

template<typename T> 
inline void bind(T *buffer_ptr, size_t element_count)
{
    _details::bind(buffer_ptr, element_count * sizeof(T));
}

template<typename T> 
inline void unbind(T *buffer_ptr)
{
    _details::unbind(buffer_ptr);
}

template<typename T> 
inline void synchronize(T *buffer_ptr, size_t element_count)
{
    _details::synchronize(buffer_ptr, element_count * sizeof(T));
}

template<typename T> 
inline void discard(T *buffer_ptr, size_t element_count)
{
    _details::discard(buffer_ptr, element_count * sizeof(T));
}

template<typename T> 
inline void refresh(T *buffer_ptr, size_t element_count)
{
    _details::refresh(buffer_ptr, element_count * sizeof(T));
}

// This API allows binding a single dimensional array_view as an AMPBLAS pointer. It can
// be used to import GPU-based arrays or staging arrays into the AMPBLAS sandbox.
// TODO: not supported yet.
template <typename value_type>
void* bind_array_view(const concurrency::array_view<value_type>& av);

// Conversely, a bound buffer can be obtained and manipulated as an array view.
template<typename value_type>
inline concurrency::array_view<value_type> get_array_view(const value_type *ptr, size_t element_count)
{
	auto av = _details::get_array_view(ptr, element_count * sizeof(value_type));
	return av.reinterpret_as<value_type>();
}

// set_current_accelerator_view set the accelerator view which will be used
// in subsequent AMPBLAS calls. There is no restriction on data access---once data 
// is bound using bind, it could be used on any accelerator.
//
// However, this function is not thread-safe. User of this function should synchronize to 
// avoid data race when calling this function from multiple threads or should synchronize 
// with calling get_current_accelerator_view.
void set_current_accelerator_view(const concurrency::accelerator_view& acc_view);

// This function is not thread-safe. But use of this function from multiple threads doesn't 
// need to be synchronized as long as user can guarantee there is no data race with calls to 
// set_current_accelerator_view.
concurrency::accelerator_view get_current_accelerator_view();

//----------------------------------------------------------------------------
// Data transformation operators and utility APIs
//----------------------------------------------------------------------------
template <int rank>
inline concurrency::index<rank> index_scalar(int s) restrict(cpu, amp)
{
	concurrency::index<rank> idx;
	for (int i=0; i<rank; i++)
		idx[i] = s;
	return idx;
}

template <int rank>
inline concurrency::index<rank> index_unity() restrict(cpu, amp)
{
	return index_scalar<rank>(1);
}

template <int rank>
inline concurrency::index<rank> last_index_of(const concurrency::extent<rank> &e) restrict(cpu, amp)
{
	concurrency::index<rank> idx;
	for (int i=0; i<rank; i++)
		idx[i] = e[i]-1;
	return idx;
}

inline concurrency::extent<1> make_extent(int n) restrict(cpu, amp)
{
	return concurrency::extent<1>(n);
}

//----------------------------------------------------------------------------
// stride_view
//
// template class stride_view wraps over an existing array-like class and 
// accesses the underlying storage with a stride.
//
//----------------------------------------------------------------------------
template <typename base_view_type>
class stride_view
{
public:
	typedef typename base_view_type::value_type value_type;
	static const int rank = base_view_type::rank;

	// The stride provided may be negative, in which case elements are retrieved in reverse order.
	stride_view(const base_view_type& bv, int stride, const concurrency::extent<rank>& logical_extent) restrict(cpu,amp)
		:base_view(bv), 
		 stride(stride), 
		 base_index(stride >= 0 ? concurrency::index<rank>() : -stride * last_index_of(logical_extent))
	{
	}

	~stride_view() restrict(cpu,amp) {}

	value_type& operator[] (const concurrency::index<rank>& idx) const restrict(cpu,amp)
	{
		return base_view[base_index + stride * idx];
	}

private:
    stride_view& operator=(const stride_view& rhs);

	const int stride;
	const concurrency::index<rank> base_index;
	base_view_type base_view;
};

template <typename base_view_type>
stride_view<base_view_type> make_stride_view(const base_view_type& bv, int stride, const concurrency::extent<base_view_type::rank>& logical_extent) restrict(cpu, amp)
{
	return stride_view<base_view_type>(bv, stride, logical_extent);
}

} // namespace ampblas
#endif // __cplusplus

//----------------------------------------------------------------------------
// AMPBLAS runtime for C BLAS 
//----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
//
// Data management API's 
// These are adapters to call the AMP C++ BLAS runtime data management functions. 
//
ampblas_result ampblas_bind(void *buffer_ptr, size_t byte_len);
ampblas_result ampblas_unbind(void *buffer_ptr);
ampblas_result ampblas_synchronize(void *buffer_ptr, size_t byte_len);
ampblas_result ampblas_discard(void *buffer_ptr, size_t byte_len);
ampblas_result ampblas_refresh(void *buffer_ptr, size_t byte_len);

// 
// ampblas_set_current_accelerator_view set the accelerator view which will be used
// in subsequent AMPBLAS calls. There is no restriction on data access---once data
// is bound using ampblas_bind, it could be used on any accelerator.
//
// However, this function is not thread-safe. User of the function should synchronize to 
// avid data race when calling this function from multiple-threads.
ampblas_result ampblas_set_current_accelerator_view(void * acc_view);

// TODO: query thread local error information
ampblas_result ampblas_get_last_errno();
void ampblas_get_last_err_message(const char**);

#ifdef __cplusplus
}
#endif

#endif //AMPBLAS_RUNTIME_H
