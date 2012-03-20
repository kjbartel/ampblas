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
 * AMPBLAS runtime implementation file.
 *
 * This file contains implementation of core runtime routines for the AMPBLAS library. 
 * Mostly, memory management of bound buffers, and error handling routines.
 *
 *---------------------------------------------------------------------------*/
#include <map>
#include <memory>
#include <amp.h>
#include <assert.h>

#include "ampblas_runtime.h"

namespace ampblas 
{
//----------------------------------------------------------------------------
// ampblas_exception 
//----------------------------------------------------------------------------
ampblas_exception::ampblas_exception(ampblas_result error_code) throw()
    : err_code(error_code) 
{
}

ampblas_exception::ampblas_exception(const char *const& msg, ampblas_result error_code) throw()
    : err_msg(msg), err_code(error_code) 
{
}

ampblas_exception::ampblas_exception(const std::string& msg, ampblas_result error_code) throw()
    : err_msg(msg), err_code(error_code) 
{
}

ampblas_exception::ampblas_exception(const ampblas_exception &other) throw()
    : std::exception(other), err_msg(other.err_msg), err_code(other.err_code) 
{
}

ampblas_exception::~ampblas_exception() throw()
{
}

ampblas_result ampblas_exception::get_error_code() const throw()
{
    return err_code;
}

const char *ampblas_exception::what() const throw()
{
    return  err_msg.data();
}

namespace _details 
{
class amp_buffer;

namespace 
{
concurrency::critical_section g_allocations_cs;
std::map<const void*, amp_buffer*> g_allocations;
std::unique_ptr<concurrency::accelerator_view> g_curr_accelerator_view; 
}
#define PTR_U64(ptr)      reinterpret_cast<uint64_t>(ptr)

//----------------------------------------------------------------------------
// amp_buffer
//
// adapter class to associate a raw buffer with an array_view 
//----------------------------------------------------------------------------
class amp_buffer
{
public:
	amp_buffer(void *buffer_ptr, size_t buffer_byte_len)
		:mem_base(reinterpret_cast<int32_t*>(buffer_ptr)), 
		 byte_len(buffer_byte_len),
         // buffer_byte_len has been asserted to be multiple of int32_t size, and the buffer_byte_len/sizeof(int32_t)
         // is no greater than INT_MAX. So we can safely cast the result to int. 
		 storage(concurrency::extent<1>(static_cast<int>(static_cast<uint64_t>(buffer_byte_len)/sizeof(int32_t))), 
                 reinterpret_cast<int32_t*>(buffer_ptr))
	{
    }

    // Check whether this bound buffer contains the region starting at buffer_ptr. 
    bool contain(const void *buffer_ptr, size_t buffer_byte_len) const
    {
        if (mem_base <= buffer_ptr && (PTR_U64(mem_base)+byte_len >= PTR_U64(buffer_ptr)+buffer_byte_len))
        {
            return true;
        }
        return false;
    }

    // Check whether this bound buffer completely excludes the region starting at buffer_ptr
    bool exclusive_with(const void *buffer_ptr, size_t buffer_byte_len) const
    {
        if ((PTR_U64(mem_base)+byte_len <= PTR_U64(buffer_ptr)) ||
            (PTR_U64(mem_base) >= PTR_U64(buffer_ptr)+buffer_byte_len))
        {
            return true;
        }
        return false;
    }

    // Check whether this bound buffer overlaps with the region starting at buffer_ptr
    // For the case the region is contained in the bound buffer, it returns false. 
    bool overlap(const void *buffer_ptr, size_t buffer_byte_len) const
    {
        return !exclusive_with(buffer_ptr, buffer_byte_len) && !contain(buffer_ptr, buffer_byte_len);
    }
    
	const int32_t *mem_base;
	const size_t byte_len;
	concurrency::array_view<int32_t> storage;
private:
    amp_buffer& operator=(const amp_buffer& right);
};

//----------------------------------------------------------------------------
// Data management facilities 
//----------------------------------------------------------------------------
#define ASSERT_BUFFER_SIZE(buf_byte_len) \
    assert(((buf_byte_len) % sizeof(int32_t) == 0) && "buffer length must be multiple of int32_t size"); \
    assert((static_cast<uint64_t>(buf_byte_len) <= static_cast<uint64_t>(INT_MAX)*sizeof(int32_t)) && "buffer length overflow");

static inline void check_buffer_length(size_t byte_len, const char* err_prefix)
{
    if (byte_len % sizeof(int32_t) != 0)
    {
        throw ampblas_exception(std::string(err_prefix) + "buffer length must be multiple of int32_t size", AMPBLAS_BAD_RESOURCE);
    }
    else if (static_cast<uint64_t>(byte_len) > static_cast<uint64_t>(INT_MAX)*sizeof(int32_t))
    {
        throw ampblas_exception(std::string(err_prefix) +  "buffer length overflow", AMPBLAS_BAD_RESOURCE);
    }
}

// Find the bound buffer which contains a region starting at buffer_ptr with byte length byte_len
//
// returns a bound buffer if the bound buffer contains the buffer [buffer_ptr, buffer_ptr+byte_len)
// returns nullptr if the buffer [buffer_ptr, buffer_ptr+byte_len) is exclusive with any bound buffer 
// throws an ampblas_exception if the buffer [buffer_ptr, buffer_ptr+byte_len) overlaps with another bound buffer
//
// The caller of this function has to hold g_allocations_cs lock for synchronization. 
static amp_buffer* find_amp_buffer(const void *buffer_ptr, size_t byte_len)
{
    ASSERT_BUFFER_SIZE(byte_len);

    amp_buffer *ampbuff = nullptr;

    // No amp_buffer in cache yet. 
	if (g_allocations.size() == 0) 
	{
        return nullptr;
	}

	auto it = g_allocations.lower_bound(const_cast<void*>(buffer_ptr));
    if (it == g_allocations.end())
    {
        it--; 
        ampbuff = it->second;
        if (ampbuff->contain(buffer_ptr, byte_len)) return ampbuff;
        else if (ampbuff->exclusive_with(buffer_ptr, byte_len)) return nullptr;
        else throw ampblas_exception("ampblas::find_amp_buffer: Buffer overlapped", AMPBLAS_UNBOUND_RESOURCE);
    }
    else if (it == g_allocations.begin())
    {
        ampbuff = it->second;
        if (ampbuff->contain(buffer_ptr, byte_len)) return ampbuff;
        else if (ampbuff->exclusive_with(buffer_ptr, byte_len)) return nullptr;
        else throw ampblas_exception("ampblas::find_amp_buffer: Buffer overlapped", AMPBLAS_UNBOUND_RESOURCE);
    }
    else
    {
        ampbuff = it->second;
        it--;
        auto ampbuff_prev = it->second;

        if (ampbuff->contain(buffer_ptr, byte_len)) return ampbuff;
        else if (ampbuff_prev->contain(buffer_ptr, byte_len)) return ampbuff_prev;
        else if (ampbuff->exclusive_with(buffer_ptr, byte_len) && ampbuff_prev->exclusive_with(buffer_ptr, byte_len)) return nullptr;
        else throw ampblas_exception("ampblas::find_amp_buffer: Buffer overlapped", AMPBLAS_UNBOUND_RESOURCE);
    }
}

concurrency::array_view<int32_t> get_array_view(const void *buffer_ptr, size_t byte_len)
{
    const amp_buffer *ampbuff = nullptr;
    try
    {
		concurrency::critical_section::scoped_lock scope_lock(g_allocations_cs);
        ampbuff = find_amp_buffer(buffer_ptr, byte_len);
    }
    catch (ampblas_exception&)
    {
        throw ampblas_exception("ampblas::get_array_view: Buffer overlapped", AMPBLAS_BAD_RESOURCE);
    }

    if (ampbuff == nullptr) 
    {
        throw ampblas_exception("ampblas::get_array_view: Unbound resource", AMPBLAS_UNBOUND_RESOURCE);
    }

    check_buffer_length(byte_len, "ampblas::get_array_view: ");
	int elem_len = static_cast<int>(byte_len / sizeof(int32_t));

    uint64_t byte_offset = PTR_U64(buffer_ptr) - PTR_U64(ampbuff->mem_base);
    if (byte_offset % sizeof(int32_t) != 0)
    {
        throw ampblas_exception("ampblas::get_array_view: Invalid buffer argument", AMPBLAS_INVALID_ARG);
    }

    uint64_t elem_offset = byte_offset / sizeof(int32_t);
    assert(elem_offset <= INT_MAX);

	return ampbuff->storage.section(concurrency::index<1>(static_cast<int>(elem_offset)), concurrency::extent<1>(elem_len));
}

void bind(void *buffer_ptr, size_t byte_len)
{
    check_buffer_length(byte_len, "ampblas::bind: ");
	try
	{
		concurrency::critical_section::scoped_lock scope_lock(g_allocations_cs);

        amp_buffer *ampbuff = nullptr;
        ampbuff = find_amp_buffer(buffer_ptr, byte_len);
 
        if (ampbuff != nullptr)
        {
			throw ampblas_exception("ampblas::bind: Duplicate binding", AMPBLAS_BAD_RESOURCE);
        }
        
		std::unique_ptr<amp_buffer> buff(new amp_buffer(buffer_ptr, byte_len));

        auto it = g_allocations.insert(std::make_pair(buffer_ptr, buff.get()));
		assert(it.second == true);

        buff.release();
	}
    catch (ampblas_exception&)
    {
        throw ampblas_exception("ampblas::bind: Buffer overlapped", AMPBLAS_BAD_RESOURCE);
    }
}

void unbind(void *buffer_ptr)
{
	std::unique_ptr<amp_buffer> ampbuff;
	{
		concurrency::critical_section::scoped_lock scope_lock(g_allocations_cs);

        auto it = g_allocations.find(buffer_ptr);
        if (it == g_allocations.end())
        {
            throw ampblas_exception("ampblas::unbind: Unbound resource", AMPBLAS_UNBOUND_RESOURCE);
        }

        ampbuff.reset(it->second);
		g_allocations.erase(it);
	}
}

void synchronize(void *buffer_ptr, size_t byte_len)
{
    check_buffer_length(byte_len, "ampblas::synchronize: ");
	get_array_view(buffer_ptr, byte_len).synchronize();
}

void discard(void *buffer_ptr, size_t byte_len)
{
    check_buffer_length(byte_len, "ampblas::discard: ");
	get_array_view(buffer_ptr, byte_len).discard_data();
}

void refresh(void *buffer_ptr, size_t byte_len)
{
    check_buffer_length(byte_len, "ampblas::refresh: ");
    get_array_view(buffer_ptr, byte_len).refresh();
}

} // namespace _details


// This function is not thread-safe. User of the function should synchronize to avid data race
// when calling this function from multiple-threads or synchronize with calling 
// get_current_accelerator_view.
void set_current_accelerator_view(const concurrency::accelerator_view& acc_view)
{
    if (_details::g_curr_accelerator_view.get() == nullptr || *_details::g_curr_accelerator_view != acc_view)   
    { 
        _details::g_curr_accelerator_view.reset(new concurrency::accelerator_view(acc_view)); 
    }
}

// This function is not thread-safe. But use of this function doesn't need to be synchronized as long
// as user can guarantee there is no data race with the calling of set_current_accelerator_view.
concurrency::accelerator_view get_current_accelerator_view()
{
    if (_details::g_curr_accelerator_view.get() == nullptr)   
    {
        return concurrency::accelerator().default_view;
    }

    return *_details::g_curr_accelerator_view; 
}

} // namespace ampblas

//----------------------------------------------------------------------------
// AMPBLAS runtime facilities for C BLAS 
//----------------------------------------------------------------------------
#define AMPBLAS_CHECKED_CALL(expr) \
    { \
        ampblas_result re = AMPBLAS_OK; \
        try \
        { \
            (expr);\
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
            re = AMPBLAS_WIN_ERROR; \
        } \
        return re; \
    }

extern "C" ampblas_result ampblas_bind(void *buffer_ptr, size_t byte_len)
{
    AMPBLAS_CHECKED_CALL(ampblas::_details::bind(buffer_ptr, byte_len));
}

extern "C" ampblas_result ampblas_unbind(void *buffer_ptr)
{
    AMPBLAS_CHECKED_CALL(ampblas::_details::unbind(buffer_ptr));
}

extern "C" ampblas_result ampblas_synchronize(void *buffer_ptr, size_t byte_len)
{
    AMPBLAS_CHECKED_CALL(ampblas::_details::synchronize(buffer_ptr, byte_len));
}

extern "C" ampblas_result ampblas_discard(void *buffer_ptr, size_t byte_len)
{
	AMPBLAS_CHECKED_CALL(ampblas::_details::discard(buffer_ptr, byte_len));
}

extern "C" ampblas_result ampblas_refresh(void *buffer_ptr, size_t byte_len)
{
    AMPBLAS_CHECKED_CALL(ampblas::_details::refresh(buffer_ptr, byte_len));
}

extern "C" ampblas_result ampblas_set_current_accelerator_view(void *acc_view)
{
    if (acc_view == nullptr)
        return AMPBLAS_INVALID_ARG;

    const concurrency::accelerator_view& av = *reinterpret_cast<concurrency::accelerator_view*>(acc_view);
	AMPBLAS_CHECKED_CALL(ampblas::set_current_accelerator_view(av));
}

extern "C" void ampblas_xerbla(const char *srname, int * info)
{
	printf("** On entry to %6s, parameter number %2i had an illegal value\n", srname, *info);
}

