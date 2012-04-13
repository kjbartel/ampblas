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
 * ampblas_complex.h
 *
 * This file contains the complex class for AMPBLAS. 
 *
 *---------------------------------------------------------------------------*/
#ifndef AMPBLAS_COMPLEX_H
#define AMPBLAS_COMPLEX_H

#ifdef __cplusplus
#include <ostream>

namespace ampblas
{
template<typename T> 
class complex
{
private: 
    T   real_val;
    T   imag_val;

public:
    typedef T   value_type;

    complex(const T& real = T(), const T& imag = T()) restrict(amp, cpu) 
        : real_val(real), imag_val(imag) 
    {
    }

    template<typename X>
    complex(const complex<X>& rhs) restrict(amp, cpu)
        : real_val(rhs.real_val), imag_val(rhs.imag_val)
    {
    }

    // assign a complex
    template<typename X>
    complex& operator=(const complex<X>& rhs) restrict(amp, cpu)
    {
        real_val = static_cast<T>(rhs.real_val);
        imag_val = static_cast<T>(rhs.imag_val);

        return *this;
    }

    // assign a real
    complex& operator=(const T& real) restrict(amp, cpu)
    {
        real_val = real;
        imag_val = T();

        return *this;
    }

    // add a complex
    template<typename X>
    complex& operator+=(const complex<X>& rhs) restrict(amp, cpu)
    {
        // Temporaries might help coalescing loads 
        auto l_re = real_val;
        auto l_im = imag_val;
        auto r_re = rhs.real_val;
        auto r_im = rhs.imag_val;

        real_val = l_re + static_cast<T>(r_re);
        imag_val = l_im + static_cast<T>(r_im);

        return *this;
    }

    // add a real
    complex& operator+=(const T& real) restrict(amp, cpu)
    {
        real_val += real;
        return *this;
    }

    // subtract a complex
    template<typename X>
    complex& operator-=(const complex<X>& rhs) restrict(amp, cpu)
    {
        // Temporaries might help coalescing loads 
        auto l_re = real_val;
        auto l_im = imag_val;
        auto r_re = rhs.real_val;
        auto r_im = rhs.imag_val;

        real_val = l_re - static_cast<T>(r_re);
        imag_val = l_im - static_cast<T>(r_im);

        return *this;
    }

    // subtract a real
    complex& operator-=(const T& real) restrict(amp, cpu)
    {
        real_val -= real;
        return *this;
    }

    // multiply a complex
    template<typename X>
    complex& operator*=(const complex<X>& rhs) restrict(amp, cpu)
    {
        // Temporaries might help coalescing loads 
        auto l_re = real_val;
        auto l_im = imag_val;
        auto r_re = rhs.real_val;
        auto r_im = rhs.imag_val;

        real_val = l_re*static_cast<T>(r_re) - l_im*static_cast<T>(r_im);
        imag_val = l_re*static_cast<T>(r_im) + l_im*static_cast<T>(r_re);

        return *this;
    }

    // multiply a real
    complex& operator*=(const T& rhs) restrict(amp, cpu)
    {
        // Temporaries might help coalescing loads 
        auto re = real_val;
        auto im = imag_val;

        real_val = re*rhs;
        imag_val = im*rhs;

        return *this;
    }

    // divide a complex
    template<typename X>
    complex& operator/=(const complex<X>& rhs) restrict(amp, cpu)
    {
        // Temporaries might help coalescing loads 
        auto l_re = real_val;
        auto l_im = imag_val;
        auto r_re = rhs.real_val;
        auto r_im = rhs.imag_val;

        real_val = l_re/static_cast<T>(r_re) - l_im/static_cast<T>(r_im);
        imag_val = l_re/static_cast<T>(r_im) + l_im/static_cast<T>(r_re);

        return *this;
    }

    // divide a real
    complex& operator/=(const T& rhs) restrict(amp, cpu)
    {
        // Temporaries might help coalescing loads 
        auto re = real_val;
        auto im = imag_val;

        real_val = re/rhs;
        imag_val = im/rhs;

        return *this;
    }

    // equal 
    bool operator==(const complex& rhs) const restrict(amp, cpu)
    {
        return (real_val == rhs.real_val && imag_val == rhs.imag_val);
    }

    // not equal
    bool operator!=(const complex& rhs) const restrict(amp, cpu)
    {
        return !(operator==(rhs));
    }

    T real() const restrict(amp, cpu) 
    { 
        return real_val; 
    }

    void real(const T& real) restrict(amp, cpu) 
    { 
        real_val = real; 
    }

    T imag() const restrict(amp, cpu) 
    { 
        return imag_val; 
    }

    void imag(const T& imag) restrict(amp, cpu) 
    { 
        imag_val = imag; 
    }
};

// operator + 
// complex + complex
template<typename T>
inline complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs) restrict(amp, cpu) 
{
    // Temporaries might help coalescing loads 
    auto l_re = lhs.real();
    auto l_im = lhs.imag();
    auto r_re = rhs.real();
    auto r_im = rhs.imag();

    return complex<T>(l_re+r_re, l_im+r_im);
}
// complex + real
template<typename T>
inline complex<T> operator+(const complex<T>& lhs, const T& rhs) restrict(amp, cpu) 
{
    // Temporaries might help coalescing loads 
    auto re = lhs.real();
    auto im = lhs.imag();

    return complex<T>(re+rhs, im);
}
// real + complex
template<typename T>
inline complex<T> operator+( const T& lhs, const complex<T>& rhs) restrict(amp, cpu) 
{
    // Temporaries might help coalescing loads 
    auto re = rhs.real();
    auto im = rhs.imag();

    return complex<T>(re+lhs, im);
}

// operator - 
// complex - complex
template<typename T>
inline complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs) restrict(amp, cpu) 
{
    // Temporaries might help coalescing loads 
    auto l_re = lhs.real();
    auto l_im = lhs.imag();
    auto r_re = rhs.real();
    auto r_im = rhs.imag();

    return complex<T>(l_re-r_re, l_im-r_im);
}
// complex - real
template<typename T>
inline complex<T> operator-(const complex<T>& lhs, const T& rhs) restrict(amp, cpu) 
{
    // Temporaries might help coalescing loads 
    auto re = lhs.real();
    auto im = lhs.imag();

    return complex<T>(re-rhs, im);
}
// real - complex
template<typename T>
inline complex<T> operator-( const T& lhs, const complex<T>& rhs) restrict(amp, cpu) 
{
    // Temporaries might help coalescing loads 
    auto re = rhs.real();
    auto im = rhs.imag();

    return complex<T>(re-lhs, im);
}

// operator * 
// complex * complex
template<typename T>
inline complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs) restrict(amp, cpu) 
{
    // Temporaries might help coalescing loads 
    auto l_re = lhs.real();
    auto l_im = lhs.imag();
    auto r_re = rhs.real();
    auto r_im = rhs.imag();

    return complex<T>(l_re*r_re - l_im*r_im, l_re*r_im + l_im*r_re);
}
// complex * real
template<typename T>
inline complex<T> operator*(const complex<T>& lhs, const T& rhs) restrict(amp, cpu) 
{
    // Temporaries might help coalescing loads 
    auto re = lhs.real();
    auto im = lhs.imag();

    return complex<T>(re*rhs, im*rhs);
}
// real * complex
template<typename T>
inline complex<T> operator*( const T& lhs, const complex<T>& rhs) restrict(amp, cpu) 
{
    // Temporaries might help coalescing loads 
    auto re = rhs.real();
    auto im = rhs.imag();

    return complex<T>(re*lhs, im*lhs);
}

// operator / 
// complex / complex
template<typename T>
inline complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs) restrict(amp, cpu) 
{
    // Temporaries might help coalescing loads 
    auto l_re = lhs.real();
    auto l_im = lhs.imag();
    auto r_re = rhs.real();
    auto r_im = rhs.imag();

    return complex<T>(l_re/r_re - l_im/r_im, l_re/r_im + l_im/r_re);
}
// complex / real
template<typename T>
inline complex<T> operator/(const complex<T>& lhs, const T& rhs) restrict(amp, cpu) 
{
    // Temporaries might help coalescing loads 
    auto re = lhs.real();
    auto im = lhs.imag();

    return complex<T>(re/rhs, im/rhs);
}
// real / complex
template<typename T>
inline complex<T> operator/( const T& lhs, const complex<T>& rhs) restrict(amp, cpu) 
{
    // Temporaries might help coalescing loads 
    auto re = rhs.real();
    auto im = rhs.imag();

    return complex<T>(lhs/re, lhs/im);
}

// operator <<
template<typename T, typename charT, typename traits>
inline std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& bos, const complex<T>& rhs) 
{
    bos << "(" << rhs.real() << "," << rhs.imag() << ")";
    return bos;
}

} // ampblas

#endif // __cplusplus
#endif // AMPBLAS_COMPLEX_H
