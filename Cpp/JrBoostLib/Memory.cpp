//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"


void* operator new(size_t n)
{
    PROFILE::PUSH(PROFILE::MEMORY);
    void* p = malloc(n);
    PROFILE::POP(1);
    if (!p) throw std::bad_alloc();
    return p;
}


void operator delete(void* p) noexcept
{
    PROFILE::PUSH(PROFILE::MEMORY);
    free(p);
    PROFILE::POP(0);
}


void* operator new[](size_t n)
{
    PROFILE::PUSH(PROFILE::MEMORY);
    void* p = malloc(n);
    PROFILE::POP(1);
    if (!p) throw std::bad_alloc();
    return p;
}


void operator delete[](void* p) noexcept
{
    PROFILE::PUSH(PROFILE::MEMORY);
    free(p);
    PROFILE::POP(0);
}
