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
