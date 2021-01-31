#include "pch.h"


void* operator new(size_t n)
{
    CLOCK::PUSH(CLOCK::MEMORY);
    void* p = malloc(n);
    CLOCK::POP(1);
    if (!p) throw std::bad_alloc();
    return p;
}


void operator delete(void* p) noexcept
{
    CLOCK::PUSH(CLOCK::MEMORY);
    free(p);
    CLOCK::POP(0);
}
