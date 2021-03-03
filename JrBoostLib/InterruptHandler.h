#pragma once


class InterruptHandler {
public:
    virtual void check() = 0;
};


inline InterruptHandler* currentInterruptHandler = nullptr;
