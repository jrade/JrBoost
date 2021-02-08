#pragma once

class Clock {
public:
    uint64_t clockCycleCount() const { return clockCycleCount_; }
    uint64_t itemCount() const { return itemCount_; }
    uint64_t callCount() const { return callCount_; }

    void start(uint64_t clockCycleCount)
    {
        clockCycleCount_ -= clockCycleCount;
    }

    void stop(uint64_t clockCycleCount, uint64_t itemCount = 0)
    {
        clockCycleCount_ += clockCycleCount;
        itemCount_ += itemCount;
        ++callCount_;
    }

    void reset() { *this = Clock(); }

private:
    uint64_t clockCycleCount_ = 0;
    uint64_t itemCount_ = 0;
    uint64_t callCount_ = 0;
};

