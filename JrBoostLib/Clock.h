//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class Clock {
public:
    Clock() = default;
    ~Clock() = default;
    Clock(const Clock&) = default;
    Clock& operator=(const Clock&) = default;

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
