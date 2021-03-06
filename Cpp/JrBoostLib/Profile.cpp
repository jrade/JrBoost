//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "Profile.h"
#include "ClockCycleCount.h"


void PROFILE::PUSH(CLOCK_ID id)
{
    if (!ENABLED) return;
    if (omp_get_thread_num() != 0) return;

    uint64_t t = clockCycleCount();

    if (!clockIndexStack_.empty()) {
        CLOCK_ID prevId = clockIndexStack_.top();
        clocks_[prevId].stop(t);
    }

    clockIndexStack_.push(id);
    clocks_[id].start(t);
}

void PROFILE::POP(size_t itemCount)
{
    if (!ENABLED) return;
    if (omp_get_thread_num() != 0) return;

    uint64_t t = clockCycleCount();

    ASSERT(!clockIndexStack_.empty());
    CLOCK_ID id = clockIndexStack_.top();
    clocks_[id].stop(t, itemCount);
    clockIndexStack_.pop();

    if (!clockIndexStack_.empty()) {
        CLOCK_ID prevId = clockIndexStack_.top();
        clocks_[prevId].start(t);
    }
}

void PROFILE::SWITCH(size_t itemCount, CLOCK_ID id)
{
    if (!ENABLED) return;
    if (omp_get_thread_num() != 0) return;

    uint64_t t = clockCycleCount();

    ASSERT(!clockIndexStack_.empty());
    CLOCK_ID prevId = clockIndexStack_.top();
    clocks_[prevId].stop(t, itemCount);

    clockIndexStack_.top() = id;
    clocks_[id].start(t);
}

string PROFILE::RESULT()
{
    if (!ENABLED) return "";

    const Clock& zeroClock = clocks_[ZERO];
    double adjustment = static_cast<double>(zeroClock.clockCycleCount()) / zeroClock.callCount();

    uint64_t totalClockCycleCount = 0;
    uint64_t totalCallCount = 0;
    for (int id = 0; id < CLOCK_COUNT; ++id) {
        const Clock& clock = clocks_[id];
        totalClockCycleCount += clock.clockCycleCount();
        totalCallCount += clock.callCount();
    }
    double totalAdjustment = adjustment * totalCallCount;
    double totalAdjustedClockCycleCount = totalClockCycleCount - totalAdjustment;


    stringstream ss;

    for (int id = 0; id < CLOCK_COUNT; ++id) {

        if (id == ZERO) continue;

        const Clock& clock = clocks_[id];
        const string& name = names_[id];

        uint64_t clockCycleCount = clock.clockCycleCount();
        uint64_t callCount = clock.callCount();
        uint64_t itemCount = clock.itemCount();
        double adjustedClockCycleCount = clockCycleCount - adjustment * callCount;

        if (callCount != 0) {
            ss << std::setw(32) << std::left << name << std::right;
            ss << std::setw(8) << std::fixed << std::setprecision(0) << 1e-6 * adjustedClockCycleCount;
            ss << "  ";
            ss << std::setw(4) << std::setprecision(1) << 100.0 * adjustedClockCycleCount / totalAdjustedClockCycleCount << "%";
            if (itemCount != 0) {
                ss << "  ";
                ss << std::setw(5) << adjustedClockCycleCount / itemCount;
            }
            ss << '\n';
        }
    }
    ss << std::setw(40) << std::fixed << std::setprecision(0) << 1e-6 * totalAdjustedClockCycleCount << '\n';;

    ss << '\n';
    //ss << "zero calibration: " << adjustment << '\n';
    ss << "profiling overhead: "
        << (100.0 * totalAdjustment) / totalAdjustedClockCycleCount << "%" << '\n';
    ss << "slow branch: " << (100.0 * SLOW_BRANCH_COUNT) / SPLIT_ITERATION_COUNT << "%" << '\n';

    for (int id = 0; id < CLOCK_COUNT; ++id)
        clocks_[id].reset();
    SLOW_BRANCH_COUNT = 0;
    SPLIT_ITERATION_COUNT = 0;

    return ss.str();
}

//..............................................................................

uint64_t PROFILE::SPLIT_ITERATION_COUNT = 0;
uint64_t PROFILE::SLOW_BRANCH_COUNT = 0;

Clock PROFILE::clocks_[CLOCK_COUNT];
StaticStack<PROFILE::CLOCK_ID, 1000> PROFILE::clockIndexStack_;
