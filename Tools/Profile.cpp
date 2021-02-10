#include "pch.h"
#include "Profile.h"
#include "ClockCycleCount.h"


#ifdef DO_PROFILE

void PROFILE::PUSH(CLOCK_ID id)
{
#pragma omp master
    {
        if (++i_ % 100 == 0)
            push_(CLOCK_COUNT);
        push_(id);
    }
}

void PROFILE::POP(size_t itemCount)
{
#pragma omp master
    {
        pop_(itemCount);
        if (clockIndexStackPos_ != -1 && clockIndexStack_[clockIndexStackPos_] == CLOCK_COUNT)
            pop_(0);
    }
}

void PROFILE::SWITCH(size_t itemCount, CLOCK_ID id)
{
#pragma omp master
    {
        if (++i_ % 100 == 0) {
            switch_(itemCount, CLOCK_COUNT);
            switch_(0, id);
        }
        else
            switch_(itemCount, id);
    }
}

void PROFILE::PRINT()
{
    const Clock& zeroClock = clocks_[CLOCK_COUNT];
    size_t adjustment = zeroClock.clockCycleCount() / zeroClock.callCount();

    uint64_t totalClockCycleCount = 0;
    uint64_t totalCallCount = 0;
    for (int id = 0; id < CLOCK_COUNT; ++id) {
        const Clock& clock = clocks_[id];
        totalClockCycleCount += clock.clockCycleCount();
        totalCallCount += clock.callCount();
    }
    uint64_t totalAdjustment = adjustment * totalCallCount;
    uint64_t totalAdjustedClockCycleCount = totalClockCycleCount - totalAdjustment;

    for (int id = 0; id < CLOCK_COUNT; ++id) {

        const Clock& clock = clocks_[id];
        const string& name = names_[id];

        uint64_t clockCycleCount = clock.clockCycleCount();
        uint64_t callCount = clock.callCount();
        uint64_t itemCount = clock.itemCount();
        uint64_t adjustedClockCycleCount = clockCycleCount - adjustment * callCount;

        cout << std::setw(22) << std::left << name << std::right;
        cout << std::setw(8) << std::fixed << std::setprecision(0) << 1e-6 * adjustedClockCycleCount;
        cout << "  ";
        cout << std::setw(4) << std::setprecision(1) << (100.0 * adjustedClockCycleCount) / totalAdjustedClockCycleCount << "%";
        if (itemCount != 0) {
            cout << "  ";
            cout << std::setw(5) << static_cast<double>(clockCycleCount) / itemCount;
        }
        cout << endl;
    }

    cout << endl;
    cout << "zero calibration: " << adjustment << endl;
    cout << "profiling overhead: "
        << (100.0 * totalAdjustment) / totalAdjustedClockCycleCount << "%" << endl;
    cout << "slow branch: " << (100.0 * SLOW_BRANCH_COUNT) / SPLIT_ITERATION_COUNT << "%" << endl;

    for (int id = 0; id <= CLOCK_COUNT; ++id)
        clocks_[id].reset();
    SLOW_BRANCH_COUNT = 0;
    SPLIT_ITERATION_COUNT = 0;
}

#else

void PROFILE::PUSH(CLOCK_ID) {}
void PROFILE::POP(size_t) {}
void PROFILE::SWITCH(size_t itemCount, CLOCK_ID id) {}
void PROFILE::PRINT() {}

#endif

//..............................................................................

void PROFILE::push_(CLOCK_ID id)
{
    uint64_t t = clockCycleCount();

    if (clockIndexStackPos_ != -1) {
        CLOCK_ID prevId = clockIndexStack_[clockIndexStackPos_];
        clocks_[prevId].stop(t);
    }

    clockIndexStack_[++clockIndexStackPos_] = id;
    // this may call operator new() which may call PROFILE::PUSH() in an infinite loop
    // that is prevented by the guard check
    clocks_[id].start(t);
}

void PROFILE::pop_(size_t itemCount)
{
    uint64_t t = clockCycleCount();

    ASSERT(clockIndexStackPos_ != -1);
    CLOCK_ID id = clockIndexStack_[clockIndexStackPos_];
    clocks_[id].stop(t, itemCount);
    --clockIndexStackPos_;

    if (clockIndexStackPos_ != -1) {
        CLOCK_ID prevId = clockIndexStack_[clockIndexStackPos_];
        clocks_[prevId].start(t);
    }
}

void PROFILE::switch_(size_t itemCount, CLOCK_ID id)
{
    uint64_t t = clockCycleCount();

    ASSERT(clockIndexStackPos_ != -1);
    CLOCK_ID prevId = clockIndexStack_[clockIndexStackPos_];
    clocks_[prevId].stop(t, itemCount);

    clockIndexStack_[clockIndexStackPos_] = id;
    clocks_[id].start(t);
}

//..............................................................................

uint64_t PROFILE::SPLIT_ITERATION_COUNT = 0;
uint64_t PROFILE::SLOW_BRANCH_COUNT = 0;

Clock PROFILE::clocks_[CLOCK_COUNT + 1];
PROFILE::CLOCK_ID PROFILE::clockIndexStack_[1000];
int PROFILE::clockIndexStackPos_ = -1;
size_t PROFILE::i_ = 0;
