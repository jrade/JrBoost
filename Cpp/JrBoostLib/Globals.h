//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


inline const std::thread::id theMainThreadId = std::this_thread::get_id();

//----------------------------------------------------------------------------------------------------------------------

inline bool globParallelTree = true;
inline size_t globOuterThreadCount = 0;

//----------------------------------------------------------------------------------------------------------------------

using RandomNumberEngine = splitmix;

class InitializedRandomNumberEngine : public RandomNumberEngine
{
public:
    InitializedRandomNumberEngine() {
        std::random_device rd;
        seed(rd);
    }
};

inline thread_local InitializedRandomNumberEngine theRne;

//----------------------------------------------------------------------------------------------------------------------

class InterruptHandler
{
public:
    virtual void check() = 0;
};

inline InterruptHandler* currentInterruptHandler = nullptr;
