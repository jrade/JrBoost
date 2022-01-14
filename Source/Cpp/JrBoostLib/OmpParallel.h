//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


inline std::atomic<bool> abortThreads = false;
// Set to true when an exception is thrown in a parallel region.
// Set to false when the parallel region is exited.

class ThreadAborted : public std::exception {
};
// The code in the parallel region should periodically check the flag abortThreads
// and throw ThreadAborted if the flag has been set.

// This code handles nested parallelism if all levels use the macros below.

// clang-format off

#define BEGIN_OMP_PARALLEL(THREAD_COUNT)                                                                               \
    {                                                                                                                  \
        const int threadCount__ = static_cast<int>(THREAD_COUNT);                                                      \
                                                                                                                       \
        const bool isProfiledThread__ = (std::this_thread::get_id() == PROFILE::CUR_THREAD_ID);                        \
        int nextProfiledThreadIndex__ = -1;                                                                            \
        if (isProfiledThread__) {                                                                                      \
            PROFILE::CUR_THREAD_ID = std::thread::id();                                                                \
            nextProfiledThreadIndex__ = std::uniform_int(0, threadCount__ - 1)(::theRne);                              \
        }                                                                                                              \
                                                                                                                       \
        std::exception_ptr ep__;                                                                                       \
        _Pragma("omp parallel num_threads(threadCount__)")                                                             \
        {                                                                                                              \
            ASSERT(omp_get_num_threads() == threadCount__);                                                            \
            if (omp_get_thread_num() == nextProfiledThreadIndex__)                                                     \
                PROFILE::CUR_THREAD_ID = std::this_thread::get_id();                                                   \
            try {


#define END_OMP_PARALLEL                                                                                               \
            }                                                                                                          \
            catch (const ThreadAborted&) {}                                                                            \
            catch (const std::exception&)                                                                              \
            {                                                                                                          \
                _Pragma("omp critical") if (!ep__)                                                                     \
                {                                                                                                      \
                    ep__ = std::current_exception();                                                                   \
                    abortThreads = true;                                                                               \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        if (isProfiledThread__)                                                                                        \
            PROFILE::CUR_THREAD_ID = std::this_thread::get_id();                                                       \
        if (ep__) {                                                                                                    \
            abortThreads = omp_in_parallel();                                                                          \
            std::rethrow_exception(ep__);                                                                              \
        }                                                                                                              \
        if (abortThreads)                                                                                              \
            throw ThreadAborted();                                                                                     \
    }

// clang-format on
