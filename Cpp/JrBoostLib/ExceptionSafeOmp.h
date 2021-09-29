//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


inline std::atomic<bool> abortThreads = false;
// Set to true when an exception is thrown in a parallel region.
// Set to false when the parallel region is exited.

class ThreadAborted : public std::exception {};
// The code in the parallel region should periodically check the flag abortThreads
// and throw ThreadAborted if the flag has been set.

// This code handles nested parallelism if all levels use the macros below.


#define BEGIN_EXCEPTION_SAFE_OMP_PARALLEL(THREAD_COUNT) \
    { \
        const int tc = THREAD_COUNT; \
        (void)tc; \
        std::exception_ptr ep; \
        _Pragma("omp parallel num_threads(tc)") \
        { \
            try {


#define END_EXCEPTION_SAFE_OMP_PARALLEL(A) \
            } \
            catch (const ThreadAborted&) {} \
            catch (const std::exception&) { \
                _Pragma("omp critical") \
                if (!ep) { \
                    ep = std::current_exception(); \
                    abortThreads = true; \
                } \
            } \
            PROFILE::PUSH(A); \
        } \
        PROFILE::POP(); \
        if (ep) { \
            abortThreads = omp_in_parallel(); \
            std::rethrow_exception(ep); \
        } \
        if (abortThreads) \
            throw ThreadAborted(); \
    }
