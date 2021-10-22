//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class AssertionError : public std::logic_error
{
public:
    AssertionError(const string& condition, const string& file, int line) : 
        logic_error(message_(condition, file, line))
    {}

private:
    static string message_(const string& condition, const string& file, int line)
    {
        stringstream ss;
        ss << "\nCondition: " << condition << "\nFile: " << file << "\nLine: " << line;
        return ss.str();
    }
};

#define ASSERT(A) if (!(A)) throw AssertionError(#A, __FILE__, __LINE__)
