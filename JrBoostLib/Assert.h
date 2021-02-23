#pragma once

// Note: profiling shows that the asserts in the code do not reduce performance 


class AssertionError : public runtime_error
{
public:
    AssertionError(string condition, string file, int line) : 
        std::runtime_error(format(condition, file, line))
    {
    }

private:
    static string format(string condition, string file, int line)
    {
        std::stringstream ss;
        ss << "\nCondition: " << condition << "\nFile: " << file << "\nLine: " << line;
        return ss.str();
    }
};




#define ASSERT(A) if (!(A)) throw AssertionError(#A, __FILE__, __LINE__)

//#define ASSERT(A) assert(A)

//#define ASSERT(A)