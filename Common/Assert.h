#pragma once

class AssertionError : public std::runtime_error
{
public:
    AssertionError(string condition, string file, int line) : std::runtime_error(format(condition, file, line))
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

#ifdef NDEBUG

#define JRASSERT(A) if (!(A)) throw AssertionError(#A, __FILE__, __LINE__)

#else

#define JRASSERT(A) assert(A)

#endif
