//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

// The code assumes that exceptions have been enabled for the stream.
// Hence there is no checking for end-of-file and stream errors.


// The following function Base128 encodes an integer and writes the resulting bytes to a file

inline void base128Save(ostream& os, size_t n)
{
    while (n >= 0x80) {
        uint8_t b = static_cast<uint8_t>(n);
        b &= 0x7f;
        b |= 0x80;
        os.put(b);
        n >>= 7;
    }
    uint8_t b = static_cast<uint8_t>(n);
    os.put(b);
}


// The following function reads a Base128 encoded integer from a file.

inline size_t base128Load(istream& is)
{
    size_t n = 0;

    for (size_t shift = 0; shift != 63; shift += 7) {
        const size_t m = is.get();
        n |= (m & 0x7f) << shift;
        if (m < 0x80)
            return n;
    }

    const size_t m = is.get();
    if (m > 0x01)
        throw std::overflow_error("Overflow when decoding Base128 data.");
    n |= m << 63;
    return n;
}


/*
inline void base128Test()
{
    bool pass = true;
    const int n = 1000000;
    stringstream ss;
    {
        size_t a = 1;
        for (int k = 0; k != n; ++k) {
            a = 3 * a + 1;
            size_t b = a >> (k % 64);
            base128Save(ss, b);
        }
    }
    {
        size_t a = 1;
        for (int k = 0; k != n; ++k) {
            a = 3 * a + 1;
            size_t b = a >> (k % 64);
            size_t c = base128Load(ss);
            pass = pass && (c == b);
        }
    }
    std::cout << (pass ? "Pass" : "Fail") << std::endl;
}
*/
