//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


ArrayXXfc selectRows(CRefXXfc inData, const vector<size_t>& samples);
ArrayXXfr selectRows(CRefXXfr inData, const vector<size_t>& samples);
ArrayXXdc selectRows(CRefXXdc inData, const vector<size_t>& samples);
ArrayXXdr selectRows(CRefXXdr inData, const vector<size_t>& samples);

ArrayXXfc selectColumns(CRefXXfc inData, const vector<size_t>& variables);
ArrayXXfr selectColumns(CRefXXfr inData, const vector<size_t>& variables);
ArrayXXdc selectColumns(CRefXXdc inData, const vector<size_t>& variables);
ArrayXXdr selectColumns(CRefXXdr inData, const vector<size_t>& variables);

ArrayXXfc select(CRefXXfc inData, const vector<size_t>& samples, const vector<size_t>& variables);
ArrayXXfr select(CRefXXfr inData, const vector<size_t>& samples, const vector<size_t>& variables);
ArrayXXdc select(CRefXXdc inData, const vector<size_t>& samples, const vector<size_t>& variables);
ArrayXXdr select(CRefXXdr inData, const vector<size_t>& samples, const vector<size_t>& variables);
