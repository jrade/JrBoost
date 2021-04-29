//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class StumpOptions {
public:   
    StumpOptions() = default;
    StumpOptions(const StumpOptions&) = default;
    StumpOptions& operator=(const StumpOptions&) = default;
    ~StumpOptions() = default;

    double usedSampleRatio() const { return usedSampleRatio_; }
    double usedVariableRatio() const { return usedVariableRatio_; }
    size_t topVariableCount() const { return topVariableCount_; }
    double minAbsSampleWeight() const { return minAbsSampleWeight_; }
    double minRelSampleWeight() const { return minRelSampleWeight_; }
    size_t minNodeSize() const { return minNodeSize_; }
    double minNodeWeight() const { return minNodeWeight_; }
    bool isStratified() const { return isStratified_; }

    void setUsedSampleRatio(double r);
    void setUsedVariableRatio(double r);
    void setTopVariableCount(size_t n);
    void setMinAbsSampleWeight(double w);
    void setMinRelSampleWeight(double w);
    void setMinNodeSize(size_t n);
    void setMinNodeWeight(double w);
    void setIsStratified(bool b);

protected:
    double cost() const;

private:
    double usedSampleRatio_{ 1.0 };
    double usedVariableRatio_{ 1.0 };
    size_t topVariableCount_{ numeric_limits<size_t>::max() };
    double minAbsSampleWeight_{ 0.0 };
    double minRelSampleWeight_{ 0.0 };
    size_t minNodeSize_{ 1 };
    double minNodeWeight_{ 0.0 };
    bool isStratified_{ true };
};
