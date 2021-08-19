//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class TreeOptions {
public:   
    TreeOptions() = default;
    TreeOptions(const TreeOptions&) = default;
    TreeOptions& operator=(const TreeOptions&) = default;
    ~TreeOptions() = default;

    size_t maxDepth() const { return maxDepth_; }
    double usedSampleRatio() const { return usedSampleRatio_; }
    double usedVariableRatio() const { return usedVariableRatio_; }
    size_t topVariableCount() const { return topVariableCount_; }
    double minAbsSampleWeight() const { return minAbsSampleWeight_; }
    double minRelSampleWeight() const { return minRelSampleWeight_; }
    size_t minNodeSize() const { return minNodeSize_; }
    double minNodeWeight() const { return minNodeWeight_; }
    double minGain() const { return minGain_; }
    bool isStratified() const { return isStratified_; }
    double pruneFactor() const { return pruneFactor_; }
    bool altImplementation() const { return  altImplementation_; }

    void setMaxDepth(size_t d);
    void setUsedSampleRatio(double r);
    void setUsedVariableRatio(double r);
    void setTopVariableCount(size_t n);
    void setMinAbsSampleWeight(double w);
    void setMinRelSampleWeight(double w);
    void setMinNodeSize(size_t n);
    void setMinNodeWeight(double w);
    void setMinGain(double g);
    void setIsStratified(bool b);
    void setPruneFactor(double p);
    void setAltImplementation(bool b);

protected:
    double cost() const;

private:
    size_t maxDepth_{ 1 };
    double usedSampleRatio_{ 1.0 };
    double usedVariableRatio_{ 1.0 };
    size_t topVariableCount_{ numeric_limits<size_t>::max() };
    double minAbsSampleWeight_{ 0.0 };
    double minRelSampleWeight_{ 0.0 };
    size_t minNodeSize_{ 1 };
    double minNodeWeight_{ 0.0 };
    double minGain_{ 0.0 };
    bool isStratified_{ true };
    double pruneFactor_{ 0.0 };
    bool altImplementation_{ false };
};
