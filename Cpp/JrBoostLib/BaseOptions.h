//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


// forest: forestSize
// tree: maxTreeDepth,
// samples: minAbsSampleWeight, minRelSampleWeight, usedSampleRatio, stratifiedSamples
// variables: topVariableCount, usedVariableRatio, selectVariablesByLevel
// nodes: minNodeSize, minNodeWeight, minNodeGain
// post-processing: pruneFactor
// other: saveMemory, test


class BaseOptions {   // POD class, so no need for virtual destructor
public:
    BaseOptions() = default;
    BaseOptions(const BaseOptions&) = default;
    BaseOptions& operator=(const BaseOptions&) = default;
    ~BaseOptions() = default;

    size_t forestSize() const { return forestSize_; }
    size_t maxTreeDepth() const { return maxTreeDepth_; }
    double minAbsSampleWeight() const { return minAbsSampleWeight_; }
    double minRelSampleWeight() const { return minRelSampleWeight_; }
    double usedSampleRatio() const { return usedSampleRatio_; }
    bool stratifiedSamples() const { return stratifiedSamples_; }
    size_t topVariableCount() const { return topVariableCount_; }
    double usedVariableRatio() const { return usedVariableRatio_; }
    bool selectVariablesByLevel() const { return selectVariablesByLevel_; }
    size_t minNodeSize() const { return minNodeSize_; }
    double minNodeWeight() const { return minNodeWeight_; }
    double minNodeGain() const { return minNodeGain_; }
    double pruneFactor() const { return pruneFactor_; }
    bool saveMemory() const { return saveMemory_; }
    size_t test() const { return test_; }

    void setForestSize(size_t n);
    void setMaxTreeDepth(size_t d);
    void setMinAbsSampleWeight(double w);
    void setMinRelSampleWeight(double w);
    void setUsedSampleRatio(double r);
    void setStratifiedSamples(bool b);
    void setTopVariableCount(size_t n);
    void setUsedVariableRatio(double r);
    void setSelectVariablesByLevel(bool b);
    void setMinNodeSize(size_t n);
    void setMinNodeWeight(double w);
    void setMinNodeGain(double g);
    void setPruneFactor(double p);
    void setSaveMemory(bool b);
    void setTest(size_t n);

private:
    size_t forestSize_{1};
    size_t maxTreeDepth_{1};
    double minAbsSampleWeight_{0.0};
    double minRelSampleWeight_{0.0};
    double usedSampleRatio_{1.0};
    bool stratifiedSamples_{true};
    size_t topVariableCount_{numeric_limits<size_t>::max()};
    double usedVariableRatio_{1.0};
    bool selectVariablesByLevel_{false};
    size_t minNodeSize_{1};
    double minNodeWeight_{0.0};
    double minNodeGain_{0.0};
    double pruneFactor_{0.0};
    bool saveMemory_{false};
    bool test_{0};
};
