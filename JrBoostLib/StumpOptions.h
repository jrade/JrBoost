#pragma once


class StumpOptions {
public:   
    StumpOptions() = default;
    StumpOptions(const StumpOptions&) = default;
    StumpOptions& operator=(const StumpOptions&) = default;
    ~StumpOptions() = default;

    double usedSampleRatio() const { return usedSampleRatio_; }
    double usedVariableRatio() const { return usedVariableRatio_; }
    optional<size_t> topVariableCount() const { return topVariableCount_; }
    size_t minNodeSize() const { return minNodeSize_; }
    double minNodeWeight() const { return minNodeWeight_; }
    bool isStratified() const { return isStratified_; }
    bool profile() const { return profile_; }

    void setUsedSampleRatio(double r);
    void setUsedVariableRatio(double r);
    void setTopVariableCount(optional<size_t> n);
    void setMinNodeSize(size_t n);
    void setMinNodeWeight(double w);
    void setIsStratified(bool b);
    void setProfile(bool p);

private:
    double usedSampleRatio_{ 1.0 };
    double usedVariableRatio_{ 1.0 };
    optional<size_t> topVariableCount_;
    size_t minNodeSize_{ 1 };
    double minNodeWeight_{ 0.0 };
    bool isStratified_{ true };
    bool profile_{ false };
};
