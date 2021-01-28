#pragma once

class StumpOptions;
class StumpTrainerShared;
class StumpPredictor;


class StumpTrainerByThread {
public:
    StumpTrainerByThread(CRefXXf inData, shared_ptr<const StumpTrainerShared> shared, std::random_device& rd);
    ~StumpTrainerByThread() = default;

    StumpPredictor train(CRefXd outData, CRefXd weights, const StumpOptions& options);

// deleted:
    StumpTrainerByThread() = delete;
    StumpTrainerByThread(const StumpTrainerByThread&) = delete;
    StumpTrainerByThread& operator=(const StumpTrainerByThread&) = delete;

private:
    void initUsedVariables_(const StumpOptions& opt);
    void initSums_(const CRefXd& outData, const CRefXd& weights);

    const CRefXXf inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const shared_ptr<const StumpTrainerShared> shared_;
    RandomNumberEngine rne_;

    vector<char> usedSampleMask_;
    vector<size_t> usedVariables_;
    vector<SampleIndex> sortedUsedSamples_;

    double sumW_;
    double sumWY_;
};
