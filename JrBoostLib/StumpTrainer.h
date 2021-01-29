#pragma once

class StumpOptions;
class StumpTrainerShared;
class StumpTrainerByThread;
class AbstractPredictor;


class StumpTrainer {
public:
    StumpTrainer(CRefXXf inData, RefXs strata);
    ~StumpTrainer() = default;
    unique_ptr<AbstractPredictor> train(CRefXd outData, CRefXd weights, const StumpOptions& options) const;

// deleted:
    StumpTrainer() = delete;
    StumpTrainer(const StumpTrainer&) = delete;
    StumpTrainer& operator=(const StumpTrainer&) = delete;

private:
    const size_t sampleCount_;
    shared_ptr<const StumpTrainerShared> shared_;
    vector<shared_ptr<StumpTrainerByThread>> byThread_;
};
