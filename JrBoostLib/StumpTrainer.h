#pragma once

class AbstractPredictor;
class StumpOptions;
class StumpTrainerImplBase;

class StumpTrainer
{
public:
    StumpTrainer(CRefXXf inData, CRefXs strata);
    ~StumpTrainer() = default;

    unique_ptr<AbstractPredictor> train(CRefXd outData, CRefXd weights, const StumpOptions& options) const;

// deleted:
    StumpTrainer(const StumpTrainer&) = delete;
    StumpTrainer& operator=(const StumpTrainer&) = delete;

private:
    static shared_ptr<StumpTrainerImplBase> createImpl_(CRefXXf inData, CRefXs strata);

    const shared_ptr<const StumpTrainerImplBase> impl_;
};
