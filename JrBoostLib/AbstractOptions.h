#pragma once

class AbstractTrainer;

class AbstractOptions {
protected:
    AbstractOptions() = default;
    AbstractOptions(const AbstractOptions&) = default; // used by clone()

public:
    virtual ~AbstractOptions() = default;
    virtual AbstractOptions* clone() const = 0;
    virtual AbstractTrainer* createTrainer() const = 0;

// deleted:
    AbstractOptions& operator=(const AbstractOptions&) = delete;
    AbstractOptions(AbstractOptions&&) = delete;
    AbstractOptions& operator=(AbstractOptions&&) = delete;
};
