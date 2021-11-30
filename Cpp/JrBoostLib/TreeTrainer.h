//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

class BasePredictor;
class BaseOptions;


class TreeTrainer {   // abstract class
public:
    static unique_ptr<const TreeTrainer> createInstance(CRefXXfc inData, CRefXu8 strata);

    virtual ~TreeTrainer() = default;

    virtual unique_ptr<const BasePredictor>
    train(CRefXd outData, CRefXd weights, const BaseOptions& options, size_t threadCount) const;

protected:
    TreeTrainer() = default;
    TreeTrainer(const TreeTrainer&) = delete;
    TreeTrainer& operator=(const TreeTrainer&) = delete;

private:
    virtual unique_ptr<const BasePredictor>
    trainImpl0_(CRefXd outData, CRefXd weights, const BaseOptions& options, size_t threadCount) const = 0;
};
