// Copyright (C) 2021 Johan Rade <johan.rade@gmail.com>
//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

class BasePredictor;
class TreeOptions;


class TreeTrainerImplBase {
public:
    virtual ~TreeTrainerImplBase() = default;
    virtual unique_ptr<BasePredictor> train(CRefXd outData, CRefXd weights, const TreeOptions& options) const = 0;

protected:
    TreeTrainerImplBase() = default;

// deleted:
    TreeTrainerImplBase(const TreeTrainerImplBase&) = delete;
    TreeTrainerImplBase& operator=(const TreeTrainerImplBase&) = delete;
};
