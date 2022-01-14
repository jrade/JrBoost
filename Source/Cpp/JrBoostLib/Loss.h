//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


double linLoss(CRefXu8 outData, CRefXd predData, optional<CRefXd> weights = std::nullopt);

class LogLoss {
public:
    LogLoss(double gamma);
    double operator()(CRefXu8 outData, CRefXd predData, optional<CRefXd> weights = std::nullopt) const;
    string name() const;

private:
    const double gamma_;
};

double auc(CRefXu8 outData, CRefXd predData, optional<CRefXd> weights = std::nullopt);
double aoc(CRefXu8 outData, CRefXd predData, optional<CRefXd> weights = std::nullopt);
double negAuc(CRefXu8 outData, CRefXd predData, optional<CRefXd> weights = std::nullopt);
