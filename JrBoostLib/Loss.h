#pragma once

inline double linLoss(CRefXs outData, CRefXd predData)
{
    return (
        outData.cast<double>() / (1.0 + predData.exp())
        + (1 - outData).cast<double>() / (1 + (-predData).exp())
    ).sum();
}
