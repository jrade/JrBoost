#pragma once

inline double linLoss(RefXs outData, RefXd predData)
{
    return (
        outData.cast<double>() / (1.0 + predData.exp())
        + (1 - outData).cast<double>() / (1 + (-predData).exp())
    ).sum();
}
