#include "pch.h"
#include "StumpPredictor.h"


StumpPredictor::StumpPredictor(size_t variableCount, size_t j, float x, double leftY, double rightY) :
    AbstractPredictor{ variableCount },
    j_{ j },
    x_{ x },
    leftY_{ leftY },
    rightY_{ rightY }
{
    ASSERT(j < variableCount);
    ASSERT(std::isfinite(x) && std::isfinite(leftY) && std::isfinite(rightY));
}

ArrayXd StumpPredictor::predict(CRefXXf inData) const
{
    validateInData_(inData);
    size_t sampleCount = inData.rows();
    ArrayXd outData{ sampleCount };
    for (size_t i = 0; i < sampleCount; ++i)
        outData(i) = (inData(i, j_) < x_) ? leftY_ : rightY_;
    return outData;
}
