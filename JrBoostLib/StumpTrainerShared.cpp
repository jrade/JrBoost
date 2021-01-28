#include "pch.h"
#include "StumpTrainerShared.h"
#include "StumpOptions.h"


StumpTrainerShared::StumpTrainerShared(CRefXXf inData, RefXs strata) :
    sampleCount_{ static_cast<size_t>(inData.rows()) },
    variableCount_{ static_cast<size_t>(inData.cols()) },
    sortedSamples_(sortSamples_(inData)),
    strata_{ strata },
    stratum0Count_{ (strata == 0).cast<size_t>().sum() },
    stratum1Count_{ (strata == 1).cast<size_t>().sum() }
{
}


vector<vector<SampleIndex>> StumpTrainerShared::sortSamples_(CRefXXf inData) const
{
    const size_t variableCount = inData.cols();

    vector<vector<SampleIndex>> sortedSamples(variableCount);
    vector<pair<float, size_t>> tmp{ sampleCount_ };
    for (size_t j = 0; j < variableCount; ++j) {
        for (size_t i = 0; i < sampleCount_; ++i)
            tmp[i] = { inData(i,j), i };
        std::sort(begin(tmp), end(tmp));
        sortedSamples[j].resize(sampleCount_);
        for (size_t i = 0; i < sampleCount_; ++i)
            sortedSamples[j][i] = static_cast<SampleIndex>(tmp[i].second);
    }

    return sortedSamples;
}


size_t StumpTrainerShared::initUsedSampleMask(vector<char>* usedSampleMask, const StumpOptions& options, RandomNumberEngine& rne) const
{
    size_t usedSampleCount;
    usedSampleMask->resize(sampleCount_);

    if (options.isStratified()) {

        array<size_t, 2> sampleCountByStratum{ stratum0Count_, stratum1Count_ };

        array<size_t, 2> usedSampleCountByStratum{
            std::max(
                static_cast<size_t>(1),
                static_cast<size_t>(static_cast<double>(options.usedSampleRatio()) * stratum0Count_ + 0.5)
            ),
            std::max(
                static_cast<size_t>(1),
                static_cast<size_t>(static_cast<double>(options.usedSampleRatio()) * stratum1Count_ + 0.5)
            )
        };

        usedSampleCount = usedSampleCountByStratum[0] + usedSampleCountByStratum[1];

        //cout << sampleCountByStratum[0] << " " << sampleCountByStratum[1] << " ";
        //cout << usedSampleCountByStratum[0] << " " << usedSampleCountByStratum[1] << endl;

        fastStratifiedRandomMask(
            &strata_(0),
            &strata_(0) + sampleCount_,
            begin(*usedSampleMask),
            begin(sampleCountByStratum),
            begin(usedSampleCountByStratum),
            rne
        );

        //ASSERT(sampleCountByStratum[0] == 0);
        //ASSERT(sampleCountByStratum[1] == 0);
        //ASSERT(usedSampleCountByStratum[0] == 0);
        //ASSERT(usedSampleCountByStratum[1] == 0);
    }

    else {
        usedSampleCount = std::max(
            static_cast<size_t>(1),
            static_cast<size_t>(static_cast<double>(options.usedSampleRatio()) * sampleCount_ + 0.5)
        );

        fastRandomMask(
            begin(*usedSampleMask),
            end(*usedSampleMask),
            usedSampleCount,
            rne
        );
    }

    return usedSampleCount;
}


void StumpTrainerShared::initSortedUsedSamples(vector<SampleIndex>* sortedUsedSamples, size_t usedSampleCount, const vector<char>& usedSampleMask, size_t j) const
{
    sortedUsedSamples->resize(usedSampleCount);

    fastCopyIf(
        cbegin(sortedSamples_[j]),
        begin(*sortedUsedSamples),
        end(*sortedUsedSamples),
        [&](size_t i) { return usedSampleMask[i]; }
    );
}