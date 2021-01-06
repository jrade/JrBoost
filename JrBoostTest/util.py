import pandas as pd

def oneHotEncode(dataSeries):

    assert isinstance(dataSeries, pd.Series)

    labels = sorted(set(dataSeries))

    columns = pd.Index(labels, name = dataSeries.name)
    dataFrame = pd.DataFrame(index = dataSeries.index, columns = columns, data = 0)
    for sample in dataSeries.index:
        dataFrame.loc[sample, dataSeries[sample]] = 1

    return dataFrame
