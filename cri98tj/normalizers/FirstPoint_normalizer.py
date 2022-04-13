import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning

from cri98tj.normalizers.NormalizerInterface import NormalizerInterface
from cri98tj.normalizers.normalizer_utils import dataframe_pivot2


class FirstPoint_normalizer(NormalizerInterface):
    def _checkFormat(self, X):
        if X.shape[1] != 3 + len(self.spatioTemporalColumns):
            raise DataDimensionalityWarning(
                "The input data must be in this form [tid, class]+spatioTemporalColumns+[partId]")
        # Altri controlli?

    def __init__(self, spatioTemporalColumns=None, maxLen=.95, fillna=None, verbose=True):
        self.spatioTemporalColumns = spatioTemporalColumns
        self.verbose = verbose
        self.fillna = fillna
        self.maxLen = maxLen

    def fit(self, X):
        self._checkFormat(X)

        return self

    """
    l'input sarà:
    tid, class, spatioTemporalColumns, partId
    """

    def transform(self, X):
        df = pd.DataFrame(X, columns=["tid", "class"] + self.spatioTemporalColumns + ["partId"])
        df_pivot = dataframe_pivot2(df, self.maxLen, self.verbose, self.fillna, self.spatioTemporalColumns)

        array_pivot = df_pivot.values

        offset = len(df.columns)-len(self.spatioTemporalColumns)

        for row in array_pivot:
            start = None
            for i in range(offset, len(row)): #Escludo le colonne contenenti tid, class e partId
                if (i - offset) % ((len(row) - offset) / len(self.spatioTemporalColumns)) == 0:
                    start = row[i]

                if row[i] is not None:
                    row[i] -= start
        return pd.DataFrame(array_pivot, columns=df_pivot.columns)

    def _transformSingleTraj(self, X):  # X è una lista di numeri semplici
        row = X.copy()

        return list(map(lambda x: x - row[0], row))
