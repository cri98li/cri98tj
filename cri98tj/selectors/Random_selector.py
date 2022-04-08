import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning

from cri98tj.selectors.SelectorInterface import SelectorInterface
from cri98tj.selectors.selector_utils import dataframe_pivot


class Random_selector(SelectorInterface):

    def __init__(self, movelets_per_class=10, maxLen=.95, spatioTemporalColumns=["c1", "c2"], fillna_value=None, n_jobs=None, verbose=True):

        self.maxLen = maxLen
        self.fillna_value = fillna_value
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_movelets = movelets_per_class
        self.spatioTemporalColumns = spatioTemporalColumns

        self._tid = 0
        self._class = 1
        self._time = 2
        self._lat = 3
        self._lon = 4
        self._partitionId = 5

    """
    Controllo il formato dei dati, l'ordine deve essere: 
    tid, class, time, c1, c2
    """

    def fit(self, X):

        return self

    """
    l'output sarà:
    tid, class, time, c1, c2, geohash
    """

    def transform(self, X):

        #df = pd.DataFrame(X, columns=["tid", "class"]+self.spatioTemporalColumns+["partId"])
        df_pivot = pd.DataFrame(X).rename(columns={0: "class"})
        #df_pivot = dataframe_pivot(df=df, maxLen=self.maxLen, verbose=self.verbose, fillna_value=self.fillna_value, columns=self.spatioTemporalColumns)

        #df_movelets = df_pivot.groupby('class', group_keys=False).apply(lambda x: x.sample(min(len(x), self.n_movelets))).drop(columns=["occupied"]).values


        #list of list
        return df_pivot.groupby('class', group_keys=False)\
            .apply(lambda x: x.sample(min(len(x), self.n_movelets))).drop(columns=["class"]).values
