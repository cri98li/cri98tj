from sklearn.exceptions import DataDimensionalityWarning
import pandas as pd
from tqdm.autonotebook import tqdm
from sklearn.cluster import OPTICS

from cri98tj.selectors.SelectorInterface import SelectorInterface
from cri98tj.selectors.selector_utils import dataframe_pivot


class Random_selector(SelectorInterface):
    def _checkFormat(self, X):
        if X.shape[1] != 6:
            raise DataDimensionalityWarning(
                "The input data must be in this form (tid, class, time, c1, c2, partitionId)")
        # Altri controlli?

    def __init__(self, movelets_per_class=10, maxLen=.95, fillna_value=None, n_jobs=None, verbose=True):

        self.maxLen = maxLen
        self.fillna_value = fillna_value
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_movelets = movelets_per_class

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
        self._checkFormat(X)

        return self

    """
    l'output sarà:
    tid, class, time, c1, c2, geohash
    """

    def transform(self, X):
        self._checkFormat(X)

        df = pd.DataFrame(X, columns=["tid", "class", "time", "c1", "c2", "partId"])
        df_pivot = dataframe_pivot(df=df, maxLen=self.maxLen, verbose=self.verbose, fillna_value=self.fillna_value)

        #df_movelets = df_pivot.groupby('class', group_keys=False).apply(lambda x: x.sample(min(len(x), self.n_movelets))).drop(columns=["occupied"]).values


        #list of list
        return df_pivot.groupby('class', group_keys=False)\
            .apply(lambda x: x.sample(min(len(x), self.n_movelets))).drop(columns=["class"]).values
