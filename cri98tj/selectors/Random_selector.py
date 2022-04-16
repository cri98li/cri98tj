import random

import pandas as pd

from cri98tj.selectors.SelectorInterface import SelectorInterface
from cri98tj.selectors.selector_utils import select_and_pivot


class Random_selector(SelectorInterface):

    def __init__(self, normalizer, movelets_per_class=10, spatioTemporalColumns=[], n_jobs=None, verbose=True):

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_movelets = movelets_per_class
        self.normalizer = normalizer
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

        df = pd.DataFrame(X, columns=["tid", "class"]+self.spatioTemporalColumns+["partId"])

        df_pivot = select_and_pivot(df, self.n_movelets, self.normalizer)

        return df_pivot.values[:, 1:]
