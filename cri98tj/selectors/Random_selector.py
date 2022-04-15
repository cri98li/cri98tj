import pandas as pd

from cri98tj.selectors.SelectorInterface import SelectorInterface


class Random_selector(SelectorInterface):

    def __init__(self, normalizer, movelets_per_class=10, n_jobs=None, verbose=True):

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_movelets = movelets_per_class
        self.normalizer = normalizer

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
        df_pivot = self.normalizer.fit_transform(X)
        #df_pivot = dataframe_pivot(df=df, maxLen=self.maxLen, verbose=self.verbose, fillna_value=self.fillna_value, columns=self.spatioTemporalColumns)

        #df_movelets = df_pivot.groupby('class', group_keys=False).apply(lambda x: x.sample(min(len(x), self.n_movelets))).drop(columns=["occupied"]).values

        for cl in df_pivot["class"].unique():
            maxMov = len(df_pivot[df_pivot["class"] == cl])
            if self.verbose: print(f'Selecting {min(maxMov, self.n_movelets)} movelet over {maxMov} for class {cl}')

        #list of list
        return df_pivot.groupby('class', group_keys=False)\
            .apply(lambda x: x.sample(min(len(x), self.n_movelets))).drop(columns=["class"]).values
