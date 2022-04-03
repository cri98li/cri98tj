from concurrent.futures import ProcessPoolExecutor
from math import inf

from sklearn.exceptions import DataDimensionalityWarning, NotFittedError
import pandas as pd
from tqdm.autonotebook import tqdm
from sklearn.cluster import OPTICS

from cri98tj.selectors.SelectorInterface import SelectorInterface
from cri98tj.selectors.selector_utils import dataframe_pivot, maxInformationGainScore


class RandomInformationGain_selector(SelectorInterface):
    def _checkFormat(self, X):
        if X.shape[1] != 6:
            raise DataDimensionalityWarning(
                "The input data must be in this form (tid, class, time, c1, c2, partitionId)")
        # Altri controlli?

    def __init__(self, top_k=10, movelets_per_class=100, trajectories_for_orderline=.10, maxLen=.95, fillna_value=None,
                 n_jobs=1, verbose=True):
        self.maxLen = maxLen
        self.fillna_value = fillna_value
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_movelets = movelets_per_class
        self.n_trajectories = trajectories_for_orderline
        self.top_k = top_k

        self._fitted = False

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
        if self.n_movelets is None:
            self.n_movelets = len(df_pivot)  # upper bound
        elif self.n_movelets < 1:
            self.n_movelets = round(self.n_movelets * len(df_pivot))
        df.partId = df.tid
        movelets_to_test = df_pivot.groupby('class', group_keys=False).apply(
            lambda x: x.sample(min(len(x), self.n_movelets))).drop(columns=["class"]).values

        df_pivot = dataframe_pivot(df=df, maxLen=self.maxLen, verbose=self.verbose, fillna_value=self.fillna_value)
        if self.n_trajectories is None:
            self.n_trajectories = len(df_pivot) # upper bound
        elif self.n_trajectories < 1:
            self.n_trajectories = round(self.n_trajectories * len(df_pivot))
        trajectories_for_orderline_df = df_pivot.groupby('class', group_keys=False).apply(
            lambda x: x.sample(min(len(x), self.n_trajectories)))
        trajectories_for_orderline = trajectories_for_orderline_df.drop(columns=["class"]).values
        y_trajectories_for_orderline = trajectories_for_orderline_df[["class"]].values

        scores = []

        if self.verbose: print(F"Computing scores")

        executor = ProcessPoolExecutor(max_workers=self.n_jobs)
        processes = []
        for movelet in tqdm(movelets_to_test, disable=not self.verbose, position=0, leave=True):
            processes.append(executor.submit(maxInformationGainScore, trajectories_for_orderline, movelet,
                                             y_trajectories_for_orderline, None))
            # scores.append(orderlineScore_leftPure(movelet=movelet, trajectories=trajectories_for_orderline,
            # y_trajectories=y_trajectories_for_orderline))

        for process in tqdm(processes):
            res = process.result()
            scores.append(res)

        movelets = []
        for i, (score, movelet) in enumerate(sorted(zip(scores, movelets_to_test), key=lambda x: x[0], reverse=True)):
            if i >= self.top_k: break

            if self.verbose: print(F"{i}.\t score={score}")

            movelets.append(movelet)

        # list of list
        return movelets
