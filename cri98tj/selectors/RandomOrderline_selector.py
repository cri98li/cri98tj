from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning
from tqdm.autonotebook import tqdm

from cri98tj.distancers.distancer_utils import euclideanBestFitting
from cri98tj.normalizers.normalizer_utils import dataframe_pivot2
from cri98tj.selectors.SelectorInterface import SelectorInterface
from cri98tj.selectors.selector_utils import orderlineScore_leftPure


class RandomOrderline_selector(SelectorInterface):
    def _checkFormat(self, X):
        if X.shape[1] != 6:
            raise DataDimensionalityWarning(
                "The input data must be in this form (tid, class, time, c1, c2, partitionId)")
        # Altri controlli?

    def __init__(self, normalizer, spatioTemporalColumns=[], distanceMeasure=euclideanBestFitting, top_k=10, movelets_per_class=100, trajectories_for_orderline=.10, n_jobs=1, verbose=True):
        self.normalizer = normalizer
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_movelets = movelets_per_class
        self.n_trajectories = trajectories_for_orderline
        self.spatioTemporalColumns = spatioTemporalColumns
        self.top_k = top_k
        self.distanceMeasure = distanceMeasure

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

        return self

    """
    l'output sarà:
    tid, class, time, c1, c2, geohash
    """

    def transform(self, X):
        df = pd.DataFrame(X, columns=["tid", "class"]+self.spatioTemporalColumns+["partId"])
        df_pivot = self.normalizer.fit_transform(X)

        if self.n_movelets is None:
            self.n_movelets = len(df_pivot)
        elif self.n_movelets < 1:
            self.n_movelets = round(self.n_movelets*len(df_pivot))
        movelets_to_test = df_pivot.groupby('class', group_keys=False)\
            .apply(lambda x: x.sample(min(len(x), self.n_movelets))).drop(columns=["tid", "class", "partId"])

        df.partId = df.tid
        df_pivot = dataframe_pivot2(df, maxLen=None, verbose=self.verbose, fillna_value=None, columns=self.spatioTemporalColumns)

        if self.n_trajectories is None:
            self.n_trajectories = len(df_pivot)
        elif self.n_trajectories < 1:
            self.n_trajectories = round(self.n_trajectories * len(df_pivot))
        trajectories_for_orderline_df = df_pivot.groupby('class', group_keys=False).apply(
            lambda x: x.sample(min(len(x), self.n_trajectories)))

        scores = []

        if self.verbose: print(F"Computing scores")

        executor = ProcessPoolExecutor(max_workers=self.n_jobs)
        processes = []
        for movelet in tqdm(movelets_to_test.values, disable=not self.verbose, position=0, leave=True):
            processes.append(executor.submit(orderlineScore_leftPure, trajectories_for_orderline_df, movelet, None,
                                             self.spatioTemporalColumns, ["tid", "class", "partId"], self.normalizer,
                                             self.distanceMeasure))

        for process in tqdm(processes):
            scores.append(process.result())


        movelets = []
        for i, (score, movelet) in enumerate(sorted(zip(scores, movelets_to_test), key=lambda x: x[0], reverse=True)):
            if i >= self.top_k: break

            if self.verbose: print(F"{i}.\t score={score}")

            movelets.append(movelet)

        #list of list
        return movelets
