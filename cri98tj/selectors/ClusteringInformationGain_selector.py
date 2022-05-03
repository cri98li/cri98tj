from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.feature_selection import mutual_info_classif
from tqdm.autonotebook import tqdm

from cri98tj.distancers.Euclidean_distancer import euclideanBestFitting
from cri98tj.normalizers.normalizer_utils import dataframe_pivot
from cri98tj.selectors.SelectorInterface import SelectorInterface
from cri98tj.selectors.selector_utils import select_and_pivot


class ClusteringInformationGain_selector(SelectorInterface):
    def _checkFormat(self, X):
        if X.shape[1] != 3+len(self.spatioTemporalColumns):
            raise DataDimensionalityWarning(
                "The input data must be in this form [tid, class] + spatioTemporalColumns + [partitionId]")
        # Altri controlli?

    def __init__(self, normalizer, bestFittingMeasure, top_k=10, n_medoids=100, movelets_for_clustering=None, trajectories_for_InfoGain=.10, maxLen=None,
                 spatioTemporalColumns=["c1", "c2"], fillna_value=None, n_neighbors=3, n_jobs=1, random_state=None,
                 verbose=True):
        self.maxLen = maxLen
        self.fillna_value = fillna_value
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_movelets = movelets_for_clustering
        self.n_medoids = n_medoids
        self.n_trajectories = trajectories_for_InfoGain
        self.top_k = top_k
        self.spatioTemporalColumns = spatioTemporalColumns
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.normalizer = normalizer
        self.bestFittingMeasure = bestFittingMeasure

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

        df = pd.DataFrame(X, columns=["tid", "class"] + self.spatioTemporalColumns + ["partId"])
        n_class = len(df["class"].unique())

        if self.n_movelets is None:
            self.n_movelets = len(df.partId.unique())  # upper bound
        elif self.n_movelets < 1:
            self.n_movelets = round(self.n_movelets * len(df.partId.unique()))


        movelets_to_test = select_and_pivot(df, self.n_movelets/n_class, self.normalizer).values[:, 1:]

        df.partId = df.tid

        if self.n_trajectories is None:
            self.n_trajectories = len(df.partId.unique())  # upper bound
        elif self.n_trajectories < 1:
            self.n_trajectories = round(self.n_trajectories * len(df.partId.unique()))
        trajectories_for_orderline_df = select_and_pivot(df, self.n_trajectories/n_class, self.normalizer)
        trajectories_for_orderline = trajectories_for_orderline_df.drop(columns=["class"]).values
        y_trajectories_for_orderline = trajectories_for_orderline_df[["class"]].values

        scores = []

        if self.verbose: print(F"Computing scores")


        dist_matrix = np.zeros((len(trajectories_for_orderline), len(movelets_to_test)))

        executor = ProcessPoolExecutor(max_workers=self.n_jobs)
        processes = []
        for movelet in tqdm(movelets_to_test, disable=not self.verbose, position=0, leave=True):
            processes.append(executor.submit(self._computeDist, trajectories_for_orderline, movelet))

        for i, process in enumerate(tqdm(processes)):
            res = process.result()
            for j, el in enumerate(res):
                dist_matrix[j][i] = el

        movelets = []
        for i, (score, movelet) in enumerate(sorted(zip(scores, movelets_to_test), key=lambda x: x[0], reverse=True)):
            if i >= self.top_k: break

            if self.verbose: print(F"{i}.\t score={score}")

        mutualInfo = mutual_info_classif(dist_matrix, y_trajectories_for_orderline, n_neighbors=self.n_neighbors,
                                         random_state=self.random_state)

        for i, (score, mov) in enumerate(sorted(zip(mutualInfo, movelets_to_test), key=lambda x: x[0], reverse=True)):
            if i >= self.top_k: break
            movelets.append(mov)
            if self.verbose: print(F"{i}.\t score={score}")

        return movelets  # list of list

    def _computeDist(self, trajectories, movelet):
        distances = []
        for i, trajectory in enumerate(trajectories):
            tmp, distance = self.bestFittingMeasure(trajectory=trajectory, movelet=movelet,
                                                 spatioTemporalColumns=self.spatioTemporalColumns,
                                                 normalizer=self.normalizer)
            distances.append(distance)
        return distances
