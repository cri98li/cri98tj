from math import inf

import numpy as np
from sklearn.exceptions import DataDimensionalityWarning
import pandas as pd
from tqdm.autonotebook import tqdm
from sklearn.cluster import OPTICS
from cri98tj.selectors.selector_utils import dataframe_pivot

from cri98tj.selectors.SelectorInterface import SelectorInterface


class OPTICS_selector(SelectorInterface):
    def _checkFormat(self, X):
        if X.shape[1] != 6:
            raise DataDimensionalityWarning(
                "The input data must be in this form (tid, class, time, c1, c2, partitionId)")
        # Altri controlli?

    def __init__(self, maxLen=.95, fillna_value=None, n_jobs=None, verbose=True, min_samples=5, max_eps=inf,
                 metric='minkowski', p=2,
                 metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True,
                 min_cluster_size=None, algorithm='auto', leaf_size=30, memory=None):
        if maxLen <= 0:
            pass  # raise .....

        self.maxLen = maxLen
        self.fillna_value = fillna_value
        self.verbose = verbose
        self.n_jobs = n_jobs

        self._OPTICS_instances = {}

        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.memory = memory

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

        if self.verbose: print("Extracting clusters", flush=True)
        self._OPTICS_instances = {}
        movelets = []
        for classe in tqdm(df_pivot["class"].unique(), disable=not self.verbose, position=0, leave=True):
            df_X = df_pivot[df_pivot["class"] == classe][[x for x in df_pivot.columns if x != "class"]]
            X = df_X.values
            if self.verbose: print(F"Class {classe}", flush=True)
            clust = OPTICS(min_samples=self.min_samples, max_eps=self.max_eps, metric=self.metric,p=self.p,
                           metric_params=self.metric_params, cluster_method=self.cluster_method, eps=self.eps,
                           xi=self.xi, predecessor_correction=self.predecessor_correction,
                           min_cluster_size=self.min_cluster_size, algorithm=self.algorithm,leaf_size=self.leaf_size,
                           memory=self.memory, n_jobs=self.n_jobs)

            df_X["labels"] = clust.fit_predict(X)

            df_X.labels = df_X.labels[df_X.labels != -1]

            for row in df_X.groupby(by=["labels"]).mean().values:
                movelets.append(list(row))


            #centroids[classe] = clust.fit_predict()
            self._OPTICS_instances[classe] = clust


        #list of list
        return movelets
