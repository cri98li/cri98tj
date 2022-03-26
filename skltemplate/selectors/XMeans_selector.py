from sklearn.exceptions import DataDimensionalityWarning
import pandas as pd
from tqdm.autonotebook import tqdm
from pyclustering.cluster.xmeans import xmeans, splitting_type

from skltemplate.partitioners.PartitionerInterface import PartitionerInterface


class XMeans_selector(PartitionerInterface):
    def _checkFormat(self, X):
        if X.shape[1] != 6:
            raise DataDimensionalityWarning(
                "The input data must be in this form (tid, class, time, c1, c2, partitionId)")
        # Altri controlli?

    def __init__(self, maxLen=.95, fillna_value=0.0, verbose=True, initial_centers=None, kmax=20, tolerance=0.001,
                 criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION, ccore=True):
        self._xmeans_instances = None
        if maxLen <= 0:
            pass  # raise .....

        self.maxLen = maxLen
        self.fillna_value = fillna_value
        self.verbose = verbose

        self.initial_centers = initial_centers
        self.kmax = kmax
        self.tolerance = tolerance
        self.criterion = criterion
        self.ccore = ccore

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

        df["pos"] = df.groupby(['tid', 'partId']).cumcount()

        if self.maxLen is not None:
            if self.maxLen >= 1:
                if self.verbose: print(F"Cutting sub-trajectories length at {self.maxLen}")
                df = df[df.pos < self.maxLen]
            else:
                if self.verbose: print(F"Cutting sub-trajectories length at {df.quantile(.95).pos}")
                df = df[df.pos < df.quantile(.95).pos]

        if self.verbose: print("Pivoting tables")
        df_pivot = df.groupby(['tid', 'pos'])[['c1', 'c2']].max().unstack().reset_index()
        df_pivot = df_pivot.merge(df.groupby(['tid'])['class'].max().reset_index(), on=["tid"])
        df_pivot = df_pivot.drop(columns=[("tid", "")]).set_index("tid")

        df_pivot.fillna(self.fillna_value, inplace=True)

        if self.verbose: print("Extracting clusters")
        centroids = {}
        self._xmeans_instances = []
        for classe in tqdm(df_pivot["class"].unique(), disable=not self.verbose):
            X = df_pivot[df_pivot["class"] == classe][[x for x in df_pivot.columns if x != "class"]].values
            if self.verbose: print(F"Class {classe}")
            clust = xmeans(X, initial_centers=self.initial_centers, kmax=self.kmax, tolerance=self.tolerance,
                           criterion=self.criterion, ccore=self.ccore)
            clust.process()
            self._xmeans_instances.append(clust)
            centroids[classe] = clust.get_centers()

        movelets = []
        for e in [x for x in centroids.values()]:
            for el in e:
                movelets.append(el)

        #return df_pivot[["class"] + [x for x in df_pivot.columns if x != "class"]].values, movelets
        return movelets
