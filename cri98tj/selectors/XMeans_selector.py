import pandas as pd
from pyclustering.cluster.xmeans import xmeans, splitting_type
from tqdm.autonotebook import tqdm

from cri98tj.selectors.SelectorInterface import SelectorInterface


class XMeans_selector(SelectorInterface):

    def __init__(self, normalizer, verbose=True,
                 initial_centers=None, kmax=20, tolerance=0.001,
                 criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION, ccore=True):
        self._xmeans_instances = None

        self.normalizer = normalizer
        self.verbose = verbose

        self.initial_centers = initial_centers
        self.kmax = kmax
        self.tolerance = tolerance
        self.criterion = criterion
        self.ccore = ccore

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
        df_pivot = pd.DataFrame(self.normalizer.fit_transform(X)).rename(columns={0: "class"})

        if self.verbose: print("Extracting clusters", flush=True)
        centroids = {}
        self._xmeans_instances = []
        for classe in tqdm(df_pivot["class"].unique(), disable=not self.verbose, position=0, leave=True):
            X = df_pivot[df_pivot["class"] == classe][[x for x in df_pivot.columns if x != "class"]].values
            if self.verbose: print(F"Class {classe}", flush=True)
            clust = xmeans(X, initial_centers=self.initial_centers, kmax=self.kmax, tolerance=self.tolerance,
                           criterion=self.criterion, ccore=self.ccore)
            clust.process()
            self._xmeans_instances.append(clust)
            centroids[classe] = clust.get_centers()

        movelets = []
        for e in [x for x in centroids.values()]:
            for el in e:
                movelets.append(el)

        # return df_pivot[["class"] + [x for x in df_pivot.columns if x != "class"]].values, movelets
        return movelets
