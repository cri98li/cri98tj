from pyclustering.cluster.xmeans import splitting_type

import pandas as pd

from skltemplate import TrajectoryTransformer
from skltemplate.partitioners.Geohash_partitioner import Geohash_partitioner
from skltemplate.selectors.XMeans_selector import XMeans_selector
from skltemplate.distancers.Euclidean_distancer import Euclidean_distancer


class gxe_prebuild(TrajectoryTransformer):

    def __init__(self, precision=7, maxLen=.95, normalize=True, scale=True, fillna_value=0.0, optimize=True, verbose=True, initial_centers=None, kmax=20, tolerance=0.001,
                 criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION, ccore=True):
        self.verbose = verbose

        self._geohash_instance = Geohash_partitioner(precision=precision, scale=scale, verbose=verbose, normalize=normalize)
        self._xmeans_instance = XMeans_selector(maxLen=maxLen, fillna_value=fillna_value, verbose=verbose,
                                                initial_centers=initial_centers, kmax=kmax, tolerance=tolerance,
                                                criterion=criterion, ccore=ccore)
        self._euclidean_instance = Euclidean_distancer(verbose=verbose)

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        df_gh = pd.DataFrame(self._geohash_instance.fit_transform(X))

        movelets = self._xmeans_instance.fit_transform(df_gh.values)

        return self._euclidean_instance.fit_transform((df_gh[df_gh.columns[:-1]].values, movelets))