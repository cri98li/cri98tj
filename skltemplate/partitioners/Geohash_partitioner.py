import Geohash.geohash
from geolib import geohash
import numpy as np

from skltemplate.partitioners.PartitionerInterface import PartitionerInterface
from sklearn.exceptions import *


class Geohash_partitioner(PartitionerInterface):

    """
    precision : number, default=7
    A value used by the partitioner to set the partitions size

    maxLen : number, default=.95
    A value used to limit segments with too many points from the dataset.
    A value < 1 limits the length of the sub-trajectories at the maxLen quantile of the lengths
    A value >= 1 limits the length of the sub-trajectories at maxLen points

    """
    def _checkFormat(self, X):
        if X.shape[1] != 5:
            raise DataDimensionalityWarning("The input data must be in this form (tid, class, time, c1, c2)")
        # Altri controlli?



    def __init__(self, precision=7, maxLen=.95):
        self.precision = precision
        self.maxLen = maxLen

        self._tid=0
        self._class=1
        self._time=2
        self._lat=3
        self._lon=4

    """
    Controllo il formato dei dati, l'ordine deve essere: 
    tid, class, time, c1, c2
    """

    def fit(self, X, y=None):
        self._checkFormat(X)

    """
    l'output sarà:
    tid, class, time, c1, c2, encode
    """

    def transform(self, X, normalize=True):
        self._checkFormat(X)
        encodes = np.ones(X.shape[0])

        for i, row in enumerate(X):
            encodes[i] = geohash.encode(row[self._lat], row[self._lon], self.precision)

        if not normalize:
            return np.c_(X, encodes)

        dizionario = dict()

        for gh in encodes:
            if gh not in encodes:
                dizionario[gh] = geohash.bounds(gh).sw

        for i, row in enumerate(X):
            bounds = dizionario[encodes[i]]
            X[i][self._lat] -= bounds.lat
            X[i][self._lon] -= bounds.lon

        return np.c_(X, encodes)

