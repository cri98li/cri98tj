import pandas as pd
from geolib import geohash
import numpy as np

from skltemplate.partitioners.PartitionerInterface import PartitionerInterface
from sklearn.exceptions import *
from tqdm.autonotebook import tqdm


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



    def __init__(self, precision=7, maxLen=.95, verbose=True):
        self.precision = precision
        self.maxLen = maxLen
        self.verbose = verbose

        self._tid=0
        self._class=1
        self._time=2
        self._lat=3
        self._lon=4

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

    def transform(self, X, normalize=True):
        self._checkFormat(X)

        df = pd.DataFrame(X, columns=["tid", "class", "time", "c1", "c2"])

        encodes = []

        if self.verbose: print(F"Encoding {X.shape[0]} points with precision {self.precision}")
        for i, row in enumerate(tqdm(X, disable=not self.verbose)):
            encodes.append(geohash.encode(row[self._lat], row[self._lon], self.precision))

        df["geohash"] = encodes

        if not normalize:
            return df.values

        decodes = dict()

        if self.verbose: print(F"Retrieving partition boundaries")
        for gh in tqdm(encodes, disable=not self.verbose):
            if gh not in decodes:
                decodes[gh] = geohash.bounds(gh)

        if self.verbose: print(F"Normalizing the sub-trajectories")
        df.c1 = df.c1 - df.geohash.apply(lambda x: decodes[x].sw.lat)
        df.c2 = df.c2 - df.geohash.apply(lambda x: decodes[x].sw.lon)

        return df.values

