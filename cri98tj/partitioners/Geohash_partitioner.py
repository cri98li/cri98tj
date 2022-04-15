import pandas as pd
from geolib import geohash
from sklearn.exceptions import *
from tqdm.auto import tqdm

from cri98tj.partitioners.PartitionerInterface import PartitionerInterface


class Geohash_partitioner(PartitionerInterface):
    """
    precision : number, default=7
    A value used by the partitioner to set the partitions size
    """

    def _checkFormat(self, X):
        if X.shape[1] != 2 + len(self.spatioTemporalColumns):
            raise DataDimensionalityWarning("The input data must be in this form (tid, class)+ spatioTemporalColumns")
        # Altri controlli?

    def __init__(self, precision=7, spatioTemporalColumns=None, verbose=True):
        self.precision = precision
        self.verbose = verbose
        self.spatioTemporalColumns = spatioTemporalColumns

    """
    Controllo il formato dei dati, l'ordine deve essere: 
    tid, class, time, c1, c2
    """

    def fit(self, X):
        self._checkFormat(X)
        if "c1" != self.spatioTemporalColumns[0] or "c2" != self.spatioTemporalColumns[1]:
            raise DataDimensionalityWarning(
                "The spatioTemporalColumns attribute must have attribute c1 and c2 (lat and lon) as first attributes")

        return self

    """
    l'output sarà:
    tid, class, spatioTemporalColumns, geohash
    """

    def transform(self, X):
        self._checkFormat(X)

        df = pd.DataFrame(X, columns=["tid", "class"] + self.spatioTemporalColumns)

        encodes = []

        if self.verbose: print(F"Encoding {X.shape[0]} points with precision {self.precision}", flush=True)

        c=-1
        prec="|"
        prec_tid= -1
        for i, row in enumerate(tqdm(X, disable=not self.verbose, position=0, leave=True)):
            gh = geohash.encode(row[2], row[3], self.precision)
            if gh != prec or row[0] != prec_tid:
                prec_tid = row[0]
                prec = gh
                c+=1
            encodes.append(gh+"_"+str(c))

        df["geohash"] = encodes

        return df.values
