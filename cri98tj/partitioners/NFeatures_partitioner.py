import pandas as pd
from sklearn.exceptions import *
from tqdm.auto import tqdm

from cri98tj.partitioners.PartitionerInterface import PartitionerInterface


class NFeatures_partitioner(PartitionerInterface):

    def _checkFormat(self, X):
        if X.shape[1] != 2 + len(self.spatioTemporalColumns):
            raise DataDimensionalityWarning("The input data must be in this form (tid, class)+ spatioTemporalColumns")
        # Altri controlli?

    def __init__(self, interval, spatioTemporalColumns=None, verbose=True):
        self.interval = interval
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
    tid, class, spatioTemporalColumns, partId
    """

    def transform(self, X):
        self._checkFormat(X)

        df = pd.DataFrame(X, columns=["tid", "class"] + self.spatioTemporalColumns)

        encodes = []

        if self.verbose: print(F"Encoding {X.shape[0]} points at interval of  {self.interval} observations", flush=True)

        tid_count=-1
        count = 0
        prec_tid= -1
        for i, row in enumerate(tqdm(X, disable=not self.verbose, position=0, leave=True)):
            if row[0] != prec_tid or count == self.interval:
                tid_count += 1
                count = 0
                prec_tid = row[0]
            encodes.append(F"{tid_count}")

            count += 1

        df["partId"] = encodes

        return df.values
