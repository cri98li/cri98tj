import pandas as pd
from geolib import geohash
from sklearn.exceptions import DataDimensionalityWarning

from cri98tj.normalizers.NormalizerInterface import NormalizerInterface
from cri98tj.normalizers.normalizer_utils import dataframe_pivot


class GeohashPartition_normalizer(NormalizerInterface):
    def _checkFormat(self, X):
        if X.shape[1] != 3 + len(self.spatioTemporalColumns):
            raise DataDimensionalityWarning(
                "The input data must be in this form [tid, class]+spatioTemporalColumns+[partId]")
        # Altri controlli?

    def __init__(self, precision, spatioTemporalColumns=None, maxLen=.95, fillna=None, verbose=True):
        self.spatioTemporalColumns = spatioTemporalColumns
        self.verbose = verbose
        self.fillna = fillna
        self.maxLen = maxLen
        self.precision = precision

    def fit(self, X):
        self._checkFormat(X)

        return self

    """
    l'input sarà:
    tid, class, spatioTemporalColumns, partId
    """

    def transform(self, X):
        df = pd.DataFrame(X, columns=["tid", "class"] + self.spatioTemporalColumns + ["partId"])

        for gh in df.partId.unique():
            decoded_gh = geohash.bounds(gh).sw
            df[df.partId == gh].c1 = decoded_gh.lat
            df[df.partId == gh].c2 = decoded_gh.lon

        return dataframe_pivot(df, self.maxLen, self.verbose, self.fillna, self.spatioTemporalColumns)

    def _transformSingleTraj(self, X):  # X è una lista di numeri semplici
        raise NotImplementedError("Please use the method _transformSingleTrajCoordinates(self, X_lat, X_lon)")

    def _transformSingleTrajCoordinates(self, X_lat, X_lon):
        decoded_gh = geohash.bounds(geohash.encode(X_lat[0], X_lon[0], self.precision)).sw

        return list(map(lambda x: x-decoded_gh.lat, X_lat)), list(map(lambda x: x-decoded_gh.lon, X_lon))