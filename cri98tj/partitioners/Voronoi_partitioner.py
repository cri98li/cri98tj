import os
from datetime import datetime

import pandas as pd
import skmob
import geopandas as gpd
from geolib import geohash
from sklearn.exceptions import *
from tqdm.auto import tqdm

from cri98tj.partitioners.PartitionerInterface import PartitionerInterface


class Voronoi_partitioner(PartitionerInterface):
    """
    precision : number, default=7
    A value used by the partitioner to set the partitions size
    """

    def _checkFormat(self, X):
        if X.shape[1] != 2 + len(self.spatioTemporalColumns):
            raise DataDimensionalityWarning("The input data must be in this form (tid, class)+ spatioTemporalColumns")
        # TODO: standardizzare la struttura dati

        if self.spatioTemporalColumns != ["c1", "c2", "t"]:
            raise Exception('This partitioner require spatioTemporalColumns=["c1", "c2", "t"]')

    def __init__(self, radius=200, stop_distance=50, stop_seconds=500, spatioTemporalColumns=None, verbose=True):
        self.radius = radius
        self.stop_distance = stop_distance
        self.stop_seconds = stop_seconds

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
    tid, class, spatioTemporalColumns, voronoi
    """

    def transform(self, X):
        self._checkFormat(X)

        df = pd.DataFrame(X, columns=["tid", "class"] + self.spatioTemporalColumns)

        self.__to_OCTO(df)
        self.__OCTO_processor()

        tessellation = gpd.read_file("voronoi.shp").reset_index().rename(columns={"index": "tile_ID"})

        tdf = skmob.TrajDataFrame(df.rename(columns={"c1": "lat", "c2": "lng"}))

        map_tdf = tdf.mapping(tessellation, remove_na=True).sort_values(by=["tid", "t", "tile_ID"])

        self.__OCTO_clear()

        prec = ()
        c = -1
        returnValue = map_tdf.values
        for row in returnValue:
            key = (row[0], row[5])
            if prec != key:
                prec = key
                c += 1
            row[5] = str(row[5])+ "_"+ str(c)

        return returnValue


    def __to_OCTO(self, df_original):
        df = df_original.copy()
        df["date"] = df.t.apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
        df.t = df.t.apply(lambda x: datetime.fromtimestamp(x).strftime("%H:%M:%S"))
        df.c1 = df.c1.apply(lambda x: x * (10 ** 6))
        df.c2 = df.c2.apply(lambda x: x * (10 ** 6))

        # campi fittizzi
        df["signal"] = 0
        df["heading"] = 0
        df["quality"] = 3
        df["status"] = 2
        df["delta"] = 0

        df = df[["tid", "date", "t", "c1", "c2", "signal", "heading", "quality", "status", "delta"]]

        os.system("mkdir tmp")

        df.to_csv("tmp/gps.csv", index=None, header=None)

    def __OCTO_processor(self):
        commands = [
            "cat tmp/gps.csv | sort -t',' -k1 -k2 -k3 -k9 -m | gzip -c > tmp/gps_sorted.csv.gz",
            f"java -Xmx2048M -jar octo_processor.jar -app spacetime_reconstruct -i tmp/gps_sorted.csv.gz -o tmp/gps_600_50.csv.gz -ds {self.stop_distance} -dt {self.stop_seconds}",
            f"java -Xmx2048M -jar octo_processor.jar -app point_aggregate -i tmp/gps_600_50.csv.gz -o tmp/gps_edgelist_200.csv.gz -r {self.radius}"
        ]

        for command in commands:
            if self.verbose: print(f"Executing command: {command}")

            if(os.system(command) != 0):
                raise Exception("Error while executing: "+command)


    def __OCTO_clear(self):
        os.system("rm -rf tmp")

