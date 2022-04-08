import functools

import numpy as np
import pandas as pd


class TrajectoryFrame():
    columns = []
    tColumns = []
    data = []

    def __init__(self, df=pd.DataFrame(), tidColumn="tid", timeDependentColumns=[]):
        self.columns = [x for x in df.columns if x not in timeDependentColumns]
        self.tColumns = timeDependentColumns
        for tid in df[tidColumn].unique():
            df_tid = df[df[tidColumn] == tid]

            d = dict()
            for col in self.columns:
                d[col] = df_tid[col].iloc[0]

            timeDipendentFeatures = []
            for row in df_tid[self.tColumns].values:
                timeDipendentFeatures.append(tuple(row))
            d["timeDipendentFeatures"] = timeDipendentFeatures

            self.data.append(d)


    #se <1 pewrcentile, altrimenti length
    def cutTColumns(self, n):
        if n <= 0:
            raise Exception("n must be > 0")
        if n < 1:
            lengths = map(lambda d: len(d["timeDipendentFeatures"]), self.data)
            lengths = sorted(lengths)
            n = lengths[round((len(lengths)+1)/n)]

        for d in self.data:
            d["timeDipendentFeatures"] = d["timeDipendentFeatures"][:n]




    def toDataFrame(self):
        df_len = functools.reduce(lambda a, b: a + b, map(lambda d: len(d["timeDipendentFeatures"]), self.data))
        df_values = np.zeros((df_len, len(self.columns + self.tColumns)))

        offset_row = 0
        for i, d in enumerate(self.data):

            for tupla in d["timeDipendentFeatures"]:
                for j, key in enumerate(self.columns):
                    df_values[i + offset_row, j] = d[key]


                tupla = list(tupla)
                for j, val in enumerate(tupla):
                    df_values[i+offset_row, j + len(self.columns)] = val
                offset_row += 1

            offset_row-=1
        return pd.DataFrame(df_values, columns=self.columns + self.tColumns)


df = pd.DataFrame([
    [1, 1, 1],
    [1, 2, 2],
    [2, 1, 1],
    [3, 1, 1]
], columns=["tid", "lat", "lon"])

tdf = TrajectoryFrame(df, "tid", ["lat", "lon"])
tdf.cutTColumns(1)

print(tdf.data)

print(tdf.toDataFrame())
