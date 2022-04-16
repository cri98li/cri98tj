from math import inf, log2
import random

import pandas as pd

from cri98tj.distancers.Euclidean_distancer import euclideanBestFitting
from cri98tj.normalizers.NormalizerInterface import NormalizerInterface

def select_and_pivot(df, n_per_class, normalizer):
    n_per_class = int(n_per_class)
    selected_tid = []
    for classe in df["class"].unique():
        df_tmp = df[df["class"] == classe]
        selected_tid += random.sample(df_tmp.partId.unique().tolist(), k=min(len(df_tmp), n_per_class))

    df = df[df.partId.isin(selected_tid)]

    return normalizer.fit_transform(df.values)


def orderlineScore_leftPure(trajectories, movelet, y_trajectories, y_movelet=None, spatioTemporalColumns=["c1", "c2"], normalizer=NormalizerInterface):
    distances = dict()
    for i, trajectory in enumerate(trajectories):
        tmp, distances[i] = euclideanBestFitting(trajectory=trajectory, movelet=movelet,
                                                 spatioTemporalColumns=spatioTemporalColumns, normalizer=normalizer)
        #print(F"[{movelet}] vs [{trajectory}], alighedAt={tmp} with score of {distances[i]}")

    #plt.scatter(distances.values(), [i for i in range(len(y_trajectories))], c=y_trajectories)
    #plt.show()

    precDist = 0.0
    for i, dist in sorted(distances.items(), key=lambda item: item[1]):
        if y_movelet is None:
            y_movelet = y_trajectories[i]

        if y_movelet != y_trajectories[i]:
            return (dist - precDist)/2

        precDist = dist

    return inf


def _computeEntropy(data={}, classes=[]):
    total = sum([v for k, v in data.items()])
    if total == 0:
        return 0
    entropy = 0.0
    for classe in classes:
        v = 0.0
        if classe in data:
            v = data[classe]
        if v / total != 0:
            entropy += v / total * log2(v / total)

    return entropy * -1


def _infoGain(df=pd.DataFrame(), split=0.0):
    classes = df['class'].unique()
    initialEntropy = _computeEntropy(df.groupby(by=["class"]).count().to_dict()["val"], classes)

    df_min = df[df["val"] <= split]
    entropymin = _computeEntropy(df_min.groupby(by=["class"]).count().to_dict()["val"], classes)

    df_gre = df[df["val"] > split]
    entropygre = _computeEntropy(df_gre.groupby(by=["class"]).count().to_dict()["val"], classes)

    return initialEntropy - (len(df_min) / len(df) * entropymin + len(df_gre) / len(df) * entropygre)

def maxInformationGainScore(trajectories, movelet, y_trajectories, y_movelet=None, spatioTemporalColumns=["c1", "c2"]):
    distances = []
    for i, trajectory in enumerate(trajectories):
        tmp, distance = euclideanBestFitting(trajectory=trajectory, movelet=movelet,spatioTemporalColumns=spatioTemporalColumns)
        distances.append(distance)
        #print(F"[{movelet}] vs [{trajectory}], alighedAt={tmp} with score of {distances[i]}")

    df = pd.DataFrame()
    df["class"] = y_trajectories.ravel()
    df["val"] = distances
    maxInfo=0.0
    for val in df.val.unique():
        info = _infoGain(df, val)
        if info > maxInfo: maxInfo = info


    return maxInfo

"""
tr = [
    [1,2,3, 1,2,3],
    [5,6,7, 5,6,7],
    [1,2,4, 1,2,4]
]

movelets = [
    [3,3],
    [1,1],
    [10,10],
    [50,50]
]

res = []

for movelet in movelets:
    res.append(orderlineScore_leftPure(tr, movelet, y_trajectories=[0,1,0]))

for i, r in sorted(zip(res, movelets), key=lambda x: x[0]):
    print(F"{i} {r}")
"""
