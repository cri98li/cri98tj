from math import inf, log2

import pandas as pd

from cri98tj.distancers.distancer_utils import euclideanBestFitting


"""
ritorna un dataset nella forma class, columns_n... con tid come index
"""
def dataframe_pivot(df, maxLen, verbose, fillna_value, columns):
    df["pos"] = df.groupby(['tid', 'partId']).cumcount()

    if maxLen is not None:
        if maxLen >= 1:
            if verbose: print(F"Cutting sub-trajectories length at {maxLen} over {df.max().pos}", flush=True)
            df = df[df.pos < maxLen]
        else:
            if verbose: print(F"Cutting sub-trajectories length at {df.quantile(.95).pos} over {df.max().pos}", flush=True)
            df = df[df.pos < df.quantile(.95).pos]

    if verbose: print("Pivoting tables", flush=True)
    df_pivot = df.groupby(['tid', 'pos'])[columns].max().unstack().reset_index()
    df_pivot = df_pivot.merge(df.groupby(['tid'])['class'].max().reset_index(), on=["tid"])
    #df_pivot["size"] = df.groupby(['tid']).size()
    df_pivot = df_pivot.drop(columns=[("tid", "")]).set_index("tid")

    if fillna_value is not None:
        df_pivot.fillna(fillna_value, inplace=True)

    return df_pivot[["class"]+[x for x in df_pivot.columns if x != "class"]]

def orderlineScore_leftPure(trajectories, movelet, y_trajectories, y_movelet=None, spatioTemporalColumns=["c1", "c2"]):
    distances = dict()
    for i, trajectory in enumerate(trajectories):
        tmp, distances[i] = euclideanBestFitting(trajectory=trajectory, movelet=movelet,
                                                 spatioTemporalColumns=spatioTemporalColumns)
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
