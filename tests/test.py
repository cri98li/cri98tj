import time
from random import random


from sklearn.model_selection import train_test_split

from cri98tj.distancers.Euclidean_distancer import Euclidean_distancer
from cri98tj.normalizers.FirstPoint_normalizer import FirstPoint_normalizer
from cri98tj.partitioners.Geohash_partitioner import Geohash_partitioner
#from cri98tj.selectors.OPTICS_selector import OPTICS_selector
from cri98tj.selectors.RandomInformationGain_selector import RandomInformationGain_selector
from cri98tj.selectors.Random_selector import Random_selector
from cri98tj.selectors.RandomOrderline_selector import RandomOrderline_selector
from cri98tj.prebuilded_transformers.GeohashXmensEuclideian_prebuild import gxe_prebuild
from cri98tj.TrajectoryTransformer import TrajectoryTransformer
from sklearn.ensemble import RandomForestClassifier

from cri98tj.selectors.XMeans_selector import XMeans_selector
import pandas as pd
if __name__ == '__main__':
    df = pd.read_csv(
        '..\\..\\Explaining-Any-Trajectory-Classifier\\Classificatore shapelets\\vehicles_preapred.zip')

    start = time.time()

    df["c1"] = df.c1/100000
    df["c2"] = df.c2/100000

    #df = df[["tid", "class", "c1", "c2"]]

    perc = 1
    max_tid = df.tid.max()
    df = df[df.tid < max_tid*perc]

    spatioTemporalCols = ["c1", "c2", "time"]

    tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                        df.groupby(by=["tid"]).max().reset_index()["class"],
                                                        test_size=.3,
                                                        stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                        random_state=3)


    partitioner = Geohash_partitioner(precision=5, spatioTemporalColumns=spatioTemporalCols)

    part = partitioner.fit_transform(df[df.tid.isin(tid_train)].values)

    normalizer = FirstPoint_normalizer(spatioTemporalColumns=spatioTemporalCols, fillna=None)

    part = normalizer.fit_transform(part)

    #selector = OPTICS_selector(n_jobs=20, fillna_value=0.0, spatioTemporalColumns=spatioTemporalCols)
    selector = Random_selector(movelets_per_class=20, spatioTemporalColumns=spatioTemporalCols)
    #selector = XMeans_selector(kmax=10, fillna_value=0.0)
    #selector = RandomOrderline_selector(top_k=20, movelets_per_class=None, trajectories_for_orderline=None, n_jobs=20)
    #selector = RandomInformationGain_selector(top_k=200, movelets_per_class=None,
                                              #trajectories_for_orderline=None, n_jobs=20, spatioTemporalColumns=["c1", "c2", "time"])

    shapelets = selector.fit_transform(part)

    distancer = Euclidean_distancer(spatioTemporalColumns=spatioTemporalCols, n_jobs=10)

    #dist_np = TrajectoryTransformer(partitioner=partitioner, selector=selector, distancer=distancer).fit_transform(df.values)


    dist_np = distancer.fit_transform((df.values, shapelets))


    clf = RandomForestClassifier(max_depth=None, random_state=0, n_jobs=10)

    dist_np_df = pd.DataFrame(dist_np)
    X = dist_np_df.drop(columns=[0]).values
    y = dist_np_df[0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=3)

    clf.fit(X_train, y_train)

    from sklearn.metrics import classification_report

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    print(F"total time: {time.time()-start}")
# %%
