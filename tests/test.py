import time
from random import random


from sklearn.model_selection import train_test_split

from cri98tj.distancers.Euclidean_distancer import Euclidean_distancer
from cri98tj.distancers.DTW_distancer import DTW_distancer
from cri98tj.normalizers.FirstPoint_normalizer import FirstPoint_normalizer
from cri98tj.partitioners.Geohash_partitioner import Geohash_partitioner
#from cri98tj.selectors.OPTICS_selector import OPTICS_selector
from cri98tj.selectors.OPTICS_selector import OPTICS_selector
from cri98tj.selectors.RandomInformationGain_selector import RandomInformationGain_selector
from cri98tj.selectors.Random_selector import Random_selector
from cri98tj.selectors.RandomOrderline_selector import RandomOrderline_selector
from cri98tj.prebuilded_transformers.GeohashXmensEuclideian_prebuild import gxe_prebuild
from cri98tj.TrajectoryTransformer import TrajectoryTransformer
from sklearn.ensemble import RandomForestClassifier

from cri98tj.selectors.XMeans_selector import XMeans_selector
import pandas as pd
if __name__ == '__main__':
    #df = pd.read_csv('..\\..\\cri98tj\\datasets\\animals\\animals_preapred.zip').sort_values(by=["tid", "t"])# precision=5, 50 movelet, DTW
    df = pd.read_csv('..\\..\\cri98tj\\datasets\\vehicles\\vehicles_preapred.zip').sort_values(by=["tid", "t"])

    start = time.time()

    #df = df[["tid", "day_of_week", "lat", "lon"]].rename(columns={"day_of_week": "class", "lat": "c1", "lon": "c2"})

    df = df[["tid", "class", "c1", "c2"]]

    df["c1"] = df.c1/100000
    df["c2"] = df.c2/100000



    perc = 1
    df.tid -= df.tid.min()
    max_tid = df.tid.max()
    df = df[df.tid < max_tid*perc]

    spatioTemporalCols = ["c1", "c2"]

    tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                        df.groupby(by=["tid"]).max().reset_index()["class"],
                                                        test_size=.3,
                                                        stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                        random_state=3)


    partitioner = Geohash_partitioner(precision=4, spatioTemporalColumns=spatioTemporalCols)

    part = partitioner.fit_transform(df[df.tid.isin(tid_train)].values)

    normalizer = FirstPoint_normalizer(spatioTemporalColumns=spatioTemporalCols, fillna=None)


    #selector = OPTICS_selector(n_jobs=20, normalizer=normalizer)
    #selector = Random_selector(movelets_per_class=50, normalizer=normalizer)
    #selector = XMeans_selector(kmax=10, normalizer=FirstPoint_normalizer(spatioTemporalColumns=spatioTemporalCols, fillna=0.0))
    selector = RandomOrderline_selector(top_k=50, movelets_per_class=None, trajectories_for_orderline=None, n_jobs=12, spatioTemporalColumns=spatioTemporalCols, normalizer=normalizer)
    #selector = RandomInformationGain_selector(top_k=50, movelets_per_class=None, trajectories_for_orderline=None, n_jobs=20, spatioTemporalColumns=["c1", "c2", "time"])

    shapelets = selector.fit_transform(part)

    distancer = Euclidean_distancer(normalizer=normalizer, spatioTemporalColumns=spatioTemporalCols, n_jobs=18)
    #distancer = DTW_distancer(normalizer=normalizer, spatioTemporalColumns=spatioTemporalCols, n_jobs=12)

    #dist_np = TrajectoryTransformer(partitioner=partitioner, selector=selector, distancer=distancer).fit_transform(df.values)


    dist_np = distancer.fit_transform((df.values, shapelets))


    clf = RandomForestClassifier(max_depth=None, random_state=0, n_jobs=10, n_estimators=1000)

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
