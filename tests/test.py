import math
import time
from random import random

import numpy as np
from sklearn.model_selection import train_test_split

from cri98tj.distancers.Euclidean_distancer import Euclidean_distancer
from cri98tj.distancers.DTW_distancer import DTW_distancer
from cri98tj.distancers.InterpolatedRootDistance_distancer import InterpolatedRootDistance_distancer, \
    InterpolatedRootDistanceBestFitting
from cri98tj.normalizers.FirstPoint_normalizer import FirstPoint_normalizer
from cri98tj.normalizers.normalizer_utils import dataframe_pivot
from cri98tj.partitioners.Geohash_partitioner import Geohash_partitioner
from cri98tj.selectors.RandomInformationGain_selector import RandomInformationGain_selector
from cri98tj.selectors.Random_selector import Random_selector
from cri98tj.selectors.RandomOrderline_selector import RandomOrderline_selector
from cri98tj.TrajectoryTransformer import TrajectoryTransformer
from sklearn.ensemble import RandomForestClassifier

import copy

import pandas as pd

import plotly.express as px

if __name__ == '__main__':
    def print_movelets(movelets, spatioTemporalColumns):
        list_df = []

        for n_mov, movelet in enumerate(movelets):
            if len(movelet) % len(spatioTemporalColumns) != 0:
                raise Exception(f"la lunghezza della traiettoria deve essere divisivile per {len(spatioTemporalColumns)}")

            offset_movelet = int(len(movelet) / len(spatioTemporalColumns))


            len_mov = 0
            for el in movelet:
                if np.isnan(el) or len_mov >= offset_movelet:
                    break
                len_mov += 1

            movelet_dict = [None for x in spatioTemporalColumns]

            for i, col in enumerate(spatioTemporalColumns):
                movelet_dict[i] = movelet[i * offset_movelet:(i * offset_movelet) + len_mov]

            df_tmp = pd.DataFrame(movelet_dict).T.rename(columns={0: "c1", 1:"c2", 2:"time"})
            df_tmp["id"] = n_mov

            list_df.append(df_tmp)

        fig = px.line_3d(pd.concat(list_df), x="c1", y="c2", z="time", color="id")
        fig.show()

    def print_movelets_in_traj(trajectory, best_i, movelets_, spatioTemporalColumns):
        list_df = []
        movelets =copy.deepcopy(movelets_)

        if len(trajectory) % len(spatioTemporalColumns) != 0:
            raise Exception(f"la lunghezza della traiettoria deve essere divisivile per {len(spatioTemporalColumns)}")

        offset_trajectory = int(len(trajectory) / len(spatioTemporalColumns))

        len_t = 0
        for el in trajectory:
            if np.isnan(el) or len_t >= offset_trajectory:
                break
            len_t += 1

        trajectory_dict = [None for x in spatioTemporalColumns]

        for i, col in enumerate(spatioTemporalColumns):
            trajectory_dict[i] = trajectory[i * offset_trajectory:(i * offset_trajectory) + len_t]

        df_tmp = pd.DataFrame(trajectory_dict).T.rename(columns={0: "c1", 1: "c2", 2: "time"})
        df_tmp["id"] = "traj"
        list_df.append(df_tmp)

        for mov_n, movelet in enumerate(movelets):
            if len(movelet) % len(spatioTemporalColumns) != 0:
                raise Exception(f"la lunghezza della traiettoria deve essere divisivile per {len(spatioTemporalColumns)}")

            offset_movelet = int(len(movelet) / len(spatioTemporalColumns))

            len_mov = 0
            for el in movelet:
                if np.isnan(el) or len_mov >= offset_movelet:
                    break
                len_mov += 1

            #if len_mov > len_t:
            #    raise Warning("Inverse-plot")

            movelet_dict = [0 for x in spatioTemporalColumns]

            for i, col in enumerate(spatioTemporalColumns):
                movelet_dict[i] = movelet[i * offset_movelet:(i * offset_movelet) + len_mov]
                for j in range(len_mov):
                    movelet_dict[i][j] += trajectory_dict[i][int(best_i[mov_n])]
            df_tmp = pd.DataFrame(movelet_dict).T.rename(columns={0: "c1", 1: "c2", 2: "time"})
            df_tmp["id"] = mov_n

            list_df.append(df_tmp)

        fig = px.line_3d(pd.concat(list_df), x="c1", y="c2", z="time", color="id")
        fig.show()







    #df = pd.read_csv('../examples/Animals Dataset/data/animals_preapred.zip').sort_values(by=["tid", "t"])# precision=5, 50 movelet, DTW
    #df = pd.read_csv('../examples/Vehicles Dataset/data/vehicles_preapred.zip').sort_values(by=["tid", "t"])
    df = pd.read_csv('../examples/Taxi Dataset/data/train_denorm_1mese.zip').sort_values(by=["tid", "TIMESTAMP"])

    start = time.time()

    df = df[["tid", "day_of_week", "lat", "lon", "TIMESTAMP"]].rename(columns={"day_of_week": "class", "lat": "c1", "lon": "c2", "TIMESTAMP": "t"})

    df = df[["tid", "class", "c1", "c2", "t"]]

    #df["c1"] = df.c1/100000
    #df["c2"] = df.c2/100000



    perc = 1
    df.tid -= df.tid.min()
    max_tid = df.tid.max()
    df = df[df.tid < max_tid*perc]

    spatioTemporalCols = ["c1", "c2", "t"]

    tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                        df.groupby(by=["tid"]).max().reset_index()["class"],
                                                        test_size=.3,
                                                        stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                        random_state=3)


    partitioner = Geohash_partitioner(precision=6, spatioTemporalColumns=spatioTemporalCols)

    part = partitioner.fit_transform(df[df.tid.isin(tid_train)].values)

    normalizer = FirstPoint_normalizer(spatioTemporalColumns=spatioTemporalCols, fillna=None)


    #selector = Random_selector(movelets_per_class=2, normalizer=normalizer)
    #selector = RandomOrderline_selector(top_k=50, movelets_per_class=None, trajectories_for_orderline=50, n_jobs=10, spatioTemporalColumns=spatioTemporalCols, normalizer=normalizer)
    #TODO: left pure troppo restrittiva (tutte le distanze sono = 0)

    selector = RandomInformationGain_selector(top_k=20, bestFittingMeasure=InterpolatedRootDistanceBestFitting, movelets_per_class=100, trajectories_for_orderline=50, n_jobs=10, spatioTemporalColumns=spatioTemporalCols, normalizer=normalizer)

    shapelets = selector.fit_transform(part)

    print_movelets(shapelets, spatioTemporalCols)

    #distancer = Euclidean_distancer(normalizer=normalizer, spatioTemporalColumns=spatioTemporalCols, n_jobs=4)
    #distancer = DTW_distancer(normalizer=normalizer, spatioTemporalColumns=spatioTemporalCols, n_jobs=12)
    distancer = InterpolatedRootDistance_distancer(normalizer=normalizer, spatioTemporalColumns=spatioTemporalCols, n_jobs=10)

    #dist_np = TrajectoryTransformer(partitioner=partitioner, selector=selector, distancer=distancer).fit_transform(df.values)


    best_is, dist_np = distancer.fit_transform((df.values, shapelets))

    df2 = df

    df2["partId"] = df2.tid

    df_pivot = dataframe_pivot(df=df, maxLen=None, verbose=False, fillna_value=None,
                               columns=spatioTemporalCols)

    for i in range(min(5, len(df_pivot))):
        print_movelets_in_traj(df_pivot.values[i][1:], best_is[:][i], shapelets, spatioTemporalCols)
        time.sleep(2)


    clf = RandomForestClassifier(max_depth=2, random_state=3, n_jobs=10, n_estimators=1000)

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
