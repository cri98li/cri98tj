import ctypes
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.sharedctypes import RawArray

import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from skltemplate.distancers.DistancerInterface import DistancerInterface
from multiprocessing import Array


class Euclidean_distancer(DistancerInterface):

    def _bestFitting(self, trajectory, movelet, optimize=True, fillna=0.0):
        if trajectory.shape[0] % 2 != 0:
            raise Exception("la lunghezza della traiettoria non è un numero pari")
        if len(movelet) % 2 != 0:
            raise Exception("la lunghezza della movelet non è un numero pari")
        offset_trajectory = round(len(trajectory) / 2)
        offset_movelet = round(len(movelet) / 2)

        best_i = -1
        best_score = float("inf")
        nullSum = 0
        for i in range(offset_trajectory - offset_movelet):
            if optimize:
                if trajectory[i] == fillna:
                    nullSum += 1
                else:
                    nullSum = 0

                if nullSum > 10: continue

            sum = 0
            for j in range(offset_movelet):
                t_lat = trajectory[i + j]
                m_lat = movelet[j]
                t_lon = trajectory[i + j + offset_trajectory]
                m_lon = movelet[j + offset_movelet]
                sum += (t_lat - m_lat) ** 2 + (t_lon - m_lon) ** 2
            if sum < best_score:
                best_score = sum
                best_i = i

        return best_i, best_score

    def __init__(self, verbose=True):
        self.verbose = verbose

    def fit(self, trajectories_movelets):
        return self

    # trajectories = tid, class, time, c1, c2
    # restituisce nparray con pos0= cluster e poi
    def transform(self, trajectories_movelets):
        trajectories, movelets = trajectories_movelets

        trajectories_df = pd.DataFrame(trajectories, columns=["tid", "class", "time", "c1", "c2"])
        trajectories_df["pos"] = trajectories_df.groupby(['tid']).cumcount()
        df_pivot = trajectories_df.groupby(['tid', 'pos'])[['c1', 'c2']].max().unstack().reset_index()
        df_pivot = df_pivot.merge(trajectories_df.groupby(['tid'])['class'].max().reset_index(), on=["tid"])
        df_pivot.fillna(0, inplace=True)
        df_pivot = df_pivot.drop(columns=[("tid", "")]).set_index("tid")

        distances = np.zeros((df_pivot.shape[0], len(movelets)))

        executor = ProcessPoolExecutor(max_workers=self.n)

        ndarray_pivot = df_pivot[[x for x in df_pivot.columns if x != "class"]].values
        processes = []
        for i, movelet in enumerate(tqdm(movelets, disable=not self.verbose, position=0)):
            processes.append(executor.submit(self._foo, i, movelet, ndarray_pivot))

        for i, process in enumerate(tqdm(processes)):
            col = process.result()
            for j, val in enumerate(col):
                distances[j, i] = val

        executor.shutdown(wait=True)

        return np.hstack((df_pivot[["class"]].values, distances))

    def _foo(self,i, movelet, ndarray_pivot):
        distances = []
        for j, trajectory in enumerate(
                tqdm(ndarray_pivot, disable=True,
                     position=i+1, leave=True)):
            best_i, best_score = self._bestFitting(trajectory, movelet)
            distances.append(best_score)

        return distances
