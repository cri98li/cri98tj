import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm.autonotebook import tqdm

from cri98tj.distancers.DistancerInterface import DistancerInterface
from cri98tj.normalizers.NormalizerInterface import NormalizerInterface
from cri98tj.normalizers.normalizer_utils import dataframe_pivot


class RotatingEuclidean_distancer(DistancerInterface):

    def __init__(self, normalizer, n_jobs=1, spatioTemporalColumns=["c1", "c2"], verbose=True):
        self.verbose = verbose
        self.normalizer = normalizer
        self.spatioTemporalColumns = spatioTemporalColumns
        self.n_jobs = n_jobs

    def fit(self, trajectories_movelets):
        return self

    # trajectories = tid, class, time, c1, c2
    # restituisce nparray con pos0= cluster e poi
    def transform(self, trajectories_movelets):
        trajectories, movelets = trajectories_movelets

        trajectories_df = pd.DataFrame(trajectories, columns=["tid", "class"]+self.spatioTemporalColumns)
        trajectories_df["partId"] = trajectories_df.tid
        df_pivot = dataframe_pivot(df=trajectories_df, maxLen=None, verbose=self.verbose, fillna_value=None, columns=self.spatioTemporalColumns)

        distances = np.zeros((df_pivot.shape[0], len(movelets)))
        best_is = np.zeros((df_pivot.shape[0], len(movelets)))

        executor = ProcessPoolExecutor(max_workers=self.n_jobs)

        ndarray_pivot = df_pivot[[x for x in df_pivot.columns if x != "class"]].values
        processes = []
        for i, movelet in enumerate(tqdm(movelets, disable=not self.verbose, position=0)):
            processes.append(executor.submit(self._foo, i, movelet, ndarray_pivot, self.spatioTemporalColumns))

        if self.verbose: print(f"Collecting distances from {len(processes)}")
        for i, process in enumerate(tqdm(processes)):
            ind, col = process.result()
            for j, (val, best_i) in enumerate(zip(col, ind)):
                distances[j, i] = val
                best_is[j, i] = int(best_i)

        executor.shutdown(wait=True)
        """for i, movelet in enumerate(tqdm(movelets, disable=not self.verbose, position=0)):
            for j, val in enumerate(self._foo(i, movelet, ndarray_pivot)):
                distances[j, i] = val"""

        return best_is, np.hstack((df_pivot[["class"]].values, distances))

    def _foo(self, i, movelet, trajectories, spatioTemporalColumns):
        distances = []
        indexes = []
        for j, trajectory in enumerate(tqdm(trajectories, disable=True, position=i + 1, leave=True)):
            best_i, best_score = rotatingEuclideanBestFitting(trajectory=trajectory, movelet=movelet,
                                                      spatioTemporalColumns=spatioTemporalColumns, normalizer=self.normalizer)
            distances.append(best_score)
            indexes.append(best_i)

        return indexes, distances

def rotatingEuclideanBestFitting(trajectory, movelet, spatioTemporalColumns, normalizer=NormalizerInterface()):  # nan == end
    if len(trajectory) % len(spatioTemporalColumns) != 0:
        raise Exception(f"la lunghezza della traiettoria deve essere divisivile per {len(spatioTemporalColumns)}")
    if len(movelet) % len(spatioTemporalColumns) != 0:
        raise Exception(f"la lunghezza della traiettoria deve essere divisivile per {len(spatioTemporalColumns)}")

    offset_trajectory = int(len(trajectory) / len(spatioTemporalColumns))
    offset_movelet = int(len(movelet) / len(spatioTemporalColumns))

    len_mov = 0
    for el in movelet:
        if np.isnan(el) or len_mov >= offset_movelet:
            break
        len_mov += 1

    len_t = 0
    for el in trajectory:
        if np.isnan(el) or len_t >= offset_trajectory:
            break
        len_t += 1

    if len_mov > len_t:
        return rotatingEuclideanBestFitting(movelet, trajectory, spatioTemporalColumns, normalizer)

    trajectory_dict = [None for x in spatioTemporalColumns]
    movelet_dict = [None for x in spatioTemporalColumns]

    for i, col in enumerate(spatioTemporalColumns):
        trajectory_dict[i] = trajectory[i * offset_trajectory:(i * offset_trajectory) + len_t]
        movelet_dict[i] = normalizer._transformSingleTraj(normalizer._transformSingleTraj(movelet[i * offset_movelet:(i * offset_movelet) + len_mov]))

    bestScore = math.inf
    best_i = -1
    for i in range(len_t - len_mov + 1):
        trajectory_dict_cut = dict()
        for j, col in enumerate(spatioTemporalColumns):
            trajectory_dict_cut[j] = normalizer._transformSingleTraj(trajectory_dict[j][i:i + len_mov])

        returned = minimize_scalar(lambda theta: _rotatingEuclideanDistance(trajectory_dict_cut,
                                                                            _rotate(movelet_dict, theta),
                                                                            spatioTemporalColumns)).fun

        #returned = _rotatingEuclideanDistance(trajectory_dict_cut, movelet_dict, spatioTemporalColumns, bestScore)
        if returned is not None and returned < bestScore:
            bestScore = returned
            best_i = i

    return best_i, bestScore

def _rotate(geolet, theta):

    newList = [(x * math.cos(theta) + y * -math.sin(theta), (x * math.sin(theta) + y * math.cos(theta))) for x, y in
            zip(geolet[0], geolet[1])]


    return [[x[0] for x in newList], [x[1] for x in newList]]


def _rotatingEuclideanDistance(trajectory=[], movelet=[], spatioTemporalColumns=[], best_score=math.inf):
    _len = len(trajectory[0])

    sum = 0
    for i in range(_len):
        tmp = 0.0
        for j in range(len(spatioTemporalColumns)):
            tmp += (trajectory[j][i] - movelet[j][i]) ** 2
        sum += math.sqrt(tmp)
        if sum / _len > best_score:
            return None

    return sum / _len