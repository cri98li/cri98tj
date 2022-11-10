import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm.autonotebook import tqdm

from cri98tj.distancers.DistancerInterface import DistancerInterface
from cri98tj.normalizers.NormalizerInterface import NormalizerInterface
from cri98tj.normalizers.normalizer_utils import dataframe_pivot


class RotatingIRD_distancer(DistancerInterface):

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
        for i, process in enumerate(tqdm(processes, disable=not self.verbose, position=0)):
            ind, col = process.result()
            for j, (val, best_i) in enumerate(zip(col, ind)):
                distances[j, i] = val
                best_is[j, i] = int(best_i)

        executor.shutdown(wait=True)
        """for i, movelet in enumerate(tqdm(movelets, disable=not self.verbose, position=0)):
            for j, val in enumerate(self._foo(i, movelet, ndarray_pivot)):
                distances[j, i] = val"""

        return best_is, np.hstack((df_pivot[["class"]].values, distances))

    def _foo(self,i, movelet, ndarray_pivot, spatioTemporalColumns):
        distances = []
        indexes = []
        for j, trajectory in enumerate( tqdm(ndarray_pivot, disable=True, position=i+1, leave=True)):
            best_i, best_score = rotatingIRDBestFitting(trajectory=trajectory, movelet=movelet,
                                                      spatioTemporalColumns=spatioTemporalColumns, normalizer=self.normalizer)
            distances.append(best_score)
            indexes.append(best_i)

        return indexes, distances

def _rotate(geolet, theta):

    newList = [(x * math.cos(theta) + y * -math.sin(theta), (x * math.sin(theta) + y * math.cos(theta))) for x, y in
            zip(geolet[0], geolet[1])]


    return [[x[0] for x in newList], [x[1] for x in newList], geolet[2]]

def rotatingIRDBestFitting(trajectory, movelet, spatioTemporalColumns, window=None, normalizer=NormalizerInterface()):  # nan == end
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

    trajectory_dict = [None for x in spatioTemporalColumns]
    movelet_dict = [None for x in spatioTemporalColumns]

    for i, col in enumerate(spatioTemporalColumns):
        trajectory_dict[i] = trajectory[i * offset_trajectory:(i * offset_trajectory) + len_t]
        movelet_dict[i] = normalizer._transformSingleTraj(movelet[i * offset_movelet:(i * offset_movelet) + len_mov])            #TODO: va bene solo se sono allo stesso sample rate, sennò va visto per il tempo

    bestScore = math.inf
    best_i = -1
    for i in [0]+list(range(1, len_t - len_mov + 1)): #mi assicuro di fare almeno 1 iterazione
        trajectory_dict_cut = dict()
        for j, col in enumerate(spatioTemporalColumns):
            trajectory_dict_cut[j] = normalizer._transformSingleTraj(trajectory_dict[j][i:i + len_mov])
        #returned = trajectory_distance(trajectory_dict_cut, movelet_dict)
        returned = minimize_scalar(lambda theta: trajectory_distance(trajectory_dict_cut,
                                                                            _rotate(movelet_dict, theta)),
                                   #options={"maxiter": 20}
                                   ).fun
        if returned is not None and returned < bestScore:
            bestScore = returned
            best_i = i

    return best_i, bestScore




def _transformTraj(x): #[[lat1, lat2, ...], [lon1, lon2, ...], [t1, t2, ...]] --> [[lat1,lon1,t1], [lat2, lon2,t2], [..., ...], ...]
    if len(x) != 3:
        raise NotImplementedError("This method is suitable only for trajectory data (lat, lon, time)")

    return list(zip(x[0], x[1], x[2]))

#From here credits to Riccardo Guidotti

def spherical_distance(a, b):
    lat1 = a[1]
    lon1 = a[0]
    lat2 = b[1]
    lon2 = b[0]
    R = 6371000
    rlon1 = lon1 * math.pi / 180.0
    rlon2 = lon2 * math.pi / 180.0
    rlat1 = lat1 * math.pi / 180.0
    rlat2 = lat2 * math.pi / 180.0
    dlon = (rlon1 - rlon2) / 2.0
    dlat = (rlat1 - rlat2) / 2.0
    lat12 = (rlat1 + rlat2) / 2.0
    sindlat = math.sin(dlat)
    sindlon = math.sin(dlon)
    cosdlon = math.cos(dlon)
    coslat12 = math.cos(lat12)
    f = sindlat * sindlat * cosdlon * cosdlon + sindlon * sindlon * coslat12 * coslat12
    f = math.sqrt(f)
    f = math.asin(f) * 2.0 # the angle between the points
    f *= R
    return f

def trajectory_distance(tr1, tr2):
    if len(tr1[0]) == 1 and len(tr2[0]) == 1:
        return .0


    tr1 = _transformTraj(tr1)
    tr2 = _transformTraj(tr2)

    i1 = 0
    i2 = 0
    np = 0

    last_tr1 = tr1[i1]
    last_tr2 = tr2[i2]

    dist = spherical_distance(last_tr1, last_tr2)
    np += 1

    while True:
        if i1 >= (len(tr1)-1) or i2 >= (len(tr2)-1):
            break

        step_tr1 = spherical_distance(last_tr1, tr1[i1 + 1])
        step_tr2 = spherical_distance(last_tr2, tr2[i2 + 1])

        if step_tr1 < step_tr2:
            i1 += 1
            last_tr1 = tr1[i1]
            last_tr2 = closest_point_on_segment(last_tr2, tr2[i2+1], last_tr1)
        elif step_tr1 > step_tr2:
            i2 += 1
            last_tr2 = tr2[i2]
            last_tr1 = closest_point_on_segment(last_tr1, tr1[i1+1], last_tr2)
        else:
            i1 += 1
            i2 += 1
            last_tr1 = tr1[i1]
            last_tr2 = tr2[i2]

        d = spherical_distance(last_tr1, last_tr2)

        dist += d
        np += 1

    for i in range(i1, len(tr1)):
        d = spherical_distance(tr2[-1], tr1[i])
        dist += d
        np += 1

    for i in range(i2, len(tr2)):
        d = spherical_distance(tr1[-1], tr2[i])
        dist += d
        np += 1

    dist = 1.0 * dist / np

    return dist

def closest_point_on_segment(a, b, p):
    sx1 = a[0]
    sx2 = b[0]
    sy1 = a[1]
    sy2 = b[1]
    sz1 = a[2]
    sz2 = b[2]
    px = p[0]
    py = p[1]

    x_delta = sx2 - sx1
    y_delta = sy2 - sy1
    z_delta = sz2 - sz1

    if x_delta == 0 and y_delta == 0:
        return p

    u = ((px - sx1) * x_delta + (py - sy1) * y_delta) / (x_delta * x_delta + y_delta * y_delta)
    if u < 0:
        closest_point = a
    elif u > 1:
        closest_point = b
    else:
        cp_x = sx1 + u * x_delta
        cp_y = sy1 + u * y_delta
        dist_a_cp = spherical_distance(a, [cp_x, cp_y, 0])
        if dist_a_cp != 0:
            cp_z = sz1 + z_delta / (spherical_distance(a, b) / spherical_distance(a, [cp_x, cp_y, 0]))
        else:
            cp_z = a[2]
        closest_point = [cp_x, cp_y, cp_z]

    return closest_point