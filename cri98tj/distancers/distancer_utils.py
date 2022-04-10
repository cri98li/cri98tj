import math
from math import inf

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def euclideanBestFitting(trajectory, movelet, spatioTemporalColumns):  # nan == end
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
        return euclideanBestFitting(movelet, trajectory, spatioTemporalColumns)

    trajectory_dict = [None for x in spatioTemporalColumns]
    movelet_dict = [None for x in spatioTemporalColumns]

    for i, col in enumerate(spatioTemporalColumns):
        trajectory_dict[i] = trajectory[i * offset_trajectory:(i * offset_trajectory) + len_t]
        movelet_dict[i] = movelet[i * offset_movelet:(i * offset_movelet) + len_mov]

    bestScore = inf
    best_i = -1
    for i in range(len_t - len_mov + 1):
        trajectory_dict_cut = dict()
        for j, col in enumerate(spatioTemporalColumns):
            trajectory_dict_cut[j] = trajectory_dict[j][i:i + len_mov]
        returned = _euclideanDistance(trajectory_dict_cut, movelet_dict, spatioTemporalColumns, bestScore)
        if returned is not None and returned < bestScore:
            bestScore = returned
            best_i = i

    return best_i, bestScore


def _euclideanDistance(trajectory=[], movelet=[], spatioTemporalColumns=[], best_score=inf):
    _len = len(trajectory[0])

    sum = 0
    for i in range(_len):
        tmp = 0.0
        for j in range(len(spatioTemporalColumns)):
            tmp += (trajectory[j][i] - movelet[j][i]+trajectory[j].min()) ** 2
        sum += math.sqrt(tmp)
        if sum / _len > best_score:
            return None

    return sum / _len


def DTWBestFitting(trajectory, movelet, spatioTemporalColumns, window=None):  # nan == end
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
        return euclideanBestFitting(movelet, trajectory, spatioTemporalColumns)

    trajectory_dict = [None for x in spatioTemporalColumns]
    movelet_dict = [None for x in spatioTemporalColumns]

    for i, col in enumerate(spatioTemporalColumns):
        trajectory_dict[i] = trajectory[i * offset_trajectory:(i * offset_trajectory) + len_t]
        movelet_dict[i] = movelet[i * offset_movelet:(i * offset_movelet) + len_mov]

    returned = _DTWBestFitting(trajectory=trajectory_dict, movelet=movelet_dict, spatioTemporalColumns=spatioTemporalColumns, window=window)

    return None, returned


def _DTWBestFitting(trajectory, movelet, spatioTemporalColumns, window=None):
    spatioTemporalColumns = [x for x in spatioTemporalColumns if x not in ["t", "time", "timestamp", "TIMESTAMP"]]

    n, m = len(movelet[0]), len(trajectory[0])
    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        loop_min =1
        loop_max = m
        if window is not None:
            loop_min = max(1, i-window)
            loop_max = min(m, i+window)

        for j in range(loop_min, loop_max + 1):
            cost = 0
            for k in range(len(spatioTemporalColumns)):
                cost += (movelet[k][i - 1]+trajectory[k].min() - trajectory[k][j - 1]) ** 2
            cost **= (1 / len(spatioTemporalColumns))
            # take last min from a square box
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n, m]
