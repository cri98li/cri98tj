from math import sqrt, inf
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def euclideanBestFitting(trajectory, movelet, spatioTemporalColumns): #nan == end
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
        if np.isnan(el) or len_t>=offset_trajectory:
            break
        len_t += 1

    if len_mov > len_t:
        return euclideanBestFitting(movelet, trajectory, spatioTemporalColumns)

    trajectory_dict = [None for x in spatioTemporalColumns]
    movelet_dict = [None for x in spatioTemporalColumns]

    for i, col in enumerate(spatioTemporalColumns):
        trajectory_dict[i] = trajectory[i*offset_trajectory:(i*offset_trajectory)+len_t]
        movelet_dict[i] = movelet[i * offset_movelet:(i * offset_movelet) + len_mov]

    bestScore = inf
    best_i = -1
    for i in range(len_t-len_mov+1):
        trajectory_dict_cut = dict()
        for j, col in enumerate(spatioTemporalColumns):
            trajectory_dict_cut[j] = trajectory_dict[j][i:i+len_mov]
        returned = _euclideanDistance(trajectory_dict_cut, movelet_dict, spatioTemporalColumns, bestScore)
        if returned is not None and returned < bestScore:
            bestScore = returned
            best_i = i


    return best_i, bestScore


def _euclideanDistance(trajectory=[], movelet=[], spatioTemporalColumns=[], best_score=inf):
    _len = len(trajectory[0])

    for i, feature in enumerate(spatioTemporalColumns):
        trajectory[i] = MinMaxScaler().fit_transform(trajectory[i].reshape(-1,1))
        movelet[i] = MinMaxScaler().fit_transform(movelet[i].reshape(-1,1))

    sum = 0
    for i in range(_len):
        for j in range(len(spatioTemporalColumns)):
            sum += abs(trajectory[j][i][0] - movelet[j][i][0])
        if sum/_len > best_score:
            return None

    return sum/_len


"""def euclideanBestFitting(trajectory, movelet, spatioTemporalColumns): #nan == end
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
        if np.isnan(el) or len_t>=offset_trajectory:
            break
        len_t += 1

    if len_mov > len_t:
        return euclideanBestFitting(movelet, trajectory, spatioTemporalColumns)



    best_i = -1
    best_score = inf

    for i in range(len_t-len_mov): #scorro traiettoria
        sum = 0
        for j in range(len_mov+1): #scorro movelet
            first_t = [None for x in range(len(spatioTemporalColumns))]
            for k in range(len(spatioTemporalColumns)): #scorro spatioTemporalColumns
                if first_t[k] is None:
                    first_t[k] = trajectory[i+j +k*offset_trajectory]
                    continue #se tutto va bene il 1 dovrebbe essere 0-0

                t = trajectory[i+j +k*offset_trajectory] - first_t[k]
                m = movelet[i+j +k*offset_movelet]
                sum += (t-m)**2

                if sum >= best_score: break

        if sum >= best_score:
            best_score=sum
            best_i=i

    return best_i, best_score**(1/len(spatioTemporalColumns))"""
