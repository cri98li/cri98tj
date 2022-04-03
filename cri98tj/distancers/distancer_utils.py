from math import sqrt, inf

import numpy as np


def euclideanBestFitting(trajectory, movelet): #nan == end
    len_mov = 0
    for el in movelet:
        if np.isnan(el): break
        len_mov += 1

    len_t = 0
    for el in trajectory:
        if np.isnan(el): break
        len_t += 1

    if len_mov > len_t:
        return euclideanBestFitting(movelet, trajectory)

    if len(trajectory) % 2 != 0:
        raise Exception("la lunghezza della traiettoria non è un numero pari")
    if len(movelet) % 2 != 0:
        raise Exception("la lunghezza della movelet non è un numero pari")
    offset_trajectory = int(len(trajectory) / 2)
    offset_movelet = int(len(movelet) / 2)

    best_i = -1
    best_score = inf
    nullSum = 0
    for i in range(int(len_t/2 - len_mov/2) + 1):
        sum = 0
        for j in range(int(len_mov/2)):
            t_lat = trajectory[i + j]
            m_lat = movelet[j]
            t_lon = trajectory[i + j + offset_trajectory]
            m_lon = movelet[j + offset_movelet]
            sum += (t_lat - m_lat) ** 2 + (t_lon - m_lon) ** 2
            if sum >= best_score:
                break

        if sum < best_score:
            best_score = sum
            best_i = i

    return best_i, sqrt(best_score)
