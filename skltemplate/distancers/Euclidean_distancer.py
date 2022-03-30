import ctypes
from concurrent.futures import ProcessPoolExecutor
from skltemplate.distancers.distancer_utils import euclideanBestFitting

import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from skltemplate.distancers.DistancerInterface import DistancerInterface
from multiprocessing import Array

from skltemplate.selectors.selector_utils import dataframe_pivot


class Euclidean_distancer(DistancerInterface):

    def __init__(self, n_jobs=1, optimize=True, verbose=True):
        self.verbose = verbose
        self.optimize = optimize
        self.n_jobs = n_jobs

    def fit(self, trajectories_movelets):
        return self

    # trajectories = tid, class, time, c1, c2
    # restituisce nparray con pos0= cluster e poi
    def transform(self, trajectories_movelets):
        trajectories, movelets = trajectories_movelets

        trajectories_df = pd.DataFrame(trajectories, columns=["tid", "class", "time", "c1", "c2"])
        trajectories_df["partId"] = trajectories_df.tid
        df_pivot = dataframe_pivot(df=trajectories_df, maxLen=None, verbose=self.verbose, fillna_value=None)

        distances = np.zeros((df_pivot.shape[0], len(movelets)))

        executor = ProcessPoolExecutor(max_workers=self.n_jobs)

        ndarray_pivot = df_pivot[[x for x in df_pivot.columns if x != "class"]].values
        processes = []
        for i, movelet in enumerate(tqdm(movelets, disable=not self.verbose, position=0)):
            processes.append(executor.submit(self._foo, i, movelet, ndarray_pivot))

        if self.verbose: print(f"Collecting distances from {len(processes)}")
        for i, process in enumerate(tqdm(processes)):
            col = process.result()
            for j, val in enumerate(col):
                distances[j, i] = val

        executor.shutdown(wait=True)
        """for i, movelet in enumerate(tqdm(movelets, disable=not self.verbose, position=0)):
            for j, val in enumerate(self._foo(i, movelet, ndarray_pivot)):
                distances[j, i] = val"""

        return np.hstack((df_pivot[["class"]].values, distances))

    def _foo(self,i, movelet, ndarray_pivot):
        distances = []
        for j, trajectory in enumerate(
                tqdm(ndarray_pivot, disable=True,
                     position=i+1, leave=True)):
            best_i, best_score = euclideanBestFitting(trajectory=trajectory, movelet=movelet)
            distances.append(best_score)

        return distances
