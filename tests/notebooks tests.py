import geolib.geohash
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from tqdm.auto import tqdm
from cri98tj.partitioners.Geohash_partitioner import Geohash_partitioner
from cri98tj.normalizers.FirstPoint_normalizer import FirstPoint_normalizer
from cri98tj.selectors.RandomInformationGain_selector import RandomInformationGain_selector
from cri98tj.distancers.Euclidean_distancer import Euclidean_distancer
from sklearn.model_selection import train_test_split
from cri98tj.distancers.Euclidean_distancer import euclideanBestFitting

if __name__ == '__main__':

    df_original = pd.read_csv('../examples/Vehicles Dataset/data/vehicles_preapred.zip').sort_values(by=["tid", "t"])# precision=5, 50 movelet, DTW
    df_original["c1"] = df_original.c1/100000
    df_original["c2"] = df_original.c2/100000

    df = df_original[["tid", "class", "c1", "c2", "t"]].copy()

    df.head()


    tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                            df.groupby(by=["tid"]).max().reset_index()["class"],
                                                            test_size=.3,
                                                            stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                            random_state=3)

    spatioTemporalCols = ["c1", "c2", "t"]
    n_movelets=50
    n_jobs = 24
    verbose = False

    from cri98tj.selectors.Random_selector import Random_selector
    from sklearn.metrics import accuracy_score
    from cri98tj.distancers.InterpolatedRootDistance_distancer import InterpolatedRootDistance_distancer, \
        InterpolatedRootDistanceBestFitting
    from datetime import datetime

    res = []
    n_mov_r = []
    time = []
    for i in tqdm([2, 5, 10]):  # tqdm(df_res_rig.n.unique()):
        df = df_original[["tid", "class", "c1", "c2", "t"]].copy()

        res.append((.0, .0, .0, .0, .0))
        n_mov_r.append(0)
        time.append(.0)

        for _ in tqdm(range(5)):
            normalizer = FirstPoint_normalizer(spatioTemporalColumns=spatioTemporalCols, fillna=None, verbose=verbose)
            distancer = InterpolatedRootDistance_distancer(normalizer=normalizer, spatioTemporalColumns=spatioTemporalCols,
                                                           n_jobs=n_jobs, verbose=verbose)
            partitioner = Geohash_partitioner(precision=6, spatioTemporalColumns=spatioTemporalCols, verbose=verbose)

            start = datetime.now()
            part = partitioner.fit_transform(df[df.tid.isin(tid_train)].values)
            selector = Random_selector(movelets_per_class=max(1, i // 3), n_jobs=n_jobs,
                                       spatioTemporalColumns=spatioTemporalCols, normalizer=normalizer, verbose=verbose)
            shapelets = selector.fit_transform(part)
            print(shapelets)

            stop = start - datetime.now()

            """_, dist_np = distancer.fit_transform((df.values, shapelets))
            stop = start - datetime.now()
    
            n_mov_r[i-2] += (dist_np.shape[1])
    
            clf = RandomForestClassifier(max_depth=3, random_state=3, n_jobs=n_jobs, n_estimators=5000)
    
            dist_np_df = pd.DataFrame(dist_np)
            X = dist_np_df.drop(columns=[0]).values
            y = dist_np_df[0].values
    
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=3)
    
            clf.fit(X_train, y_train)
    
            y_pred = clf.predict(X_test)
    
            res[i-2] = tuple(a+b for a, b in zip(compute_measures(y_test, y_pred), res[i-2]))"""
            time[i - 2] += stop.total_seconds() * 1000  # millisecondi

        res[i - 2] = list(map(lambda x: x / 5, res[i - 2]))
        n_mov_r[i - 2] /= 5
        time[i - 2] /= 5