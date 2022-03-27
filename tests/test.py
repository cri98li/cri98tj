import pandas as pd

from skltemplate.distancers.Euclidean_distancer import Euclidean_distancer
from skltemplate.partitioners.Geohash_partitioner import Geohash_partitioner
from skltemplate.selectors.OPTICS_selector import OPTICS_selector
from skltemplate.selectors.XMeans_selector import XMeans_selector
from skltemplate.prebuilded_transformers.GeohashXmensEuclideian_prebuild import gxe_prebuild
from skltemplate.TrajectoryTransformer import TrajectoryTransformer

df = pd.read_csv(
    '..\\..\\Explaining-Any-Trajectory-Classifier\\Classificatore shapelets\\cabs_preapred.zip')

perc = .01
max_tid = df.tid.max()
df = df[df.tid < max_tid*perc]



gh = Geohash_partitioner()

xms = OPTICS_selector()#XMeans_selector(kmax=5)

ed = Euclidean_distancer()

dist_np = TrajectoryTransformer(partitioner=gh, selector=xms, distancer=ed).fit_transform(df.values)


#dist_np = gxe_prebuild(kmax=5).fit_transform(df.values)



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=None, random_state=0, n_jobs=-1)

splitPoint = round(len(dist_np) * .8)

X_train, X_test, y_train, y_test = dist_np[:splitPoint, 1:], dist_np[splitPoint:, 1:], dist_np[:splitPoint,
                                                                                       :1].astype('int'), dist_np[splitPoint:, :1].astype('int')

clf.fit(X_train, y_train)

from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
# %%
