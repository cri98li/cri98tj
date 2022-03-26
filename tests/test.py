import pandas as pd
from skltemplate.partitioners.Geohash_partitioner import Geohash_partitioner
from skltemplate.selectors.XMeans_selector import XMeans_selector
from skltemplate.distancers.Euclidean_distancer import Euclidean_distancer

df = pd.read_csv(
    '..\\..\\Explaining-Any-Trajectory-Classifier\\Classificatore shapelets\\cabs_preapred.zip').head(1000)

gh = Geohash_partitioner()
df_gh = pd.DataFrame(gh.fit_transform(df.values))

print(df_gh.head())

xms = XMeans_selector()
movelets = xms.fit_transform(df_gh.values)

print(pd.DataFrame(movelets).head())

ed = Euclidean_distancer()
dist_np = ed.fit_transform((df.values, movelets))
print(pd.DataFrame(dist_np).head())

#%%
