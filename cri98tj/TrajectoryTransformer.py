import pandas as pd
from sklearn.base import TransformerMixin

from cri98tj.distancers.DistancerInterface import DistancerInterface
from cri98tj.partitioners.PartitionerInterface import PartitionerInterface
from cri98tj.selectors.SelectorInterface import SelectorInterface


class TrajectoryTransformer(TransformerMixin):
    def __init__(self, partitioner, selector, distancer):
        assert isinstance(partitioner, PartitionerInterface)
        assert isinstance(selector, SelectorInterface)
        assert isinstance(distancer, DistancerInterface)

        self.partitioner = partitioner
        self.selector = selector
        self.distancer = distancer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_partitions = pd.DataFrame(self.partitioner.fit_transform(X))

        movelets = self.selector.fit_transform(df_partitions.values)

        return self.distancer.fit_transform((df_partitions[df_partitions.columns[:-1]].values, movelets))
