import numpy as np
from geolib import geohash
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class TrajectoryTransformer(TransformerMixin):
    """ An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    partitioner : module, default=geohash
        A module able to encode and decode points at different resolutions. The default module is geohash by geolib.
        All custom modules must contains at least the contains encode, decode and bounds function as implementend in geolib.

    precision : number, default=7
        A value used by the partitioner to set the partitions size

    maxLen : number, default=.95
        A value used to limit segments with too many points from the dataset.
        A value < 1 limits the length of the sub-trajectories at the maxLen quantile of the lengths
        A value >= 1 limits the length of the sub-trajectories at maxLen points

    movelet_extractor : module, default=xmeans_module

    distance : function, default=euclidean_optimized
        function to compute distance between movelets and trajectories

    n_jobs : int, default=1
        Max number of concurrent instances

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, partitioner=geohash, precision=7, maxLen=.95, movelet_extractor=None, n_jobs=1):
        self.partitioner = partitioner

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return np.sqrt(X)
