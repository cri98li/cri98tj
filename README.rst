Cri98tj - A "modular" transformer for trajectories data
============================================================

.. _scikit-learn: https://scikit-learn.org

Cri98tj is a project developed using scikit-learn guidelines. At the end of the development will be made the request for integration in the toolkit.

The advantages of this transformer lie in the higher speed of computation compared to others such as Movelets or MasterMovelets. Also, unlike black-box like Rocket, the transformations applied to the data can be interpreted by humans.

The basic idea is inspired by shapelets: you try to extract meaningful segments from trajectories and then calculate the best-fitting distance on new trajectories and use the resulting matrix as input to a classifier.

At present the transformer is an order of magnitude faster than movelet and negligibly less accurate than Rocket.

Also in this repository are prototypes of other classifiers with similar ideas:
 - Modification of algorithms such as MrSQM to convert trajectories into multiple symbol sequences (via sax or geohash), find the most frequent subsequences, and classify the data using the occurrence matrix
 - Conversion of trajectories into identified symbol sequences using ad-hoc clustering algorithms. Similar to before, we search for the most discriminating symbol sequences and use the occurrence matrix as input to a classifier

.. _documentation: https://sklearn-template.readthedocs.io/en/latest/quick_start.html

Documentation: not available yet
