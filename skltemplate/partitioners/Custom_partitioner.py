from skltemplate.partitioners.PartitionerInterface import PartitionerInterface


class Custom_partitioner(PartitionerInterface):
    """
    partitioner_module : module
    A module able to encode and decode points at different resolutions.
    All custom modules must contains at least the contains encode, decode and bounds function as implementend in geolib.

    precision : number
    A value used by the partitioner to set the partitions size

    maxLen : number
    A value used to limit segments with too many points from the dataset.
    A value < 1 limits the length of the sub-trajectories at the maxLen quantile of the lengths
    A value >= 1 limits the length of the sub-trajectories at maxLen points

    """
    def __init__(self, partitioner_module, precision, maxLen):
        self.partitioner_module = partitioner_module
        self.precision = precision
        self.maxLen = maxLen