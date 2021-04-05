import math
from typing import NamedTuple

from torch.utils.data import random_split
from torch_geometric.data import Dataset

from data.crystal_benchmark2 import CrystalBenchmark


class MetaDataset(NamedTuple):
    """
    Poorly named class to hold our datasplits, and meta information.
    """
    num_node_features: int
    train: Dataset
    val: Dataset
    test: Dataset


def load_data() -> MetaDataset:
    """
    Loads the dataset. Will take an arg in the future, for now loads the only dataset.
    Also purposely puts aside 40% of the data as unseen, since we are just experimenting for now.
    :return: A train test split. Which is 60%, 20% and 20% of 40% of the data. Poorly explained, but
    see below.
    """
    dataset = CrystalBenchmark()
    num_total_instances = len(dataset)
    all_data_for_now, _ = random_split(dataset, [math.floor(num_total_instances * .6),
                                                 math.ceil(num_total_instances * .4)])
    num_instances_for_now = len(all_data_for_now)
    train, val, test = random_split(all_data_for_now,
                                    [math.ceil(num_instances_for_now * .6),
                                     math.ceil(num_instances_for_now * .2),
                                     math.floor(num_instances_for_now * .2)])

    return MetaDataset(dataset.num_node_features, train, val, test)

