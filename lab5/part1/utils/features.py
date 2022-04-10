# All classes methods related to feature functions phi

import numpy as np
from typing import Tuple


class Phi:
    """Class to mimic the phi function which returns the features of a given
    state.  """
    def __init__(self, num_feats: int) -> None:
        self.num_feats = num_feats

    def __getitem__(self, index_: Tuple[int, int]):
        index = index_[0] * index_[1]
        if index >= self.num_feats:
            raise IndexError("Index out of range")
        features = np.zeros(self.num_feats)
        features[index] = 1
        return features