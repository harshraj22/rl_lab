from abc import ABC, abstractmethod
from typing import List, Tuple


class ArmRewardInitilizer(ABC):
    @abstractmethod
    def initialize(self, num_arms: int) -> Tuple[List, int]:
        """Initialize the arms with the reward setting and return the optimal
        arm index.

        Parameters
        ----------
        num_arms: int
            number of arms to be initialized

        Returns
        -------
        tuple(List, int): A tuple, with first element representing the reward
            objects corresponding to each arm, and the second element being the
            index of the optimal arm. The reward objects should have a method
            'sample' to sample the reward for the corresponding arm, from the
            underlying distribution of the reward as they have been initilized.
        """
        pass



    # @abstractmethod
    # def sample(self):
    #     pass