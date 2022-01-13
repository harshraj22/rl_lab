import numpy as np
import unittest


def get_transition_probs():
    """Retuns the probability distribution of the transition from one state to another
    for the given problem statement depending on the result of the roll of the die."""
    # 9 states, and corresponding transition probabilities

    # In general, player remains in the same state iff the dice roll is 3, 4, 5, 6
    same_place_prob = np.full(9, 4/6)
    # state 8 is the terminal state, so all transitions from it go to itself
    # from state 7, all rolls result in same state except 1
    same_place_prob[7:] = [5/6, 1]
    P = np.diag(same_place_prob)

    # fill the general probability transitions
    for i in range(9):
        if i+1 < 9: # if die rolls to 1, it goes to immediate next state
            P[i, i+1] = 1/6
        if i+2 < 9: # if die rolls to 2, it goes to second next state
            P[i, i+2] = 1/6
        # for all other cases, it stays at its own place

    dead_states = [2, 4]
    for i in dead_states:
        P[i,i] = 1 # can't escape from the same place
        if i+1 < 9: # can't go to next state from dead states
            P[i,i+1] = 0
        if i+2 < 9:
            P[i,i+2] = 0

    # from 8, let us assume we can not reach 8. This would help in calculating the probability of reaching 8
    # only from other states
    P[8, 8] = 0

    return P


class TestTransitionProbs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.P = get_transition_probs()

    def test_row_sum_to_one(self):
        """Each row should sum to 1. 
        Note: It doesn't test the last row due to special condition that we defined."""
        for row, _ in enumerate(self.P[:-1]):
            self.assertAlmostEqual(self.P[row].sum(), 1)

    def test_all_items_are_probs(self):
        """Probability value should lie in [0, 1]"""
        for row in self.P:
            for val in row:
                self.assertTrue(0 <= val <= 1)


if __name__ == '__main__':
    unittest.main()
