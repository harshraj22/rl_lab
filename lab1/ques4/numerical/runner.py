import numpy as np
from data_loader.transition_probs import get_transition_probs


if __name__ == '__main__':
    initial_prob = np.zeros(9)
    initial_prob[0] = 1

    transition_probs = get_transition_probs()
    P_8 = np.linalg.matrix_power(transition_probs, 8)

    final_probs = initial_prob.T @ P_8
    print(final_probs)

    print(f'sum of probabilities of all states: {final_probs.sum()}')
