import numpy as np
from data_loader.transition_probs import get_transition_probs
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

np.set_printoptions(linewidth=np.inf)

if __name__ == '__main__':
    initial_prob = np.zeros(9)
    initial_prob[0] = 1
    probs_sum = 0

    transition_probs = get_transition_probs()
    current_prob = np.identity(n = 9, like=transition_probs)
    INFINITY = 10**7

    for i in tqdm(range(INFINITY), desc='Summing prob of ending at 8th state in iteration'):
        current_prob = current_prob @ transition_probs
        probs_sum += (initial_prob @ current_prob)[8]

    print(f'Probability of ending at state 8 is: {probs_sum}')

