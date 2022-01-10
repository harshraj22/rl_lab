import numpy as np

# 9 states, and corresponding transition probabilities
same_place_prob = np.full(9, 4/6)
same_place_prob[7] = 5/6
same_place_prob[8] = 1
P = np.diag(same_place_prob)

# fill general probs
for i in range(9):
    if i+1 < 9:
        P[i, i+1] = 1/6
    if i+2 < 9:
        P[i, i+2] = 1/6

for row, _ in enumerate(P):
    for col, val in enumerate(_):
        print(f'Prob from ({row} -> {col}) is {val}')

dead_states = [2, 4]
for i in dead_states:
    P[i,i] = 1
    if i+1<9:
        P[i,i+1] = 0
    if i+2<9:
        P[i,i+2] = 0

# correct for index 2, 4
# print(np.arange(9))
with np.printoptions(linewidth=np.inf):
    print(P)

    print(P.sum(axis=1))
