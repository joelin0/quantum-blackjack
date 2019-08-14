"""
File for going from blackjack configuration to rotation angles

Mirrors the example in Section VII of the paper
"""

from blackjack import *
from numpy.linalg import cholesky, norm
import numpy as np
from math import atan2, sqrt

if __name__ == '__main__':
    # face up: 1, 9, 10
    # possible down cards: 1, 8, 10
    ua, ub, ud = 1, 9, 10
    shoe = {1: 3, 8: 1, 9: 1, 10: 2}
    C = get_C_matrix(ua, ub, ud, shoe)
    nonzero_rows = C[np.any(C, axis=1)]
    nonzero = nonzero_rows[:, ~np.all(nonzero_rows == 0, axis=0)]
    # note the dimensions, in order, correspond to card 1, 8, 10
    S_block, alpha_vector, D_block, X_block, Y_block = get_biased_hyperbit_sdp_discrete(nonzero)
    X = cholesky(X_block)  # x vectors stored in ROWS, i.e. X[0], X[1], X[2]

    Y = []  # y vectors stored in ROWS, i.e. Y[0], Y[1], Y[2]
    for t in range(3):
        numerator = np.sum([nonzero[s, t] * X[s] for s in range(3)], axis=0)
        Y.append(numerator / norm(numerator))

    alice_thetas = [atan2(X[s][1], X[s][0]) for s in range(3)]

    bob_thetas = [atan2(-Y[s][1], Y[s][0]) for s in range(3)]

    alice_phis = [atan2(abs(X[s][2]), sqrt(X[s][0] ** 2 + X[s][1] ** 2)) for s in range(3)]

    bob_phis = [atan2(abs(Y[s][2]), sqrt(Y[s][0] ** 2 + Y[s][1] ** 2)) for s in range(3)]

    print(alice_thetas)

    print(alice_phis)

    print(bob_thetas)

    print(bob_phis)
