"""
This file deals with calculating optimal strategy matrices S, given game parameters C.
"""
import picos as pic
from random import random
import numpy as np


def get_unlimited_comm_direct(C_arr):
    """
    Optimizes over strategies with no restriction on S_st (i.e. unlimited communication/perfect information).

    :param np.ndarray C_arr: the coefficient C matrix of the game
    :return: The S matrix
    :rtype: np.array
    """
    return np.sign(C_arr)


def get_unlimited_comm_sdp(C_arr):
    """
    Optimizes over strategies with no restriction on S_st (i.e. unlimited communication/perfect information).

    For completeness, we use an SDP solver; however, S can be trivially calculated by taking
    S_st = sgn C_st

    :param np.ndarray C_arr: the coefficient C matrix of the game
    :return: The S matrix
    :rtype: np.array
    """
    m, n = C_arr.shape

    P = pic.Problem()
    C = pic.new_param('C', C_arr)
    S = P.add_variable('S', (m, n))

    IJ = [(i, j) for i in range(m) for j in range(n)]

    P.set_objective('max', pic.sum([C[ij] * S[ij] for ij in IJ], "IJ"))

    P.add_list_of_constraints([S[ij] < 1 for ij in IJ], 'IJ')
    P.add_list_of_constraints([S[ij] > -1 for ij in IJ], 'IJ')

    P.solve(verbose=0)

    return np.array(P.get_valued_variable("S"))


def get_classical_discrete(C_arr):
    """
    Optimizes over strategies constrained to single classical bit communication, i.e. S matrix of the form
    
    S_st = p_s * alpha_t + (1 - p_s) * beta_t

    :param np.ndarray C_arr: the coefficient C matrix of the game
    :return: The S matrix, and p, alpha, and beta vectors
    :rtype: tuple of np.ndarray
    """
    m, n = C_arr.shape
    best_score = -float('inf')
    S_best = np.empty((m, n))
    alpha_best, beta_best, p_best = None, None, None
    for i in xrange(2 ** m):
        p_guess = [((i >> j) % 2) for j in xrange(m)]
        alpha_guess = [1 if sum(C_arr[a][b] * p_guess[a] for a in xrange(m)) > 0 else -1 for b in xrange(n)]
        beta_guess = [1 if sum(C_arr[a][b] * (1 - p_guess[a]) for a in xrange(m)) > 0 else -1 for b in xrange(n)]
        S_guess = np.array([[p_guess[a] * alpha_guess[b] + (1 - p_guess[a]) * beta_guess[b] for b in xrange(n)]
                            for a in xrange(m)])
        score = get_payout(C_arr, S_guess)
        if score > best_score:
            S_best = S_guess
            alpha_best, beta_best, p_best, best_score = alpha_guess, beta_guess, p_guess, score

    return S_best, np.array(p_best), np.array(alpha_best), np.array(beta_best)


def get_unbiased_hyperbit_sdp(C_arr):
    """
    Optimizes over strategies of the form

    S_st = x_s \cdot y_t

    i.e. pure hyperbit strategies.

    The Gramian matrix is blocked out as
         X^TX | X^TY
     G = ------------
         Y^TX | Y^T Y


    :param np.ndarray C_arr: the coefficient matrix of the game
    :return: The S matrix (X^TY block), the X^TX block, and the Y^TY block
    :rtype: tuple of np.arrays

    """
    m, n = C_arr.shape

    P = pic.Problem()
    C = pic.new_param('C', C_arr)
    S = P.add_variable('S', (m, n))
    X = P.add_variable('X', (m, m), vtype='symmetric')
    Y = P.add_variable('Y', (n, n), vtype='symmetric')

    IJ = [(i, j) for i in range(m) for j in range(n)]

    # While this is an explicit way to take the Hadamard product (elementwise), this can also be
    # achieved using the '^' operator
    P.set_objective('max', pic.sum([C[ij] * S[ij] for ij in IJ], "IJ"))

    P.add_constraint(((X & S) // (S.H & Y)) >> 0)

    P.add_list_of_constraints([X[i, i] == 1 for i in range(m)], 'i')
    P.add_list_of_constraints([Y[i, i] == 1 for i in range(n)], 'i')

    P.solve(verbose=0)

    return np.array(P.get_valued_variable("S")), \
           np.array(P.get_valued_variable("X")), \
           np.array(P.get_valued_variable("Y"))


def get_biased_hyperbit_sdp_discrete(C_arr):
    """
    Optimizes over strategies of the form

    S_st = gamma_t + x_s \cdot y_t

    i.e. optional hyperbit strategies, where Bob can choose to use or ignore the hyperbit measurement.

    The Gramian matrix is blocked out as
         X^TX | X^TY
     G = ------------
         Y^TX | Y^T Y


    :param np.ndarray C_arr: the coefficient matrix of the game
    :return: The S matrix, the gamma vector, the X^TY block, the X^TX block, and the Y^TY block
    :rtype: tuple of np.arrays

    """
    m, n = C_arr.shape

    gamma_guess_0 = [1 if sum(C_arr[a][b] for a in xrange(m)) > 0 else -1 for b in xrange(n)]
    S_best, gamma_best, D_best, X_best, Y_best = None, None, None, None, None
    best_score = -float('inf')
    for i in xrange(2**n):
        gamma_guess = [gamma_guess_0[j] if ((i >> j) % 2 == 1) else 0 for j in xrange(n)]
        C_reduced = C_arr[:, [b for b in xrange(n) if gamma_guess[b] == 0]]
        if C_reduced.shape[1] > 0:
            D, X, Y = get_unbiased_hyperbit_sdp(C_reduced)
        S_guess = np.empty((m, n))
        D_col = 0
        for b in xrange(n):
            D_flag = (gamma_guess[b] == 0)
            for a in xrange(m):
                if D_flag:
                    S_guess[a][b] = D[a][D_col]
                else:
                    S_guess[a][b] = gamma_guess[b]
            if D_flag:
                D_col += 1
        score = get_payout(C_arr, S_guess)
        if score > best_score:
            S_best, gamma_best, D_best, X_best, Y_best, best_score = S_guess, gamma_guess, D, X, Y, score

    return S_best, gamma_best, D_best, X_best, Y_best


#### Tools ####
def get_payout(C, S):
    """
    Calculate
    I(S) = <C, S> = tr(C^TS)

    :param np.ndarray C: the coefficient matrix
    :param np.ndarray S: the strategy matrix
    :return: <C, S>
    :rtype: float
    """
    # in numpy, '*' is elementwise multiplication
    return np.sum(C*S)


def generate_random_C_matrix(m=None, n=None, sign_matrix=None):
    """
    Helper function to generate random matrices of a given dimension, and possibly a given sign matrix.

    Note at least one dimension argument must be specified.

    :param int m: the number of rows in the matrix. If None specified, m = n.
    :param int n: the number of columns in the matrix. If None specified, n = m.
    :param np.ndarray sign_matrix: the sign matrix of C. Must have elements equal to +/-1.
    :return: A random m x n matrix. If optimal is not specified,
             each entry is uniformly randomly selected from (-1, 1).
             If optimal is specified, C[a, b] is uniformly selected from between 0 and optimal[a, b].
             optimal is most useful as a matrix that specifies signs, i.e. a matrix of +/-1.
    :rtype: np.ndarray
    """
    if sign_matrix is not None:
        m, n = sign_matrix.shape
    elif m is None and n is not None:
        m = n
    elif n is None and m is not None:
        n = m
    elif m is None and n is None:
        raise ValueError

    C = [[0 for _ in xrange(n)] for _ in xrange(m)]
    for a in xrange(m):
        for b in xrange(n):
            Cab = random()
            if sign_matrix is not None:
                Cab *= sign_matrix[a][b]
            else:
                Cab = 2 * Cab - 1
            C[a][b] = Cab
    return np.array(C)
