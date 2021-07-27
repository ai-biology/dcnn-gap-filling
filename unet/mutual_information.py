"""
Compute discrete mutual information
"""

import numpy as np


def mutual_information_discrete(X, Y):
    """ Compute discrete mutual information """
    assert np.issubdtype(X.dtype, np.integer), "X must have discrete dtype"
    assert np.issubdtype(Y.dtype, np.integer), "Y must have discrete dtype"

    # flatten X
    X = X.reshape(len(X), -1)

    # compute p(y)
    y_space, y_counts = np.unique(Y, return_counts=True)
    y_freqs = y_counts / y_counts.sum()
    assert np.array_equal(y_space, [0, 1]), "Y must be binary"

    # compute p(x)
    x_space, x_inv, x_counts = np.unique(
        X, return_inverse=True, return_counts=True, axis=0
    )
    x_freqs = x_counts / x_counts.sum()

    # compute conditionals p(x | y)
    x_given_p_1_space, x_given_p_1_idx, x_given_p_1_counts = np.unique(
        X[Y == 1], return_index=True, return_counts=True, axis=0
    )
    x_given_p_1_freqs = x_given_p_1_counts / x_given_p_1_counts.sum()

    x_given_p_0_space, x_given_p_0_idx, x_given_p_0_counts = np.unique(
        X[Y == 0], return_index=True, return_counts=True, axis=0
    )
    x_given_p_0_freqs = x_given_p_0_counts / x_given_p_0_counts.sum()

    # compute joint p(x, y)
    x_p_1_freqs = x_given_p_1_freqs * y_freqs[1]
    x_p_0_freqs = x_given_p_0_freqs * y_freqs[0]

    # compute idxs in x_space and x_freqs that correspond to freqs in x_p_0/1_freqs
    x_p_1_freqs_idx = x_inv[np.flatnonzero(Y == 1)[x_given_p_1_idx]]
    x_p_0_freqs_idx = x_inv[np.flatnonzero(Y == 0)[x_given_p_0_idx]]

    # test x_p_1_freqs_idx
    assert (x_space[x_p_1_freqs_idx] == x_given_p_1_space).all()
    assert (x_space[x_p_0_freqs_idx] == x_given_p_0_space).all()

    # verify densities sum to 1
    assert np.isclose(np.sum(y_freqs), 1)
    assert np.isclose(np.sum(x_freqs), 1)
    assert np.isclose(np.sum(x_p_1_freqs) + np.sum(x_p_0_freqs), 1)

    # sum over joint to get MI
    mi_given_p_0 = np.sum(
        x_p_0_freqs * np.log(x_p_0_freqs / (x_freqs[x_p_0_freqs_idx] * y_freqs[0]))
    )
    mi_given_p_1 = np.sum(
        x_p_1_freqs * np.log(x_p_1_freqs / (x_freqs[x_p_1_freqs_idx] * y_freqs[1]))
    )

    return mi_given_p_0 + mi_given_p_1


def entropy_discrete(X, return_space=False):
    """ Compute discrete entropy """
    assert np.issubdtype(X.dtype, np.integer), "X must have discrete dtype"

    # flatten X
    X = X.reshape(len(X), -1).astype(np.uint8)

    x_space, x_counts = np.unique(X, return_counts=True, axis=0)
    x_freqs = x_counts / x_counts.sum()
    x_log_freqs = np.log(x_freqs)
    entropy = -np.sum(x_freqs * x_log_freqs)

    if return_space:
        return entropy, x_space

    return entropy


def proficiency_discrete(X, Y):
    """ Compute discrete proficiency """
    return mutual_information_discrete(X, Y) / entropy_discrete(X)
