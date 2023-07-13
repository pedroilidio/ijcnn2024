import numpy as np


def faro_shuffle(n_samples, n_features=None):
    n_features = n_features or np.ceil(np.log2(n_samples)).astype(int)
    X = np.empty((n_features, n_samples), dtype=np.float32, order="F")
    X[0] = np.arange(n_samples)
    for i in range(1, n_features):
        np.concatenate((X[i-1][::2], X[i-1][1::2]), out=X[i])
    return X


def halving_shuffle(n_samples, n_features=None):
    n_features = n_features or np.ceil(np.log2(n_samples)).astype(int)
    half = n_samples // 2
    X = np.empty((n_features, n_samples), dtype=np.float32, order="F")
    X[0] = np.arange(n_samples)
    for i in range(1, n_features):
        np.concatenate((X[i-1][half:], X[i-1][:half]), out=X[i])
    return X


def divide_and_reverse(n_samples, n_features=None):
    n_features = n_features or n_samples // 2
    X = np.empty((n_features, n_samples), dtype=np.float32, order="F")
    X[0] = np.arange(n_samples -1, -1, -1)
    for i in range(1, n_features):
        np.concatenate(np.array_split(X[0], i + 1)[::-1], out=X[i])
    # Restore the original order to facilitate sorting
    X[0] = X[0][::-1]
    return X


def binary_tree_shuffle(n_samples, n_features=None):
    n_features = n_features or np.ceil(np.log2(n_samples)).astype(int)
    X = np.empty((n_features, n_samples), dtype=np.float32, order="F")
    _binary_tree_shuffle(
        np.arange(n_samples, dtype=np.float32), X, 0, n_features
    )
    # Restore the original order to facilitate sorting
    X[0] = X[0][::-1]
    return X


def _binary_tree_shuffle(idx, X, depth, n_features):
    if depth > n_features or len(idx) == 1:
        return idx

    half = len(idx) // 2
    return np.concatenate(
        (
            _binary_tree_shuffle(idx[half:], X[:, half:], depth + 1, n_features),
            _binary_tree_shuffle(idx[:half], X[:, :half], depth + 1, n_features),
        ),
        out=X[depth],
    )