import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def fcm(X, n_clusters, m, max_iter, error_threshold=1e-4, metric='euclidean', v0=None):    
    n_samples, n_features = X.shape
    
    # Initialize cluster centers randomly
    # v = X[np.random.choice(n_samples, n_clusters, replace=False)]
    if not v0:
        u0 = np.random.rand(n_samples, n_clusters)
        u0 /= np.sum(u0, axis=1)[:, np.newaxis]
        u = u0.copy()
    elif v0 == 'random_choice':
        v = X[np.random.choice(n_samples, n_clusters, replace=False)]
    else: 
        v = v0

    for iteration in range(max_iter):
        # Compute the fuzzt center vectors
        if v0 is None:
            v = (u ** m).T.dot(X) / ((u**m).sum(axis=0)[np.newaxis,:]).T
        elif v0 is not None and iteration !=0:
            v = (u ** m).T.dot(X) / ((u**m).sum(axis=0)[np.newaxis,:]).T

        # Compute the membership matrix U
        d = cdist(X, v, metric=metric)

        # change the zero distances for a very small number
        d = np.fmax(d, 1e-16)

        exp = 2. / (m - 1)
        d2 = 1/ (d ** exp)

        u = d2 / np.sum(d2, axis=1)[:,np.newaxis]

        # Check for convergence
        error = np.sqrt(np.sum(np.square(u - u0)))
        if error < error_threshold:
            break
        u0 = u.copy()

    return u, v, d

