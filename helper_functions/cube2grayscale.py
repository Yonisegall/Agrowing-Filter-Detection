from logging import exception

import numpy as np
from sklearn.decomposition import PCA

def cube2grayscale(stack, mode='mean'):
    if mode == "mean":
        return np.mean(stack, axis=2)

    elif mode == "weighted_mean":
        weights = np.array([1]*10+[2]+[1]*3)  # length 14
        assert len(weights) == 14 , "Expected 14 weights"
        weights = weights / weights.sum()
        return np.tensordot(stack, weights, axes=([2], [0]))

    elif mode == 'pca': # todo: check this works
        h, w, b = stack.shape
        reshaped = stack.reshape(-1, b)
        pca = PCA(n_components=1)
        grayscale = pca.fit_transform(reshaped)
        return grayscale.reshape(h, w)

    else:
        raise Exception("average mode not selected")