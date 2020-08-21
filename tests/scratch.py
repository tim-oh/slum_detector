import numpy as np
import numpy.ma as ma
import imageio
import warnings
from tabulate import tabulate
import src.detector.data_prep


def convert_features(feature_array):
    if np.unique(feature_array).shape == (1,):
        warnings.warn("Feature values: all elements are set to %r." % feature_array[0, 0], UserWarning)
    valid = np.arange(0, 256)
    if not np.isin(feature_array, valid).all():
        raise ValueError("Feature values: pixel range not in %r." % valid)
    if np.unique(feature_array).shape == (1,):
        warnings.warn("Feature values: all elements are set to %r." % feature_array[0, 0], UserWarning)
    if not feature_array.shape[2] == 3:
        raise ValueError("Feature shape: png should have 3 channels, but shape %r." % feature_array.shape)
    return feature_array


