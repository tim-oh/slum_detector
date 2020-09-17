import pickle
import numpy as np
from matplotlib import pyplot as plt
import imageio


def scale_distance_pkl(distance_pkl, dist_thresh=0.5):
    """
    Scale pseudo-probability values of slums-world predictions to pixel classes.

    The 'distance_estimated' of slums-world is actually the probability of being slum (dist > .05 = slum).
    This function scales those values to the repo convention of marking slums as >63.
    Use of uint8 data type avoids scaling of values to max=255 by imageio.imwrite.

    :param distance_pkl: Output of slums-world in terms of estimated distances to slum boundary.
    :return: Scaled pixel values.
    """
    dist_raw = np.array(distance_pkl)
    factor = 63 / dist_thresh
    dist_scaled = dist_raw * factor
    dist_scaled = dist_scaled.astype('uint8')
    return dist_scaled


def load_slums_world_pkl(input_path, output_path, show=False):
    """
    Load pkl of slums_world prediction, convert values to correct scale and save as png, with optional visualisation.

    When show=True, the histograms shows the distribution of prediction values and the respective converted values.
    The first image plot shows the original estimated distances, the second image plot shows the resulting slum map.

    Usage:
    load_slums_world_pkl('PATH/TO/input.pkl', 'PATH/TO/output.png', show=True)

    :param input_path: Path to .pkl containing distance_estimated.
    :param output_path: File name path to save output to.
    :param input_path: Visualise original, converted values and resulting slum mask.
    """
    with open(input_path, 'rb') as f:
        distance_pkl = pickle.load(f, encoding="bytes")
    scaled_distance = scale_distance_pkl(distance_pkl)
    imageio.imwrite(output_path, scaled_distance)
    if show:
        plt.hist(distance_pkl)
        plt.show()
        plt.hist(scaled_distance)
        plt.show()
        plt.imshow(scaled_distance)
        plt.show()
        plt.imshow(scaled_distance > 63)
        plt.show()

