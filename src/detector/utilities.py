import pickle
import numpy as np
from matplotlib import pyplot as plt
import imageio


def scale_distance_pkl(distance_pkl, dist_thresh=0.5):
    """
    The 'distance_estimated' of slums-world is actually the probability of being slum (dist > .05 = slum). This \
    function scales those values to the repo convention of marking slums as >63.

    :param distance_pkl: Output of slums-world in terms of estimated distances to slum boundary.
    :return: Scaled pixel values.
    """
    dist_raw = np.array(distance_pkl)
    factor = 63 / dist_thresh
    dist_scaled = dist_raw * factor
    dist_scaled = dist_scaled.astype('uint8') # To avoid scaling to 255
    plt.hist(dist_scaled)
    plt.show()
    return dist_scaled


# Usage example to run scale_distance_pkl, including visual illustration via plotting:
# pkl_path = './../predictions/slums-world_17082020/mulimage3_predestimated_distance.pkl'
# with open(pkl_path,'rb') as f:
#     distance_pkl = pickle.load(f, encoding="bytes")
# plt.hist(distance_pkl)
# plt.show()
# scaled_distance = scale_distance_pkl(distance_pkl)
# plt.imshow(scaled_distance)
# plt.show()
# plt.imshow(scaled_distance > 63)
# plt.show()
#
# png_path = './../predictions/slums-world_17082020/pred_y.png'
# imageio.imwrite(png_path, scaled_distance)



