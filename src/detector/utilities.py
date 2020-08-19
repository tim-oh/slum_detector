import pickle
import numpy as np
from matplotlib import pyplot as plt
import imageio

# NOTE: There's an issue here in that 'distance_estimated' is actually the probability of being slum (dist > .05 = slum)
# That shouldn't be so, but it can easily be scaled, as per the blow.

pkl_path = './../predictions/slums-world_17082020/mulimage3_predestimated_distance.pkl'
with open(pkl_path,'rb') as f:
    distance_pkl = pickle.load(f, encoding="bytes")
plt.hist(distance_pkl)
plt.show()

def scale_distance_pkl(distance_pkl, dist_thresh=0.5, maxval=127, n_positive=[]):
    """

    :param distance_pkl: Output of slums-world in terms of estimated distances to slum boundary.
    :param maxval: Linearly scale pixel distances to this maximum. Default=127 scales to the number of pixel classes.
    :param n_positive: Linearly scale pixel values so that n_positive lie above a threshold of 63. Truncated at 255.
    :return: Scale pixel values
    """
    dist_raw = np.array(distance_pkl)
    factor = 63 / dist_thresh
    dist_scaled = dist_raw * factor
    dist_scaled = dist_scaled.astype('uint8') # To avoid scaling to 255
    #
    # if dist_thresh == 0.5:
    #
    # elif not maxval == 127:
    #     factor = maxval / np.max(dist_raw)
    #     dist_scaled = dist_raw * factor
    # elif not n_positive == []: # This function is imprecise, don't use(?)
    #     dist_sorted = np.sort(dist_raw).reshape(-1)
    #     lower_bound = dist_sorted[-n_positive]
    #     factor = 63 / lower_bound
    #     dist_scaled = dist_raw * factor
    #     dist_scaled[dist_scaled > 255] = 255
    # else:
    #     slum = dist_raw > 0.5
    #     dist_scaled = slum * 127
    plt.hist(dist_scaled)
    plt.show()
    return dist_scaled

scaled_distance = scale_distance_pkl(distance_pkl)
plt.imshow(scaled_distance)
plt.show()
plt.imshow(scaled_distance > 63)
plt.show()

png_path = './../predictions/slums-world_17082020/pred_y.png'
imageio.imwrite(png_path, scaled_distance)



