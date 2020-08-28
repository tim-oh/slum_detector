import numpy as np
import numpy.ma as ma
import imageio
import warnings
from tabulate import tabulate
import src.detector.data_prep


# TODO: Eliminate the loop, perhaps with a meshgrid() style approach
def tile(features, tile_size):
    """

    :param features:
    :param tile_size:
    :return:
    """
    i_upperleft = np.arange(0, features.shape[0], tile_size[0])
    j_upperleft = np.arange(0, features.shape[1], tile_size[1])
    n_coordinates = len(i_upperleft) * len(j_upperleft)
    coordinates = np.zeros((2, 2, n_coordinates))
    counter = 0
    for i in i_upperleft:
        for j in j_upperleft:
            coordinates[0, :, counter] = [i, j]
            coordinates[1, :, counter] = [i + tile_size[0] - 1, j + tile_size[1] - 1]
            counter += 1
    return coordinates


def features_6x12_masknone():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
    ])
    block_digits_rgb = np.dstack([block_digits] * 3)
    mask = np.zeros(block_digits_rgb.shape)
    masked_blocks = ma.masked_array(block_digits_rgb, mask=mask)
    return masked_blocks



def features_6x12_mask_topleft6x4():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
    ])
    block_digits_rgb = np.dstack([block_digits] * 3)
    mask= np. array([
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    mask_rgb = np.dstack([mask] * 3)
    masked_blocks = ma.masked_array(block_digits_rgb, mask=mask_rgb)
    return masked_blocks




def coordinates_6x12_tile_3x3_mask_none():
    coordinates = np.dstack((
        [[0, 0], [2, 2]],
        [[0, 3], [2, 5]],
        [[0, 6], [2, 8]],
        [[0, 9], [2, 11]],
        [[3, 0], [5, 2]],
        [[3, 3], [5, 5]],
        [[3, 6], [5, 8]],
        [[3, 9], [5, 11]]
    ))
    return coordinates



def coordinates_6x12_tile_2x2_mask_none():
    coordinates = np.dstack((
        [[0, 0], [1, 1]],
        [[0, 2], [1, 3]],
        [[0, 4], [1, 5]],
        [[0, 6], [1, 7]],
        [[0, 8], [1, 9]],
        [[0, 10], [1, 11]],
        [[2, 0], [3, 1]],
        [[2, 2], [3, 3]],
        [[2, 4], [3, 5]],
        [[2, 6], [3, 7]],
        [[2, 8], [3, 9]],
        [[2, 10], [3, 11]],
        [[4, 0], [5, 1]],
        [[4, 2], [5, 3]],
        [[4, 4], [5, 5]],
        [[4, 6], [5, 7]],
        [[4, 8], [5, 9]],
        [[4, 10], [5, 11]]
    ))
    return coordinates

def coordinates_6x12_tile_3x3_mask_topleft():
    coordinates = np.dstack((
        [[0, 6], [2, 8]],
        [[0, 9], [2, 11]],
        [[3, 0], [5, 2]],
        [[3, 3], [5, 5]],
        [[3, 6], [5, 8]],
        [[3, 9], [5, 11]]
    ))
    return coordinates


def coordinates_6x12_tile_2x2_mask_topleft():
    coordinates = np.dstack((
        [[0, 6], [1, 7]],
        [[0, 8], [1, 9]],
        [[0, 10], [1, 11]],
        [[2, 6], [3, 7]],
        [[2, 8], [3, 9]],
        [[2, 10], [3, 11]],
        [[4, 0], [5, 1]],
        [[4, 2], [5, 3]],
        [[4, 4], [5, 5]],
        [[4, 6], [5, 7]],
        [[4, 8], [5, 9]],
        [[4, 10], [5, 11]]
    ))
    return coordinates




a = features_6x12_masknone()
b = coordinates_6x12_tile_3x3_mask_none()
c = read_tiles(a, b)

@pytest.fixture
def features_6x12_tiled_3x3():
    tiles = np.stack((
        np.dstack([np.ones((3, 3))] * 3),
        np.dstack([np.ones((3, 3)) * 2] * 3),
        np.dstack([np.ones((3, 3)) * 3] * 3),
        np.dstack([np.ones((3, 3)) * 4] * 3),
        np.dstack([np.ones((3, 3)) * 5] * 3),
        np.dstack([np.ones((3, 3)) * 6] * 3),
        np.dstack([np.ones((3, 3)) * 7] * 3),
        np.dstack([np.ones((3, 3)) * 8] * 3)), axis=3)
    return tiles
assert np.array_equal(c, d)