import numpy as np
import numpy.ma as ma
import imageio
import warnings
from tabulate import tabulate
import src.detector.data_prep

def labels_6x12_masked_6x4_tiled_3x3():
    tiles_3d = np.stack((
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 0]],
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
        [[1, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]), axis=2)
    tiles_4d = tiles_3d[:, :, np.newaxis, :]
    lowlefttiles = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    tile_mask = np.stack((
        np.ones((3, 3)),
        np.ones((3, 3)),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        lowlefttiles,
        lowlefttiles,
        np.zeros((3, 3)),
        np.zeros((3, 3))), axis=2)
    tile_mask = tile_mask[:, :, np.newaxis, :]
    masked_tiles = ma.masked_array(tiles_4d, mask=tile_mask)
    return masked_tiles

def labels_6x12_masked_6x4_cleaned_tiled_3x3():
    tiles_3d = np.stack((
        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 0]],
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
        [[1, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]), axis=2)
    tiles_4d = tiles_3d[:, :, np.newaxis, :]
    lowlefttiles = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    tile_mask = np.stack((
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        lowlefttiles,
        lowlefttiles,
        np.zeros((3, 3)),
        np.zeros((3, 3))), axis=2)
    tile_mask = tile_mask[:, :, np.newaxis, :]
    masked_tiles = ma.masked_array(tiles_4d, mask=tile_mask)
    return masked_tiles

def features_6x12_mask_topleft6x4_tiled_3x3():
    tile_data = np.stack((
        np.dstack([np.ones((3, 3))] * 3),
        np.dstack([np.ones((3, 3)) * 2] * 3),
        np.dstack([np.ones((3, 3)) * 3] * 3),
        np.dstack([np.ones((3, 3)) * 4] * 3),
        np.dstack([np.ones((3, 3)) * 5] * 3),
        np.dstack([np.ones((3, 3)) * 6] * 3),
        np.dstack([np.ones((3, 3)) * 7] * 3),
        np.dstack([np.ones((3, 3)) * 8] * 3)), axis=3)
    lowlefttiles = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    tile_mask = np.stack((
        np.dstack([np.ones((3, 3))] * 3),
        np.dstack([np.ones((3, 3))] * 3),
        np.dstack([np.zeros((3, 3))] * 3),
        np.dstack([np.zeros((3, 3))] * 3),
        np.dstack([lowlefttiles] * 3),
        np.dstack([lowlefttiles] * 3),
        np.dstack([np.zeros((3, 3))] * 3),
        np.dstack([np.zeros((3, 3))] * 3)), axis=3)
    masked_tiles = ma.masked_array(tile_data, mask=tile_mask)
    return masked_tiles


# Note: A more memory-efficient version of this function could output tile indices based on tiles.shape instead of tiles
def split_tiles(n_tiles, splits=(0.6, 0.2, 0.2)):
    """
    Generates randomly sampled indices of train, validation and test sets given a number of tiles and set proportions.

    :param tiles: Number of tiles to be split into datasets.
    :param splits: Tuple of dataset proportions allocated to (training, validation, test) sets. Must sum to 1.
    :return: Lists of tile indices for each of the split [training], [validation], [test] sets, in that order.
    """
    splits = np.array(splits)
    if splits.any() < 0 or splits.any() > 1:
        raise ValueError(
            f"Split values: dataset proportions must range from 0 and 1 but are {str(splits)!r}.")
    if splits[0] + splits[1] + splits[2] != 1:
        raise ValueError(
            f"Split values: proportions must add up to 1 but are {str(splits)!r}.")
    n_train = round(splits[0] * n_tiles)
    train_indices = np.random.choice(n_tiles, n_train, replace=False)
    indices_remain = np.delete(np.arange(n_tiles), train_indices)
    if splits[2] > 0:
        n_test = round(splits[2] / (splits[1] + splits[2]) * len(indices_remain))
        test_indices = np.random.choice(indices_remain, n_test, replace=False)
    else:
        test_indices = np.array([], dtype=int)
    val_indices = np.delete(np.arange(n_tiles), np.concatenate((train_indices, test_indices)))
    return train_indices, val_indices, test_indices

def stratified_split(tiles, slum_tiles, splits):
    """
    Random split of image tiles into training, validation and test sets, according to 'splits' proportions.
    Stratification according to 'slum_tiles' marker. Output needs to be shuffled prior to training.

    :param tiles: Array of N image tiles of format (x, y, channels, N), to be split.
    :param slum_tiles: Boolean array of length N that marks slum tiles along the 4th tile array dimension.
    :param splits: Tile proportions (p1, p2, p3) to be allocaated to (training, validation, test) sets.
    :return: Training, validation and test sets of format (x, y, channels, N * px).
    """
    slum_data = tiles.data[:, :, :, slum_tiles]
    slum_mask = tiles.mask[:, :, :, slum_tiles]
    slum = ma.masked_array(slum_data, mask=slum_mask)
    assert tiles.mask.shape == tiles.data.shape
    rest_data = tiles.data[:, :, :, np.invert(slum_tiles)]
    rest_mask = tiles.mask[:, :, :, np.invert(slum_tiles)]
    rest = ma.masked_array(rest_data, mask=rest_mask)
    assert rest.mask.shape == rest.data.shape
    assert slum.mask.shape == slum.data.shape
    n_slum = slum.shape[3]
    n_rest = rest.shape[3]
    slum_train, slum_val, slum_test = split_tiles(n_slum, splits)
    rest_train, rest_val, rest_test = split_tiles(n_rest, splits)
    train_set_data = np.concatenate((slum.data[:, :, :, slum_train], rest.data[:, :, :, rest_train]), axis=3)
    train_set_mask = np.concatenate((slum.mask[:, :, :, slum_train], rest.mask[:, :, :, rest_train]), axis=3)
    train_set = ma.masked_array(train_set_data, mask=train_set_mask)
    val_set_data = np.concatenate((slum.data[:, :, :, slum_val], rest.data[:, :, :, rest_val]), axis=3)
    val_set_mask = np.concatenate((slum.mask[:, :, :, slum_val], rest.mask[:, :, :, rest_val]), axis=3)
    val_set = ma.masked_array(val_set_data, mask=val_set_mask)
    test_set_data = np.concatenate((slum.data[:, :, :, slum_test], rest.data[:, :, :, rest_test]), axis=3)
    test_set_mask = np.concatenate((slum.mask[:, :, :, slum_test], rest.mask[:, :, :, rest_test]), axis=3)
    test_set = ma.masked_array(test_set_data, mask=test_set_mask)
    assert train_set.mask.shape == train_set.data.shape
    assert val_set.mask.shape == val_set.data.shape
    assert test_set.mask.shape == test_set.data.shape
    return train_set, val_set, test_set

tiles = labels_6x12_masked_6x4_tiled_3x3()
# tiles = features_6x12_mask_topleft6x4_tiled_3x3()
splits = (0.34, 0.33, 0.33)
slum_tiles = np.array([True, True, True, False, False, False, True, False])

train, val, test = stratified_split(tiles, slum_tiles, splits)
print("train mask", train.mask.shape)
print("val mask", val.mask.shape)
print("test mask", test.mask.shape)
