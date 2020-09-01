import numpy as np
import numpy.ma as ma
import imageio
import warnings


def png_to_labels(png, mask=None):
    """
    Turns a png label file into a masked numpy array with converted coding.

    :param png: Label file path relative to working directory.
    :param mask: Optional path to area-of-interest mask corresponding to png; all pixels unmasked if none.
    :return: Masked label array.
    """
    label_array = imageio.imread("./" + png)
    if mask:
        mask_array = imageio.imread("./" + mask)
    else:
        mask_array = np.ones(label_array.shape) * 127
    label_converted = convert_labels(label_array)
    mask_converted = convert_mask(mask_array)
    if not label_converted.shape == mask_converted.shape:
        raise ValueError(
            f"Mask size: mask array size does not match label array size {str(label_converted.shape)!r}.")
    masked_labels = ma.masked_array(label_converted, mask_converted)
    return masked_labels


def png_to_features(png, mask=None):
    """

    :param png:
    :param mask:
    :return:
    """
    feature_array = np.array(imageio.imread(png))
    if mask:
        mask_array = imageio.imread("./" + mask)
        mask_array = np.dstack([mask_array] * 3)
    else:
        mask_array = np.ones(feature_array.shape) * 127
    features_converted = convert_features(feature_array)
    mask_converted = convert_mask(mask_array)
    masked_features = ma.masked_array(features_converted, mask=mask_converted)
    return masked_features


# TEMPORARY function adjusted for missing (first?) column in label array
def png_to_labels2(png, mask):
    """
    Turns a png label file into a masked numpy array with converted coding.

    :param png: Label file path relative to working directory.
    :param mask: Optional path to area-of-interest mask corresponding to png; all pixels unmasked if none.
    :return: Masked label array.
    """
    label_array = imageio.imread("./" + png)
    mask_array = imageio.imread("./" + mask)
    mask_array = mask_array[:, 1:]
    labels_converted = convert_labels(label_array)
    mask_converted = convert_mask(mask_array)
    if not labels_converted.shape == mask_converted.shape:
        raise ValueError(
            f'Mask size: mask shape {mask_array.shape} does not match label shape {str(labels_converted.shape)!r}.')
    masked_labels = ma.masked_array(labels_converted, mask_converted)
    return masked_labels


# TODO: Use dictionary to look up conversion scheme
def convert_labels(label_array):
    """
    Converts slum_detection_lib greyscale label coding [0:63 slum, 64:127 no slum] to [0 no slum, 1 slum].

    :param label_array: Numpy array of imported pixel labels.
    :return: Numpy array of converted pixel labels.
    """
    valid = np.arange(0, 128)
    if not np.isin(label_array, valid).all():
        raise ValueError(f"Label values: all elements must be one of {valid!r}.")
    if np.unique(label_array).shape == (1,):
        warnings.warn(f"Label values: all elements are set to {np.take(label_array, 0)!r}.", UserWarning)
    label_array[label_array <= 63] = 0
    label_array[label_array > 63] = 1
    return label_array


# TODO: Use dictionary to look up conversion scheme
def convert_mask(mask_array):
    """
    Converts slum_detection_lib greyscale pixel coding [127: area of interest, 0: mask] to [0: AOI, 1: mask].

    :param mask_array: Numpy array of imported mask values.
    :return: Numpy array of converted mask values.
    """
    valid = [127, 0]
    if not np.isin(mask_array, valid).all():
        raise ValueError(f'Mask values: must all be one of{valid!r} ')
    if np.unique(mask_array).shape == (1,):
        warnings.warn(f'Mask values: all elements are set to {np.take(mask_array, 0)!r}.', UserWarning)
    mask_array[mask_array == 0] = 1
    mask_array[mask_array == 127] = 0
    return mask_array


# Note: only numerical checks, no pixel value conversion
def convert_features(feature_array):
    """

    :param feature_array:
    :return:
    """
    valid = np.arange(0, 256)
    if not np.isin(feature_array, valid).all():
        raise ValueError(f"Feature values: pixel values not all in {valid!r}.")
    if np.unique(feature_array).shape == (1,):
        warnings.warn(f"Feature values: all elements are set to {np.take(feature_array, 0)!r}.", UserWarning)
    if not feature_array.shape[2] == 3:
        raise ValueError(f"Feature shape: png should have 3 colour channels, but shape is {feature_array.shape!r}.")
    return feature_array


def pad(input_array, tile_size):
    """

    :param input_array:
    :param tile_size:
    :return:
    """
    array_height = input_array.shape[0]
    array_width = input_array.shape[1]
    tile_height = tile_size[0]
    tile_width = tile_size[1]
    target_height = (array_height // tile_height) * tile_height + (not (array_height % tile_height) == 0) * tile_height
    target_width = (array_width // tile_width) * tile_width + (not (array_width % tile_width) == 0) * tile_width
    add_n_below = target_height - array_height
    add_n_right = target_width - array_width
    if input_array.ndim == 2:
        padded_array = ma.masked_array(
            np.pad(input_array.data, ((0, add_n_below), (0, add_n_right))),
            mask=np.pad(input_array.mask, ((0, add_n_below), (0, add_n_right)), 'constant', constant_values=(1,)))
    elif input_array.ndim == 3:
        padded_array = ma.masked_array(
            np.pad(input_array.data, ((0, add_n_below), (0, add_n_right), (0, 0))),
            mask=np.pad(input_array.mask, ((0, add_n_below), (0, add_n_right), (0, 0)), 'constant',
                        constant_values=(1,)))
    return padded_array


# TODO: Refactor out the loop, perhaps with a meshgrid() style approach
def tile_coordinates(features, tile_size):
    """
    Labels alone don't require tiling, they are associated with features that have the same 2D extent/coordinates.

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
    return coordinates.astype('int')


def stack_tiles(img, coordinates):
    """
    Turns an image with K channels into tiles specified by coordinates.

    :param img: K-channel image to be split into N tiles.
    :param coordinates: Top left and bottom right corners of N tile coordinates, with shape (2, 2, N)
    :return: Array of N image tiles, with shape (tile_height, tile_width, K, N)
    """
    if img.ndim == 2:
        img = ma.masked_array(img.data[..., np.newaxis], mask=img.mask[..., np.newaxis])
    big_k = img.shape[2]
    big_n = coordinates.shape[2]
    tile_img = np.zeros((
        1 + coordinates[1, 0, 0] - coordinates[0, 0, 0], 1 + coordinates[1, 1, 0] - coordinates[0, 1, 0], big_k, big_n
    ))
    tile_mask = np.zeros((
        1 + coordinates[1, 0, 0] - coordinates[0, 0, 0], 1 + coordinates[1, 1, 0] - coordinates[0, 1, 0], big_k, big_n
    ))
    for tile_n in np.arange(0, big_n):
        topleft = coordinates[0, :, tile_n]
        bottomright = coordinates[1, :, tile_n]
        tile_img[:, :, :, tile_n] = img.data[topleft[0]:bottomright[0] + 1, topleft[1]:bottomright[1] + 1, :]
        tile_mask[:, :, :, tile_n] = img.mask[topleft[0]:bottomright[0] + 1, topleft[1]:bottomright[1] + 1, :]
    masked_tiles = ma.masked_array(tile_img, mask=tile_mask)
    return masked_tiles


def clean_stack(stack_array):
    """
    Removes tiles that are completely masked from the stack_array.

    :param stack_array: Masked array of image times with dimension (x_value, y_value, channel, tile)
    :return: Array of same first three dimensions, but possibly with tiles removed (i.e. reduction in dimension 4).
    """
    n_tiles = stack_array.shape[3]
    include_tile = np.zeros(n_tiles)
    for i in np.arange(n_tiles):
        include_tile[i] = (stack_array.mask[:, :, :, i].all() != 1)
    cleaned_stack = stack_array[:, :, :, include_tile.astype('bool')]
    return cleaned_stack


def mark_slum_tiles(tiled_labels):
    """
    Take in N tiled labels of dimension (x,y,1,N) and return array of length N that marks slum-containing tiles as True.

    :param tiled_labels:
    :return:
    """
    n_tiles = tiled_labels.shape[3]
    slum_tiles = np.zeros(n_tiles, dtype='bool')
    for i in np.arange(n_tiles):
        slum_tiles[i] = (tiled_labels.data[:, :, 0, i].any() == True)
    return slum_tiles


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
    n_train = np.round(splits[0] * n_tiles).astype('int')
    train_indices = np.random.choice(n_tiles, n_train, replace=False)
    indices_remain = np.delete(np.arange(n_tiles), train_indices)
    if splits[2] > 0:
        n_test = round(splits[2] / (splits[1] + splits[2]) * len(indices_remain))
        test_indices = np.random.choice(indices_remain, n_test, replace=False)
    else:
        test_indices = np.array([], dtype=int)
    val_indices = np.delete(np.arange(n_tiles), np.concatenate((train_indices, test_indices)))
    return train_indices, val_indices, test_indices


# def stratified_split(tiles, slum_tile_marker, splits):
#     """
#     Random split of image tiles into training, validation and test sets, according to 'splits' proportions.
#     Stratification according to 'slum_tiles' marker. Output needs to be shuffled prior to training.
#
#     :param tiles: Array of N image tiles of format (x, y, channels, N), to be split.
#     :param slum_tiles: Boolean array of length N that marks slum tiles along the 4th tile array dimension.
#     :param splits: Tile proportions (p1, p2, p3) to be allocaated to (training, validation, test) sets.
#     :return: Training, validation and test sets of format (x, y, channels, N * px).
#     """
#     slum = tiles[:, :, :, slum_tile_marker]
#     rest = tiles[:, :, :, np.invert(slum_tile_marker)]
#     n_slum = slum.shape[3]
#     n_rest = rest.shape[3]
#     slum_train, slum_val, slum_test = split_tiles(n_slum, splits)
#     rest_train, rest_val, rest_test = split_tiles(n_rest, splits)
#     train_set = ma.concatenate((slum[:, :, :, slum_train], rest[:, :, :, rest_train]), axis=3)
#     val_set = ma.concatenate((slum[:, :, :, slum_val], rest[:, :, :, rest_val]), axis=3)
#     test_set = ma.concatenate((slum[:, :, :, slum_test], rest[:, :, :, rest_test]), axis=3)
#     return train_set, val_set, test_set


# TODO: Update docstrings
# Note: The function above seems to bug in the slicing or concatenation of array masks, hence this clunky version
def stratified_split(features, labels, slum_tiles, splits):
    """
    Random split of image tiles into training, validation and test sets, according to 'splits' proportions.
    Stratification according to 'slum_tiles' marker. Output needs to be shuffled prior to training.

    :param tiles: Array of N image tiles of format (x, y, channels, N), to be split.
    :param slum_tiles: Boolean array of length N that marks slum tiles along the 4th tile array dimension.
    :param splits: Tile proportions (p1, p2, p3) to be allocaated to (training, validation, test) sets.
    :return: Training, validation and test sets of format (x, y, channels, N * px).
    """
    slum_features = features[:, :, :, slum_tiles]
    rest_features = features[:, :, :, np.invert(slum_tiles)]
    slum_labels = labels[:, :, :, slum_tiles]
    rest_labels = labels[:, :, :, np.invert(slum_tiles)]
    n_slum = np.sum(slum_tiles)
    n_rest = len(slum_tiles) - n_slum
    slum_train, slum_val, slum_test = split_tiles(n_slum, splits)
    rest_train, rest_val, rest_test = split_tiles(n_rest, splits)
    features_train_data = \
        np.concatenate((slum_features.data[:, :, :, slum_train], rest_features.data[:, :, :, rest_train]), axis=3)
    features_train_mask = \
        np.concatenate((slum_features.mask[:, :, :, slum_train], rest_features.mask[:, :, :, rest_train]), axis=3)
    features_train = ma.masked_array(features_train_data, mask=features_train_mask)
    features_val_data = \
        np.concatenate((slum_features.data[:, :, :, slum_val], rest_features.data[:, :, :, rest_val]), axis=3)
    features_val_mask = \
        np.concatenate((slum_features.mask[:, :, :, slum_val], rest_features.mask[:, :, :, rest_val]), axis=3)
    features_val = ma.masked_array(features_val_data, mask=features_val_mask)
    features_test_data = \
        np.concatenate((slum_features.data[:, :, :, slum_test], rest_features.data[:, :, :, rest_test]), axis=3)
    features_test_mask = \
        np.concatenate((slum_features.mask[:, :, :, slum_test], rest_features.mask[:, :, :, rest_test]), axis=3)
    features_test = ma.masked_array(features_test_data, mask=features_test_mask)
    labels_train_data = \
        np.concatenate((slum_labels.data[:, :, :, slum_train], rest_labels.data[:, :, :, rest_train]), axis=3)
    labels_train_mask = \
        np.concatenate((slum_labels.mask[:, :, :, slum_train], rest_labels.mask[:, :, :, rest_train]), axis=3)
    labels_train = ma.masked_array(labels_train_data, mask=labels_train_mask)
    labels_val_data = \
        np.concatenate((slum_labels.data[:, :, :, slum_val], rest_labels.data[:, :, :, rest_val]), axis=3)
    labels_val_mask = \
        np.concatenate((slum_labels.mask[:, :, :, slum_val], rest_labels.mask[:, :, :, rest_val]), axis=3)
    labels_val = ma.masked_array(labels_val_data, mask=labels_val_mask)
    labels_test_data = \
        np.concatenate((slum_labels.data[:, :, :, slum_test], rest_labels.data[:, :, :, rest_test]), axis=3)
    labels_test_mask = \
        np.concatenate((slum_labels.mask[:, :, :, slum_test], rest_labels.mask[:, :, :, rest_test]), axis=3)
    labels_test = ma.masked_array(labels_test_data, mask=labels_test_mask)
    return features_train, features_val, features_test, labels_train, labels_val, labels_test


def prepare(feature_png, tile_size, mask_png=None, label_png=None, splits=None, path=None):
    if not mask_png:
        loaded_features = png_to_features(feature_png)
    else:
        loaded_features = png_to_features(feature_png, mask=mask_png)
    padded_features = pad(loaded_features, tile_size)
    coordinates = tile_coordinates(padded_features, tile_size)
    tiled_features = stack_tiles(padded_features, coordinates)
    cleaned_features = clean_stack(tiled_features)
    if not label_png:
        if path:
            np.savez(
                path,
                cleaned_features_data=cleaned_features.data,
                cleaned_features_mask=cleaned_features.mask
            )
        return cleaned_features
    else:
        if not mask_png:
            loaded_labels = png_to_labels(label_png)
        else:
            loaded_labels = png_to_labels(label_png, mask=mask_png)
        padded_labels = pad(loaded_labels, tile_size)
        tiled_labels = stack_tiles(padded_labels, coordinates)
        cleaned_labels = clean_stack(tiled_labels)
        if splits:
            slum_marker = mark_slum_tiles(cleaned_labels)
            features_train, features_val, features_test, labels_train, labels_val, labels_test = \
                stratified_split(cleaned_features, cleaned_labels, slum_marker, splits)
            if path:
                np.savez(
                    path,
                    features_train_data=features_train.data,  features_train_mask=features_train.mask,
                    features_val_data=features_val.data, features_val_mask=features_val.mask,
                    features_test_data=features_test.data, features_test_mask=features_test.mask,
                    labels_train_data=labels_train.data, labels_train_mask=labels_train.mask,
                    labels_val_data=labels_val.data, labels_val_mask=labels_val.mask,
                    labels_test_data=labels_test.data, labels_test_mask=labels_test.mask
                )
            return features_train, features_val, features_test, labels_train, labels_val, labels_test
        else:
            if path:
                np.savez(
                    path,
                    cleaned_features_data=cleaned_features.data,
                    cleaned_features_mask=cleaned_features.mask,
                    cleaned_labels_data=cleaned_labels.data,
                    cleaned_labels_mask=cleaned_labels.mask
                )
            return cleaned_features, cleaned_labels