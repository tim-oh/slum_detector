"""
Prepare satellite images, possibly including area-of-interest masks and slum maps for ML training and prediction.

-- Scripts

prepare(): Orchestrate data preparation from png to tiles, optionally split into train/val/test sets and saved to disk.

Args -- features: RGB satellite png to be used for training or prediction; tile_size: (x, y) size of image tiles.

Optional args -- mask_png: single channel png with pixels = 0 for mask, 127 for non-masked; label_png: single channel
png with slum = 64-127, non-slum = 0-63; splits: tuple (training, validation, test) proportion, must sum to 1; path_npz:
disk location to save prepared data set(s) in numpy format, path_png: disk location to save prepared image tiles to.

Usage example with all options - only the first two args are required; path_npz and path_png mutually exclusive:
prepare('path/to/features.png',
    (32, 32),
    mask_png='path/to/mask.png',
    label_png='path/to/label.png',
    splits=(0.6, 0.2, 0.2),
    path_npz='desired/path/to/npz/tile/storage' OR path_png='desired/path/to/png/tile/storage')

-- Support functions

png_to_labels(): Turn a png label file into a masked numpy array with converted pixel values.

png_to_features(): Turn a satellite image into a masked numpy array.

convert_labels(): Convert slum_detection_lib greyscale label coding [0:63 slum, 64:127 no slum] to [0 no slum, 1 slum].

convert_features(): Perform numerical checks on pixel-level feature (i.e. satellite image) values.

convert_mask(): Convert slum_detection_lib greyscale pixel coding [127: area of interest, 0: mask] to [0: AOI, 1: mask].

pad(): Add rows of zeros to the bottom or columns to the right of a masked array so it can be tiled without remainder.

tile_coordinates(): Produce a set of top left and bottom right corner coordinates for 2-dimensional image tiles.

stack_tiles(): Turn an image with K channels into tiles specified by coordinates.

clean_stack(): Remove tiles that are completely masked from the stack_array.

mark_slum_tiles: Produce a marker vector of slum tiles for a numpy array of tiled labels.

split_tiles(): Generate randomly sampled indices of train, validation and test sets into specified set proportions.

split_stratified(): Randomly split image tiles into training, validation and test sets.

save_npz(): Save prepared data in numpy savez format, containing different arrays for data and corresponding mask.

save_png(): Save tiles, masks and labels as png images in separate folders.
"""

import numpy as np
import numpy.ma as ma
import imageio
import warnings
import os


def _png_to_labels(png, mask=None):
    """
    Turn a png label file into a masked numpy array with converted pixel values.

    :param png: Label file path.
    :param mask: Optional path to area-of-interest mask corresponding to png; all pixels unmasked if none.
    :return: Masked label array.
    """
    label_array = imageio.imread(png)
    if mask:
        mask_array = imageio.imread(mask)
    else:
        mask_array = np.ones(label_array.shape) * 127
    label_converted = _convert_labels(label_array)
    mask_converted = _convert_mask(mask_array)
    if not label_converted.shape == mask_converted.shape:
        raise ValueError(
            f"Sizing: mask shape{str(mask_converted.shape)!r} doesnt match label shape{str(label_converted.shape)!r}.")
    masked_labels = ma.masked_array(label_converted, mask_converted)
    return masked_labels


def _png_to_features(png, mask=None):
    """
     Turn a satellite image, i.e. a png features file, into a masked numpy array.

    :param png: Feature file path.
    :param mask: Optional path to area-of-interest mask corresponding to png; all pixels unmasked if none.
    :return: Masked label array.
    """
    feature_array = np.array(imageio.imread(png))
    if mask:
        mask_array = imageio.imread(mask)
        mask_array = np.dstack([mask_array] * 3)
    else:
        mask_array = np.ones(feature_array.shape) * 127
    features_converted = _convert_features(feature_array)
    mask_converted = _convert_mask(mask_array)
    masked_features = ma.masked_array(features_converted, mask=mask_converted)
    return masked_features


def _convert_labels(label_array):
    """
    Convert slum_detection_lib greyscale label coding [0:63 no slum, 64:127  slum] to [0 no slum, 1 slum].

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


def _convert_mask(mask_array):
    """
    Convert slum_detection_lib greyscale pixel coding [127: area of interest, 0: mask] to [0: AOI, 1: mask].

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
def _convert_features(feature_array):
    """
    Perform numerical checks on pixel-level feature (i.e. satellite image) values.

    Note that the function only performs value checks as pixel value conversion from png is not required.

    :param feature_array: Numpy array of feature values imported from an RGB png file.
    :return: Input array, unless input was not RGB with values of integers 0-256.
    """
    valid = np.arange(0, 256)
    if not np.isin(feature_array, valid).all():
        raise ValueError(f"Feature values: pixel values not all in {valid!r}.")
    if np.unique(feature_array).shape == (1,):
        warnings.warn(f"Feature values: all elements are set to {np.take(feature_array, 0)!r}.", UserWarning)
    if not feature_array.shape[2] == 3:
        raise ValueError(f"Feature shape: png should have 3 colour channels, but shape is {feature_array.shape!r}.")
    return feature_array


def _pad(input_array, tile_size):
    """
    Add rows of zeros to the bottom or columns to the right of a masked array so that it can be tiled without remainder.

    :param input_array: Feature or label array that may require padding.
    :param tile_size: Desired tuple (x, y) dimension of image tiles.
    :return: Padded array that can be tiled without remainder due to addition of columns/rows of zeros as required.
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
            mask=np.pad(
                input_array.mask, ((0, add_n_below), (0, add_n_right), (0, 0)), 'constant', constant_values=(1,)))
    return padded_array


def _tile_coordinates(image, tile_size):
    """
    Produce a set of top left and bottom right corner coordinates for 2-dimensional image tiles.

    :param image: Image to be split into tiles.
    :param tile_size: Desired tile size.
    :return: Array of dimension 2 by 2 by number of tiles.
    """
    i_upperleft = np.arange(0, image.shape[0], tile_size[0])
    j_upperleft = np.arange(0, image.shape[1], tile_size[1])
    n_coordinates = len(i_upperleft) * len(j_upperleft)
    coordinates = np.zeros((2, 2, n_coordinates))
    counter = 0
    for i in i_upperleft:
        for j in j_upperleft:
            coordinates[0, :, counter] = [i, j]
            coordinates[1, :, counter] = [i + tile_size[0] - 1, j + tile_size[1] - 1]
            counter += 1
    return coordinates.astype('int')


def _stack_tiles(img, coordinates):
    """
    Turn an image with K channels into tiles specified by coordinates.

    :param img: K-channel image to be split into N tiles.
    :param coordinates: Top left and bottom right corners of N tile coordinates, with shape (2, 2, N)
    :return: Array of N image tiles, with shape (tile_height, tile_width, K, N)
    """
    if img.ndim == 2:
        img = ma.masked_array(img.data[..., np.newaxis], mask=img.mask[..., np.newaxis])
    big_k = img.shape[2]
    big_n = coordinates.shape[2]
    tile_img = np.zeros((
        1 + coordinates[1, 0, 0] - coordinates[0, 0, 0], 1 + coordinates[1, 1, 0] - coordinates[0, 1, 0], big_k, big_n))
    tile_mask = np.zeros((
        1 + coordinates[1, 0, 0] - coordinates[0, 0, 0], 1 + coordinates[1, 1, 0] - coordinates[0, 1, 0], big_k, big_n))
    for tile_n in np.arange(0, big_n):
        topleft = coordinates[0, :, tile_n]
        bottomright = coordinates[1, :, tile_n]
        tile_img[:, :, :, tile_n] = img.data[topleft[0]:bottomright[0] + 1, topleft[1]:bottomright[1] + 1, :]
        tile_mask[:, :, :, tile_n] = img.mask[topleft[0]:bottomright[0] + 1, topleft[1]:bottomright[1] + 1, :]
    masked_tiles = ma.masked_array(tile_img, mask=tile_mask)
    return masked_tiles


def _clean_stack(stacked_tiles):
    """
    Remove tiles that are completely masked from the stack_array.

    :param stacked_tiles: Masked array of image times with dimension (x, y, channels, tile)
    :return: Array of same first three dimensions, but possibly with tiles removed (i.e. reduction in final dimension).
    """
    n_tiles = stacked_tiles.shape[3]
    include_tile = np.zeros(n_tiles)
    for i in np.arange(n_tiles):
        include_tile[i] = (stacked_tiles.mask[:, :, :, i].all() != 1)
    cleaned_stack = stacked_tiles[:, :, :, include_tile.astype('bool')]
    return cleaned_stack, tile_register


def _mark_slum_tiles(tiled_labels):
    """
    Produce a marker vector of slum tiles for a numpy array of tiled labels.

    :param tiled_labels: Numpy array of N tiled labels of dimension (x,y,1,N)
    :return: Boolean array of length N with slum = True, non-slum = False.
    """
    n_tiles = tiled_labels.shape[3]
    slum_tiles = np.zeros(n_tiles, dtype='bool')
    for i in np.arange(n_tiles):
        slum_tiles[i] = (tiled_labels.data[:, :, 0, i].any() == True)
    return slum_tiles


def _split_tiles(n_tiles, splits=(0.6, 0.2, 0.2)):
    """
    Generate randomly sampled indices of train, validation and test sets given a number of tiles and set proportions.

    :param n_tiles: Number of tiles to be split into datasets.
    :param splits: Tuple of dataset proportions allocated to (training, validation, test) sets. Must sum to 1.
     The default of (0.6, 0.2, 0.2) is not invoked by prepare(), which only splits if specific proportions are provided.
    :return: Lists of tile indices for each of the split [training], [validation], [test] sets, in that order.
    """
    splits = np.array(splits)
    if splits.any() < 0 or splits.any() > 1:
        raise ValueError(
            f"Split values: dataset proportions must range from 0 to 1 but are {str(splits)!r}.")
    if splits[0] + splits[1] + splits[2] != 1:
        raise ValueError(
            f"Split values: proportions must sum to 1 but are {str(splits)!r}.")
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


# Note: The function above seems to bug in the slicing or concatenation of array masks, hence this clunky version
def _split_stratified(features, labels, slum_tiles, splits):
    """
    Randomly split image tiles into training, validation and test sets, according to 'splits' proportions.

    Stratification according to 'slum_tiles' marker. Output needs to be shuffled prior to training.

    :param features: Array of N image tiles of format (x, y, 3, N), to be split.
    :param labels: Array of N label tiles of format (x, y, 1, N), to be split.
    :param slum_tiles: Boolean array of length N that marks slum tiles along the 4th tile array dimension.
    :param splits: Tile proportions (p1, p2, p3) to be allocated to (training, validation, test) sets.
    :return: Training, validation and test sets of format (x, y, 3, N * px) for features, (x, y, 1, N * px) for labels.
    """
    slum_features = features[:, :, :, slum_tiles]
    rest_features = features[:, :, :, np.invert(slum_tiles)]
    slum_labels = labels[:, :, :, slum_tiles]
    rest_labels = labels[:, :, :, np.invert(slum_tiles)]
    n_slum = np.sum(slum_tiles)
    n_rest = len(slum_tiles) - n_slum
    slum_train, slum_val, slum_test = _split_tiles(n_slum, splits)
    rest_train, rest_val, rest_test = _split_tiles(n_rest, splits)
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



# TODO: Consider how to save for several satellite images so as to avoid conflicts arising from identical png file names
# TODO: Check if the ma.dump() utility which wraps pickle produces files of reasonable size. Replace np.savez() if so.
def prepare(feature_png, tile_size, mask_png=None, label_png=None, splits=None, path_npz=None, path_png=None):
    """
    Orchestrate data preparation functions to produce required outputs for training or prediction.

    Turns a satellite png image into a masked numpy array of desired tile_size and removes fully masked tiles.
    Optionally also turns the associated area-of-interest mask or labels files into corresponding arrays.
    Optionally creates training-test-validation splits to specified proportions.
    Optionally saves prepared data to .npz for whole arrays and to .png tile by tile.

    :param feature_png: Image features in 3-image channel png format to be prepared for training or prediction.
    :param tile_size: Tuple of desired (x, y) tile size, to match neural network architecture.
    :param mask_png: Optional 1-channel png that marks area of interest (AOI), with coding: AOI 127, non-AOI 0.
    :param label_png: Optional 1-channel label value png to match feature_png, with coding: slum 64-127, non-slum 0-63.
    :param splits: Optional tuple of (training, validation, test) set proportions. No splitting unless provided.
    :param path_npz: Optional (absolute) file path for saving function outputs as .npz.
    :param path_png: Optional (absolute) directory (.../) path for saving each tile (and maybe label/mask) as a .png.
    :return: Returns eight objects. Type None except:
    1 -- prepared array of feature tiles if only a feature png is given;
    1,2 -- corresponding label array if a label png is given in addition, but without splits;
    3,4,5,6,7,8 -- training, test and validation sets features & labels if splits in addition to label & feature pngs.
    """
    loaded_features = _png_to_features(feature_png, mask=mask_png)
    padded_features = _pad(loaded_features, tile_size)
    coordinates = _tile_coordinates(padded_features, tile_size)
    tiled_features = _stack_tiles(padded_features, coordinates)
    cleaned_features = _clean_stack(tiled_features)
    if not label_png:
        if path_npz:
            _save_npz(path_npz, features_all=cleaned_features)
        if path_png:
            _save_png(path_png, features_all=cleaned_features)
        return cleaned_features, None, None, None, None, None, None, None
    else:
        loaded_labels = _png_to_labels(label_png, mask=mask_png)
        padded_labels = _pad(loaded_labels, tile_size)
        tiled_labels = _stack_tiles(padded_labels, coordinates)
        cleaned_labels = _clean_stack(tiled_labels)
        if splits:
            slum_marker = _mark_slum_tiles(cleaned_labels)
            features_train, features_val, features_test, labels_train, labels_val, labels_test = \
                _split_stratified(cleaned_features, cleaned_labels, slum_marker, splits)
            if path_npz:
                _save_npz(path_npz,
                         features_train=features_train, features_val=features_val, features_test=features_test,
                         labels_train=labels_train, labels_val=labels_val, labels_test=labels_test)
            if path_png:
                _save_png(path_png,
                         features_train=features_train, features_val=features_val, features_test=features_test,
                         labels_train=labels_train, labels_val=labels_val, labels_test=labels_test)
            return None, None, features_train, features_val, features_test, labels_train, labels_val, labels_test
        else:
            if path_npz:
                _save_npz(path_npz, features_all=cleaned_features, labels_all=cleaned_labels)
            if path_png:
                _save_png(path_png, features_all=cleaned_features, labels_all=cleaned_labels)
            return cleaned_features, cleaned_labels, None, None, None, None, None, None


def _save_png(path_png, features_all=None, labels_all=None, features_train=None, features_val=None, features_test=None,
             labels_train=None, labels_val=None, labels_test=None):
    """
    Save tiles, masks and labels as png images in separate folders.

    For prediction, features only should be provided. For evaluation, features and labels only should be provided.
    For training, features and labels split into (possibly empty) training, validation and test sets should be provided.
    Throw error if a different combination of input data is provided.

    :param path_npz: Absolute path to save data to.
    :param features_all: Tiled, padded, cleaned satellite images.
    :param labels_all: Labels corresponding to the tiled images.
    :param features_train: Training set features.
    :param features_val: Validation set features.
    :param features_test: Test set features.
    :param labels_train: Training set labels.
    :param labels_val: Validation set labels.
    :param labels_test: Test set labels
    :return: Print statement showing location data was saved to.
    """
    if features_all is not None and labels_all is None:
        os.makedirs(path_png + 'images', exist_ok=True)
        os.makedirs(path_png + 'masks', exist_ok=True)
        for i in np.arange(features_all.shape[3]):
            imageio.imwrite(path_png + 'images/image_' + str(i) + '.png',
                            features_all.data[:, :, :, i].astype('uint8'))
            imageio.imwrite(path_png + 'masks/mask_' + str(i) + '.png',
                            features_all.mask[:, :, 0, i].astype('uint8'))
            return print(f'Prediction data saved to {path_png!r} ')
    elif features_all is not None and labels_all is not None:
        os.makedirs(path_png + 'images', exist_ok=True)
        os.makedirs(path_png + 'masks', exist_ok=True)
        os.makedirs(path_png + 'labels', exist_ok=True)
        for i in np.arange(features_all.shape[3]):
            imageio.imwrite(path_png + 'images/image_' + str(i) + '.png',
                            features_all.data[:, :, :, i].astype('uint8'))
            imageio.imwrite(path_png + 'masks/mask_' + str(i) + '.png',
                            features_all.mask[:, :, 0, i].astype('uint8'))
            imageio.imwrite(path_png + 'labels/label_' + str(i) + '.png',
                            labels_all.data[:, :, :, i].astype('uint8'))
        return print(f'Evaluation data saved to {path_png!r} ')
    elif features_train is not None and features_val is not None and features_test is not None and labels_train is not \
            None and labels_val is not None and labels_test is not None:
        os.makedirs(path_png + 'training/images', exist_ok=True)
        os.makedirs(path_png + 'training/masks', exist_ok=True)
        os.makedirs(path_png + 'training/labels', exist_ok=True)
        os.makedirs(path_png + 'validation/images', exist_ok=True)
        os.makedirs(path_png + 'validation/masks', exist_ok=True)
        os.makedirs(path_png + 'validation/labels', exist_ok=True)
        os.makedirs(path_png + 'testing/images', exist_ok=True)
        os.makedirs(path_png + 'testing/masks', exist_ok=True)
        os.makedirs(path_png + 'testing/labels', exist_ok=True)
        for i in np.arange(features_train.shape[3]):
            imageio.imwrite(path_png + 'training/images/image_' + str(i) + '.png',
                            features_train.data[:, :, :, i].astype('uint8'))
            imageio.imwrite(path_png + 'training/masks/mask_' + str(i) + '.png',
                            features_train.mask[:, :, 0, i].astype('uint8'))
            imageio.imwrite(path_png + 'training/labels/label_' + str(i) + '.png',
                            labels_train.data[:, :, :, i].astype('uint8'))
        for i in np.arange(features_val.shape[3]):
            imageio.imwrite(path_png + 'validation/images/image_' + str(i) + '.png',
                            features_val.data[:, :, :, i].astype('uint8'))
            imageio.imwrite(path_png + 'validation/masks/mask_' + str(i) + '.png',
                            features_val.mask[:, :, 0, i].astype('uint8'))
            imageio.imwrite(path_png + 'validation/labels/label_' + str(i) + '.png',
                            labels_val.data[:, :, :, i].astype('uint8'))
        for i in np.arange(features_test.shape[3]):
            imageio.imwrite(path_png + 'testing/images/image_' + str(i) + '.png',
                            features_test.data[:, :, :, i].astype('uint8'))
            imageio.imwrite(path_png + 'testing/masks/mask_' + str(i) + '.png',
                            features_test.mask[:, :, 0, i].astype('uint8'))
            imageio.imwrite(path_png + 'testing/labels/label_' + str(i) + '.png',
                            labels_test.data[:, :, :, i].astype('uint8'))
            return print(f'Training/validation/test data saved to {path_png!r} ')
    else:
        raise ValueError(
            f'Data to be saved: Unexpected argument or combination of arguments provided. Review save_png().')


def _save_npz(path_npz, features_all=None, labels_all=None, features_train=None, features_val=None, features_test=None,
             labels_train=None, labels_val=None, labels_test=None):
    """
    Save prepared data in numpy savez format, containing different arrays for data and corresponding mask.

    For prediction, features only should be provided. For evaluation, features and labels only should be provided.
    For training, features and labels split into (possibly empty) training, validation and test sets should be provided.
    Throw error if a different combination of input data is provided.

    :param path_npz: Absolute path to save data to.
    :param features_all: Tiled, padded, cleaned satellite images.
    :param labels_all: Labels corresponding to the tiled images.
    :param features_train: Training set features.
    :param features_val: Validation set features.
    :param features_test: Test set features.
    :param labels_train: Training set labels.
    :param labels_val: Validation set labels.
    :param labels_test: Test set labels
    :return: Print statement showing location data was saved to.
    """
    if features_all is not None and labels_all is None and features_train is None and features_val is None and \
            features_test is None and labels_train is None and labels_val is None and labels_test is None:
        np.savez(path_npz,
                 cleaned_features_data=features_all.data,
                 cleaned_features_mask=features_all.mask)
        return print(f'Prediction data saved to {path_npz!r} ')
    elif features_all is not None and labels_all is not None and features_train is None and features_val is None and \
            features_test is None and labels_train is None and labels_val is None and labels_test is None:
        np.savez(path_npz,
                 cleaned_features_data=features_all.data,
                 cleaned_features_mask=features_all.mask,
                 cleaned_labels_data=labels_all.data,
                 cleaned_labels_mask=labels_all.mask)
        return print(f'Evaluation data saved to {path_npz!r} ')
    elif features_all is None and features_train is not None and features_val is not None and features_test is not \
            None and labels_train is not None and labels_val is not None and labels_test is not None:
        np.savez(path_npz,
                 features_train_data=features_train.data, features_train_mask=features_train.mask,
                 features_val_data=features_val.data, features_val_mask=features_val.mask,
                 features_test_data=features_test.data, features_test_mask=features_test.mask,
                 labels_train_data=labels_train.data, labels_train_mask=labels_train.mask,
                 labels_val_data=labels_val.data, labels_val_mask=labels_val.mask,
                 labels_test_data=labels_test.data, labels_test_mask=labels_test.mask)
        return print(f'Training/validation/test data saved to {path_npz!r} ')
    else:
        raise ValueError(
            f'Data to be saved: Unexpected argument or combination of arguments provided. Review save_npz().')
