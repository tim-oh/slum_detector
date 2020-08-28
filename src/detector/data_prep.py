import numpy as np
import numpy.ma as ma
import imageio
import warnings


def png_to_labels(png, mask=[]):
    """
    Turns a png label file into a masked numpy array with converted coding.

    :param png: Label file path relative to working directory.
    :param mask: Optional path to area-of-interest mask corresponding to png; all pixels unmasked if none.
    :return: Masked label array.
    """
    label_array = imageio.imread("./" + png)
    if mask == []:
        mask_array = np.ones(label_array.shape) * 127
    else:
        mask_array = imageio.imread("./" + mask)
    label_converted = convert_labels(label_array)
    mask_converted = convert_mask(mask_array)
    if not label_converted.shape == mask_converted.shape:
        raise ValueError(
            f"Mask size: mask array size does not match label array size {str(label_converted.shape)!r}.")
    masked_labels = ma.masked_array(label_converted, mask_converted)
    return masked_labels

def png_to_features(png, mask=[]):
    """

    :param png:
    :param mask:
    :return:
    """
    feature_array = np.array(imageio.imread(png))
    if mask == []:
        mask_array = np.ones(feature_array.shape) * 127
    else:
        mask_array = imageio.imread("./" + mask)
        mask_array = np.dstack([mask_array] * 3)
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
            mask=np.pad(input_array.mask, ((0, add_n_below), (0, add_n_right), (0, 0)),'constant',constant_values=(1,)))
    return padded_array


# TODO: Refactor out the loop, perhaps with a meshgrid() style approach
def tile_coordinates(features, tile_size):
    """

    :param features:
    :param tile_size:
    :return:
    """
    i_upperleft = np.arange(0, features.shape[0], tile_size[0])
    j_upperleft =  np.arange(0, features.shape[1], tile_size[1])
    n_coordinates = len(i_upperleft) * len(j_upperleft)
    coordinates = np.zeros((2, 2, n_coordinates))
    counter = 0
    for i in i_upperleft:
        for j in j_upperleft:
            coordinates[0, :, counter] = [i, j]
            coordinates[1, :, counter] = [i + tile_size[0] - 1, j + tile_size[1] - 1]
            counter += 1
    return coordinates


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
        tile_img[:, :, :, tile_n] = img.data[topleft[0]:bottomright[0]+1, topleft[1]:bottomright[1]+1, :]
        tile_mask[:, :, :, tile_n] = img.mask[topleft[0]:bottomright[0] + 1, topleft[1]:bottomright[1] + 1, :]
    masked_tiles = ma.masked_array(tile_img, mask=tile_mask)
    return masked_tiles



