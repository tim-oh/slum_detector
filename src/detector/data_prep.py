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
    valid = np.arange(0, 256)
    if not np.isin(feature_array, valid).all():
        raise ValueError(f"Feature values: pixel values not all in {valid!r}.")
    if np.unique(feature_array).shape == (1,):
        warnings.warn(f"Feature values: all elements are set to {np.take(feature_array, 0)!r}.", UserWarning)
    if not feature_array.shape[2] == 3:
        raise ValueError(f"Feature shape: png should have 3 channels, but shape is {feature_array.shape!r}.")
    return feature_array


def pad(input_array, tile_size):
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