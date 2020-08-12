import numpy as np
import numpy.ma as ma
import imageio
import warnings

# Note: consider evaluation for multiple files. How to aggregate predictions from different cities?
# A utility to stitch predictions for neighbouring areas together could be useful, depending on satellite imagery.

def png_to_labels(png, mask=[]):
    """
    Turns a png label file into a masked numpy array with converted coding.

    :param png: Label file path relative to working directory.
    :param mask: Optional path to area-of-interest mask corresponding to png; all pixels unmasked if none.
    :return: Masked label array.
    """
    pred_array = imageio.imread("./" + png)
    if mask == []:
        mask_array = np.ones(pred_array.shape) * 127
    else:
        mask_array = imageio.imread("./" + mask)
    pred_converted = convert_pred(pred_array)
    mask_converted = convert_mask(mask_array)
    if not pred_converted.shape == mask_converted.shape:
        raise ValueError("Mask size: prediction array size does not match mask array size.")
    masked_labels = ma.masked_array(pred_converted, mask_converted)
    return masked_labels

# TODO: Refactor ambiguous use of valid: notation might suggest a range, but it's a set unless notation is a:b
def convert_mask(mask_array):
    """
    Converts slum_detection_lib greyscale pixel coding [127: area of interest, 0: mask] to [0: AOI, 1: mask].

    :param mask_array: Numpy array of imported mask values.
    :return: Numpy array of converted mask values.
    """
    valid = [0, 127]
    if not np.isin(mask_array, valid).all():
        raise ValueError("Mask values: all elements must be one of %r." % valid)
    if np.unique(mask_array).ndim == 1:
        warnings.warn("Mask values: all elements are set to %r." % mask_array[0, 0], UserWarning)
    mask_array[mask_array == 0] = 1
    mask_array[mask_array == 127] = 0
    return mask_array

def convert_pred(pred_array):
    """
    Converts slum_detection_lib greyscale label coding [0:63 slum, 64:127 no slum] to [0 no slum, 1 slum].

    :param mask: Numpy array of imported pixel labels.
    :return: Numpy array of converted pixel labels.
    """
    valid = np.arange(0, 128)
    if not np.isin(pred_array, valid).all():
        raise ValueError("Label values: all elements must be one of %r." % valid)
    if np.unique(pred_array).ndim == 1:
        warnings.warn("Label values: all elements are set to %r." % pred_array[0, 0], UserWarning)
    pred_array[pred_array <= 63] = 0
    pred_array[pred_array > 63] = 1
    return pred_array