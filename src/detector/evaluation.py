import numpy as np
import numpy.ma as ma
import imageio

def png_to_labels(png, mask=[]):
    """
    Turns a png label file into a masked numpy array.
    :param png: Label file path relative to working directory.
    :param mask: Optional path to corresponding mask; all pixels unmasked if none.
    Converts slum_detection_lib greyscale pixel coding [127: area of interest, 0: mask] to [0: AOI, 1: mask].
    :return: Masked label array.
    """
    img_array = imageio.imread("./" + png)
    if mask == []:
        mask = np.zeros(img_array.shape)
    else:
        mask = imageio.imread("./" + mask)
    mask[mask == 0] = 1
    mask[mask == 127] = 0
    masked_array = ma.masked_array(img_array, mask)
    return masked_array
