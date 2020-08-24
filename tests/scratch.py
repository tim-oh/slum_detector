import numpy as np
import numpy.ma as ma
import imageio
import warnings
from tabulate import tabulate
import src.detector.data_prep


def padded_all_127_mask_all():
    tst_dim = (4, 8)
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("tests/tmp/all_grey.png", all_grey)
    padded_array = np.concatenate((np.concatenate((np.ones(tst_dim), np.zeros((2,8))),axis=0), np.zeros((6,1))),axis=1)
    mask_array_nopad= np.ones(tst_dim)
    mask_array = np.pad(mask_array_nopad, ((0, 2), (0, 1)), 'constant', constant_values=(1,))
    masked_array = ma.masked_array(padded_array, mask=mask_array)
    return masked_array


def padded_all_127_no_mask():
    tst_dim = (4, 8)
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("tests/tmp/all_grey.png", all_grey)
    padded_array = np.concatenate((np.concatenate((np.ones(tst_dim), np.zeros((2,8))),axis=0), np.zeros((6,1))),axis=1)
    mask_array_nopad= np.zeros(tst_dim)
    mask_array = np.pad(mask_array_nopad, ((0, 2), (0, 1)), 'constant', constant_values=(1,))
    masked_array = ma.masked_array(padded_array, mask=mask_array)
    return masked_array

a = padded_all_127_mask_all()a
b = padded_all_127_no_mask()
a
b

