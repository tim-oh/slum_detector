import numpy as np
import numpy.ma as ma
import imageio
import os
import src.detector.data_prep
from pytest_lazyfixture import lazy_fixture
import pytest


@pytest.fixture(scope="session")
def all_127_bottom_mask_png(tmpdir_factory):
    tst_dim = (4, 8)
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("all_grey.png"))
    imageio.imwrite(path, all_grey)
    return path

@pytest.fixture(scope="session")
def all_127_bottom_mask_array():
    tst_dim = (4, 8)
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask = np.concatenate((np.zeros((dim_one, dim_two)), np.ones((dim_one, dim_two))), axis=0)
    all_grey = ma.masked_array(np.ones(tst_dim), mask=mask, dtype=int)
    return all_grey

@pytest.mark.parametrize("labels,mask,expected", [
    (lazy_fixture("all_0_no_mask_png"), lazy_fixture("mask_none_png"), lazy_fixture("all_0_no_mask_array")),
    (lazy_fixture("all_0_no_mask_png"), lazy_fixture("mask_all_png"), lazy_fixture("all_0_mask_all_array")),
    (lazy_fixture("all_127_no_mask_png"), lazy_fixture("mask_none_png"), lazy_fixture("all_127_no_mask_array")),
    (lazy_fixture("all_127_no_mask_png"), lazy_fixture("mask_all_png"), lazy_fixture("all_127_full_mask_array")),
    (lazy_fixture("all_127_no_mask_png"), lazy_fixture("mask_none_png"), lazy_fixture("all_127_bottom_mask_array"))
])
def test_load_labels_masked(labels, mask, expected):
    loaded_mask = src.detector.data_prep.png_to_labels(labels, mask)
    assert np.array_equal(loaded_mask.data, expected.data)
    assert np.array_equal(loaded_mask.mask, expected.mask)
    assert type(loaded_mask) == type(expected) == ma.masked_array

# contents of test_image.py
def test_converter(all_127_bottom_mask_array, all_127_bottom_mask_png):
    img_loaded = imageio.imread(str(all_127_bottom_mask_png))
    img_converted = src.detector.data_prep.convert_labels(img_loaded)
    assert np.array_equal(all_127_bottom_mask_array, img_converted)

def blocks_no_pad_mask_leftmost_array():
    blocks_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]])
    blocks_rgb = np.dstack([blocks_digits] * 3)
    unconverted_mask = np.concatenate((np.ones((6, 1)), np.zeros((6, 8))), axis=1)
    mask_rgb = np.dstack([unconverted_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks



a = blocks_no_pad_mask_leftmost_array()
print(a)