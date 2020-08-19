import pytest
import src.detector.data_prep as prep
import numpy as np

tile_size = (3, 3)

# TODO:
@pytest.mark.xfail
def test_load_image_no_padding_no_mask(blocks_no_pad):
    img_loaded = prep.load_image("./blocks_no_pad.png")
    img_padded = prep.pad_image(img_loaded, tile_size)
    assert np.array_equal(img_padded, blocks_no_pad)

@pytest.mark.xfail
def test_load_image_bottom_padding_no_mask(blocks_bottom_pad):
    img_loaded = prep.load_image("./all_127_bottom_pad.png")
    img_padded = prep.pad_image(img_loaded, tile_size)
    assert np.array_equal(img_padded, blocks_bottom_pad)

@pytest.mark.xfail
def test_load_image_right_padding_no_mask(blocks_right_pad):
    img_loaded = prep.load_image("./blocks_right_pad.png")
    img_padded = prep.pad_image(img_loaded, tile_size)
    assert np.array_equal(img_padded, blocks_no_pad)

@pytest.mark.xfail
def test_load_image_both_padding_no_mask(blocks_both_pad):
    img_loaded = prep.load_image("./blocks_right_pad.png")
    img_padded = prep.pad_image(img_loaded, tile_size)
    assert np.array_equal(img_padded, all_127_no_pad)

def test_