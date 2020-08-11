import pytest
import numpy as np
import numpy.ma as ma
import imageio
import os
import src.detector.evaluation as ev

tst_dim = (2, 4)

# TODO: Refactor boilerplate fixtures, perhaps with a fixture generator
@pytest.fixture
def all_white():
    all_white = np.ones(tst_dim) * 255
    if not os.path.exists("./tests/all_white.png"):
        imageio.imwrite("./tests/all_white.png", all_white)
    all_white = ma.masked_array(all_white, mask=np.zeros(tst_dim), dtype=int)
    return all_white

@pytest.fixture
def all_white_masked():
    all_white = np.ones(tst_dim) * 255
    if not os.path.exists("./tests/all_white.png"):
        imageio.imwrite("./tests/all_white.png", all_white)
    all_white = ma.masked_array(all_white, mask=np.ones(tst_dim), dtype=int)
    return all_white

@pytest.fixture
def all_black():
    all_black = np.zeros(tst_dim)
    if not os.path.exists("./tests/all_black.png"):
        imageio.imwrite("./tests/all_black.png", all_black)
    all_black = ma.masked_array(all_black, mask=np.zeros(tst_dim), dtype=int)
    return all_black

@pytest.fixture
def mask_all():
    mask = np.zeros(tst_dim)
    mask_path = "./tests/mask_all.png"
    if not os.path.exists(mask_path):
        imageio.imwrite(mask_path, mask)
    mask = mask + 1 # Convert to masked_array convention of masked == 1
    return mask

@pytest.fixture
def mask_none():
    mask = np.ones(tst_dim) * 127
    mask_path = "./tests/mask_none.png"
    if not os.path.exists(mask_path):
        imageio.imwrite(mask_path, mask)
    mask = mask - 127 # Convert to masked_array convention of unmasked == 0
    return mask

# Note imageio.imread and imwrite are not symmetrical; imwrite maps [0, 1] to [0, 255] while imread converts 1:1
def test_load_white_png(all_white):
    all_white_png = ev.png_to_labels("tests/all_white.png")
    assert np.array_equal(all_white_png, all_white)
    assert type(all_white_png) == ma.masked_array

def test_load_black_png(all_black):
    all_black_png = ev.png_to_labels("tests/all_black.png")
    assert np.array_equal(all_black_png, all_black)
    assert type(all_black_png) == ma.masked_array

def test_load_fail(all_black, all_white):
    with pytest.raises(AssertionError):
        all_black_png = ev.png_to_labels("tests/all_black.png")
        assert np.array_equal(all_black_png, all_white)

# Todo: Array of masked black pixels evaluates same as unmasked one
@pytest.mark.parametrize("png,mask,expected", [
    ("tests/all_black.png", "tests/mask_none.png", pytest.lazy_fixture("all_black")),
    ("tests/all_black.png", "tests/mask_all.png", pytest.lazy_fixture("all_black")),
    ("tests/all_white.png", "tests/mask_none.png", pytest.lazy_fixture("all_white")),
    ("tests/all_white.png", "tests/mask_all.png", pytest.lazy_fixture("all_white_masked"))
])
def test_load_masked_png(png, mask, expected):
    assert np.array_equal(ev.png_to_labels(png, mask), expected)

## Behaviour of the loader
# Case 1: give it a png without mask and it creates a masked array of that data
# Case 2: give it mask as well and it creates the according masked array