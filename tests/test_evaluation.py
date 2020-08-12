import pytest
import numpy as np
import numpy.ma as ma
import imageio
import os
import src.detector.evaluation as ev

tst_dim = (140, 204) # Pick two even number to avoid breakage

# TODO: Refactor boilerplate fixtures, perhaps with a fixture generator
@pytest.fixture
def all_grey():
    all_grey = np.ones(tst_dim) * 127
    if not os.path.exists("./tests/all_grey.png"):
        imageio.imwrite("./tests/all_grey.png", all_grey)
    all_grey = ma.masked_array(np.ones(tst_dim), mask=np.zeros(tst_dim), dtype=int)
    return all_grey

@pytest.fixture
def all_grey_masked():
    all_grey = np.ones(tst_dim) * 127
    if not os.path.exists("./tests/all_grey.png"):
        imageio.imwrite("./tests/all_grey.png", all_grey)
    all_grey = ma.masked_array(np.ones(tst_dim), mask=np.ones(tst_dim), dtype=int)
    return all_grey

@pytest.fixture
def grey_masked_bottom():
    all_grey = np.ones(tst_dim) * 127
    if not os.path.exists("./tests/all_grey.png"):
        imageio.imwrite("./tests/all_grey.png", all_grey)
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask = np.concatenate((np.zeros((dim_one, dim_two)), np.ones((dim_one, dim_two))), axis=0)
    all_grey = ma.masked_array(np.ones(tst_dim), mask=mask, dtype=int)
    return all_grey

@pytest.fixture
def all_black():
    all_black = np.zeros(tst_dim)
    if not os.path.exists("./tests/all_black.png"):
        imageio.imwrite("./tests/all_black.png", all_black)
    all_black = ma.masked_array(all_black, mask=np.zeros(tst_dim), dtype=int)
    return all_black

@pytest.fixture
def mixed_values():
    mixed_values = np.arange(0, 128).reshape(64, 2).astype("uint8")
    if not os.path.exists("./tests/mixed_values.png"):
        imageio.imwrite("./tests/mixed_values.png", mixed_values)
    mixed_values = ma.masked_array(mixed_values, mask=np.zeros(mixed_values.shape), dtype=int)
    return mixed_values

@pytest.fixture
def mask_all():
    mask = np.zeros(tst_dim)
    mask = mask.astype("uint8") # imageio.imwrite scales max value to 255 when in another number format
    mask_path = "./tests/mask_all.png"
    if not os.path.exists(mask_path):
        imageio.imwrite(mask_path, mask) # Write mask to disk that follows slums-world conventions
    mask = mask + 1 # Convert to masked_array convention of masked == 1
    return mask

@pytest.fixture
def mask_bottom():
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask = np.concatenate((np.ones((dim_one, dim_two))*127, np.zeros((dim_one, dim_two))), axis=0).astype("uint8")
    mask_path = "./tests/mask_bottom.png"
    if not os.path.exists(mask_path):
        imageio.imwrite(mask_path, mask) # Write mask to disk that follows slums-world conventions
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask = np.concatenate((np.zeros((dim_one, dim_two)), np.ones((dim_one, dim_two))), axis=0)
    return mask

@pytest.fixture
def mask_small():
    mask = np.array([[127, 127, 127], [0, 0, 0]])
    mask = mask.astype("uint8")
    mask_path = "./tests/mask_small.png"
    if not os.path.exists(mask_path):
        imageio.imwrite(mask_path, mask) # Write mask to disk that follows slums-world conventions
    mask = np.array([[0, 0, 0], [1, 1, 1]]) # Convert to masked_array convention of masked == 1, unmasked == 0
    return mask

@pytest.fixture
def mask_none():
    mask = np.ones(tst_dim) * 127
    mask = mask.astype("uint8")
    mask_path = "./tests/mask_none.png"
    if not os.path.exists(mask_path):
        imageio.imwrite(mask_path, mask) # Write mask to disk that follows slums-world conventions
    mask = mask - 127 # Convert to masked_array convention of unmasked == 0
    return mask

# Note: imwrite scales img range (e.g [0, 1]) to [0, 255] unless dtype=uint8, whereas imread always converts 1:1
def test_load_grey_png(all_grey):
    all_grey_png = ev.png_to_labels("tests/all_grey.png")
    assert np.array_equal(all_grey_png, all_grey)
    assert type(all_grey_png) == ma.masked_array

def test_load_mixed_png(mixed_values):
    mixed_png = ev.png_to_labels("tests/mixed_values.png")
    converted = np.concatenate((np.zeros((32, 2)), np.ones((32, 2))), axis=0)
    assert np.array_equal(mixed_png, converted)

def test_load_black_png(all_black):
    all_black_png = ev.png_to_labels("tests/all_black.png")
    assert np.array_equal(all_black_png, all_black)
    assert type(all_black_png) == ma.masked_array

def test_load_fail(all_black, all_grey):
    with pytest.raises(AssertionError):
        all_black_png = ev.png_to_labels("tests/all_black.png")
        assert np.array_equal(all_black_png, all_grey)

# TODO: Array of masked black pixels wrongly evaluates same as unmasked one.
# TODO: Tests fail unless tests/xxx.png exists already, so have to py.test twice. Fix via fixture scopes?
@pytest.mark.parametrize("pred,mask,expected", [
    ("tests/all_black.png", "tests/mask_none.png", pytest.lazy_fixture("all_black")),
    ("tests/all_black.png", "tests/mask_all.png", pytest.lazy_fixture("all_black")),
    ("tests/all_grey.png", "tests/mask_none.png", pytest.lazy_fixture("all_grey")),
    ("tests/all_grey.png", "tests/mask_all.png", pytest.lazy_fixture("all_grey_masked")),
    ("tests/all_grey.png", "tests/mask_bottom.png", pytest.lazy_fixture("grey_masked_bottom"))
])
def test_load_masked_pred(pred, mask, expected):
    expected = expected
    loaded_mask = ev.png_to_labels(pred, mask)
    assert np.array_equal(loaded_mask, expected)

def test_wrong_mask_size(all_grey, mask_small):
    with pytest.raises(ValueError):
        ev.png_to_labels("tests/all_grey.png", "tests/mask_small.png")

def test_wrong_mask_value(mask_all):
    with pytest.raises(ValueError):
        ev.convert_mask(mask_all)

def test_uniform_mask_value_warning(mask_none):
    with pytest.warns(UserWarning):
        ev.convert_mask(mask_none)

@pytest.mark.parametrize("mask,expected", [
    ("tests/mask_all.png", pytest.lazy_fixture("mask_all")),
    ("tests/mask_none.png", pytest.lazy_fixture("mask_none")),
    ("tests/mask_bottom.png", pytest.lazy_fixture("mask_bottom")),
    ])
def test_convert_masks(mask, expected):
    loaded = imageio.imread(mask)
    converted = ev.convert_mask(loaded)
    assert np.array_equal(converted, expected)

# Test_all_slum: Convert a file with all 127 to get mask_none, but throw warning.
# Test_no_slum: Convert a file with all 0 to get mask_all, but throw warning.
# Test_slum_left_no_slum_right: Convert a file that has a mix and see the correct result.

@pytest.fixture()
def pred_wrong_value():
    pred = np.arange(122, 130).reshape(2, 4).astype("uint8")
    if not os.path.exists("./tests/pred_wrong_value.png"):
        imageio.imwrite("./tests/pred_wrong_value.png", pred)
    return pred

def test_wrong_pred_value(pred_wrong_value):
    with pytest.raises(ValueError):
        ev.convert_pred(pred_wrong_value)

def test_uniform_pred_value_warning(all_black):
    with pytest.warns(UserWarning):
        ev.convert_pred(all_black)

@pytest.mark.parametrize("pred,expected", [
    ("tests/all_black.png", pytest.lazy_fixture("mask_none")),
    ("tests/all_grey.png", pytest.lazy_fixture("mask_all"))
    # ,
    # ("tests/mask_bottom.png", pytest.lazy_fixture("mask_bottom")),
    ])
def test_convert_pred(pred, expected):
    loaded = imageio.imread(pred)
    converted = ev.convert_pred(loaded)
    assert np.array_equal(converted, expected)