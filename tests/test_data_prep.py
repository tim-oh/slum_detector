import pytest
import src.detector.data_prep
import numpy as np
import numpy.ma as ma
import imageio


########################################################################################################################
# Loading & conversion                                                                                                 #
########################################################################################################################
# TODO: These tests fail unless tests/xxx.png exists already, so have to py.test twice.
@pytest.mark.parametrize("labels,mask,expected", [
    ("tests/tmp/all_black.png", "tests/tmp/mask_none.png", pytest.lazy_fixture("all_0_no_mask")),
    ("tests/tmp/all_black.png", "tests/tmp/mask_all.png", pytest.lazy_fixture("all_0_mask_all")),
    ("tests/tmp/all_grey.png", "tests/tmp/mask_none.png", pytest.lazy_fixture("all_127_no_mask")),
    ("tests/tmp/all_grey.png", "tests/tmp/mask_all.png", pytest.lazy_fixture("all_127_full_mask")),
    ("tests/tmp/all_grey.png", "tests/tmp/grey_top_black_bottom.png", pytest.lazy_fixture("all_127_bottom_mask"))
])
def test_load_labels_masked(labels, mask, expected):
    loaded_mask = src.detector.data_prep.png_to_labels(labels, mask)
    assert np.array_equal(loaded_mask.data, expected.data)
    assert np.array_equal(loaded_mask.mask, expected.mask)
    assert type(loaded_mask) == type(expected) == ma.masked_array


def test_load_labels_wrong_mask_size(all_127_no_mask, mask_small):
    with pytest.raises(ValueError):
        src.detector.data_prep.png_to_labels("tests/tmp/all_grey.png", "tests/tmp/mask_small.png")


@pytest.mark.parametrize("labels,expected", [
    ("tests/tmp/all_grey.png", pytest.lazy_fixture("all_127_no_mask")),
    ("tests/tmp/all_black.png", pytest.lazy_fixture("all_0_no_mask"))
])
def test_load_labels_no_mask(labels, expected):
    loaded = src.detector.data_prep.png_to_labels(labels)
    assert np.array_equal(loaded, expected)
    assert type(loaded) == ma.masked_array


# TODO: Include in parametrized test above by generating proper expected outcome from fixture(?)
def test_load_labels_no_mask_mixed(mixed_values):
    loaded = src.detector.data_prep.png_to_labels("tests/tmp/mixed_values.png")
    expected = np.concatenate((np.zeros((32, 2)), np.ones((32, 2))), axis=0)
    assert np.array_equal(loaded, expected)
    assert type(loaded) == ma.masked_array


def test_load_labels_failure(all_0_no_mask, all_127_no_mask):
    with pytest.raises(AssertionError):
        all_black_png = src.detector.data_prep.png_to_labels("tests/tmp/all_black.png")
        assert np.array_equal(all_black_png, all_127_no_mask)


@pytest.mark.parametrize("labels,expected", [
    ("tests/tmp/all_black.png", pytest.lazy_fixture("mask_none")),  # Array of 0s turns into array of 1s
    ("tests/tmp/all_grey.png", pytest.lazy_fixture("mask_all")),  # Array of 127s turns into array of 0s
    ("tests/tmp/grey_top_black_bottom.png", pytest.lazy_fixture("mask_top"))
    ])
def test_convert_labels(labels, expected):
    loaded = imageio.imread(labels)
    converted = src.detector.data_prep.convert_labels(loaded)
    assert np.array_equal(converted, expected)


@pytest.mark.parametrize("mask,expected", [
    ("tests/tmp/mask_all.png", pytest.lazy_fixture("mask_all")),
    ("tests/tmp/mask_none.png", pytest.lazy_fixture("mask_none")),
    ("tests/tmp/grey_top_black_bottom.png", pytest.lazy_fixture("mask_bottom")),
    ])
def test_convert_masks(mask, expected):
    loaded = imageio.imread(mask)
    converted = src.detector.data_prep.convert_mask(loaded)
    assert np.array_equal(converted, expected)


def test_wrong_label_value():
    label_wrong_value = np.arange(122, 130).reshape(2, 4).astype("uint8")
    with pytest.raises(ValueError):
        src.detector.data_prep.convert_labels(label_wrong_value)


def test_uniform_label_value_warning(all_0_no_mask):
    with pytest.warns(UserWarning):
        src.detector.data_prep.convert_labels(all_0_no_mask)


def test_wrong_mask_value(mask_all):
    with pytest.raises(ValueError):
        src.detector.data_prep.convert_mask(mask_all)


def test_uniform_mask_value_warning(mask_none):
    with pytest.warns(UserWarning):
        src.detector.data_prep.convert_mask(mask_none)


@pytest.fixture
def blocks_no_pad():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    imageio.imwrite("tests/tmp/blocks_no_pad.png", blocks_rgb)
    return blocks_rgb


@pytest.fixture
def blocks_bottom_pad():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    imageio.imwrite("tests/tmp/blocks_missing_bottom.png", blocks_rgb)
    bottom_row = np.dstack([np.zeros((1, 9))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, bottom_row), axis=0)
    return blocks_rgb


@pytest.fixture
def blocks_right_pad():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    imageio.imwrite("tests/tmp/blocks_missing_right.png", blocks_rgb)
    right_rows = np.dstack([np.zeros((6, 2))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, right_rows), axis=1)
    return blocks_rgb


@pytest.fixture
def blocks_both_pad():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    imageio.imwrite("tests/tmp/blocks_missing_both.png", blocks_rgb)
    right_rows = np.dstack([np.zeros((5, 2))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, right_rows), axis=1)
    bottom_row = np.dstack([np.zeros((1, 9))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, bottom_row), axis=0)
    return blocks_rgb


def test_png_to_features_failure(blocks_no_pad, blocks_bottom_pad):
    with pytest.raises(AssertionError):
        features = src.detector.data_prep.png_to_features("tests/tmp/blocks_no_pad.png")
        assert np.array_equal(features, blocks_bottom_pad)


@pytest.fixture(scope="module")
def blocks_mask_leftmost():
    mask_png = np.concatenate((np.zeros((6, 1)), np.ones((6, 8)) * 127), axis=1).astype('uint8')
    imageio.imwrite("tests/tmp/blocks_mask_leftmost.png", mask_png)
    mask_one_layer = np.concatenate((np.ones((6, 1)), np.zeros((6, 8))), axis=1)
    mask_array = np.dstack((mask_one_layer, mask_one_layer, mask_one_layer))
    return mask_array


def test_png_to_features_masked(blocks_no_pad, blocks_mask_leftmost):
    masked_blocks = src.detector.data_prep.png_to_features(
        "tests/tmp/blocks_no_pad.png",
        "tests/tmp/blocks_mask_leftmost.png")
    assert type(masked_blocks) == ma.masked_array
    assert np.array_equal(masked_blocks.data, blocks_no_pad.data)
    assert np.array_equal(masked_blocks.mask, blocks_mask_leftmost)


@pytest.fixture
def blocks_small_mask():
    mask_png = np.concatenate((np.ones((3, 8)) * 127, np.zeros((3, 8))), axis=0)
    imageio.imwrite("tests/tmp/blocks_small_mask.png", mask_png)


def test_png_to_features_wrong_mask_size(blocks_no_pad, blocks_small_mask):
    with pytest.raises(ValueError):
        src.detector.data_prep.png_to_features("tests/tmp/blocks_no_pad.png", "tests/tmp/blocks_small_mask.png")


@pytest.fixture
def blocks_negative_value():
    block_digits = np. array([
        [1, -1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ])
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    return blocks_rgb


def test_convert_features_negative_pixel_value(blocks_negative_value):
    with pytest.raises(ValueError):
        src.detector.data_prep.convert_features(blocks_negative_value)


@pytest.fixture
def blocks_excessive_value():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 312, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ])
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    return blocks_rgb

@pytest.fixture
def blocks_large_width_tstpad():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0],
        [4, 4, 4, 5, 5, 5, 6, 6, 6, 0, 0, 0],
        [4, 4, 4, 5, 5, 5, 6, 6, 6, 0, 0, 0],
        [4, 4, 4, 5, 5, 5, 6, 6, 6, 0, 0, 0]
    ])
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    padded_mask = np.concatenate((np.zeros((6, 9)), np.ones((6, 3))), axis=1)
    mask_rgb = np.dstack([padded_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks


def test_convert_features_excessive_pixel_value(blocks_excessive_value):
    with pytest.raises(ValueError):
        src.detector.data_prep.convert_features(blocks_excessive_value)


def test_convert_features_basic_functioning(blocks_no_pad):
    converted_blocks = src.detector.data_prep.convert_features(blocks_no_pad)
    assert np.array_equal(converted_blocks, blocks_no_pad)


def test_unique_feature_value_warning():
    grey_rgb_array = np.tile(127, (6, 9, 3))
    with pytest.warns(UserWarning):
        src.detector.data_prep.convert_features(grey_rgb_array)


def test_no_unique_feature_value_warning(blocks_no_pad):
    with pytest.warns(None) as record:
        src.detector.data_prep.convert_features(blocks_no_pad)
    assert not record


########################################################################################################################
# Padding                                                                                                              #
########################################################################################################################
@pytest.fixture
def blocks_no_pad_tstpad():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    imageio.imwrite("tests/tmp/blocks_no_pad_tstpad.png", blocks_rgb)
    padded_mask = np.zeros((6, 9))
    mask_rgb = np.dstack([padded_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks


@pytest.fixture
def blocks_bottom_pad_tstpad():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    imageio.imwrite("tests/tmp/blocks_missing_bottom_tstpad.png", blocks_rgb)
    bottom_row = np.dstack([np.zeros((1, 9))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, bottom_row), axis=0)
    padded_mask = np.concatenate((np.zeros((5, 9)), np.ones((1, 9))), axis=0)
    mask_rgb = np.dstack([padded_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks


@pytest.fixture
def blocks_right_pad_tstpad():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    imageio.imwrite("tests/tmp/blocks_missing_right_tstpad.png", blocks_rgb)
    right_rows = np.dstack([np.zeros((6, 2))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, right_rows), axis=1)
    padded_mask = np.concatenate((np.zeros((6, 7)), np.ones((6, 2))), axis=1)
    mask_rgb = np.dstack([padded_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks


@pytest.fixture
def blocks_both_pad_tstpad():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    imageio.imwrite("tests/tmp/blocks_missing_both_tstpad.png", blocks_rgb)
    right_rows = np.dstack([np.zeros((5, 2))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, right_rows), axis=1)
    bottom_row = np.dstack([np.zeros((1, 9))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, bottom_row), axis=0)
    unpadded_mask = np.zeros((5, 7))
    padded_mask = np.pad(unpadded_mask, ((0, 1), (0, 2)), 'constant', constant_values=(1,))
    mask_rgb = np.dstack([padded_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks

# TODO: Make test_pad_features_masked
# THE PROBLEM IS HERE: The expected output is a masked array, so it needs to be a different fixture.
# The original fixture was created to test convert_features, which takes a numpy array and spits out a numpy array.
# It's called by png_to_features, which is the function that turns this into a masked array.
# So what I need is a fixture that reads e.g. 'blocks no pad' and produces a masked array version of it WITH PADDING
@pytest.mark.parametrize("tile_size,img,expected", [
    ((3, 3), "tests/tmp/blocks_no_pad_tstpad.png", pytest.lazy_fixture("blocks_no_pad_tstpad")),
    ((3, 3), "tests/tmp/blocks_missing_bottom_tstpad.png", pytest.lazy_fixture("blocks_bottom_pad_tstpad")),
    ((3, 3), "tests/tmp/blocks_missing_right_tstpad.png", pytest.lazy_fixture("blocks_right_pad_tstpad")),
    ((3, 3), "tests/tmp/blocks_missing_both_tstpad.png", pytest.lazy_fixture("blocks_both_pad_tstpad")),
    ((6, 6), "tests/tmp/blocks_no_pad_tstpad.png", pytest.lazy_fixture("blocks_large_width_tstpad")),
    ((2, 4), "tests/tmp/blocks_no_pad_tstpad.png", pytest.lazy_fixture("blocks_large_width_tstpad")),
])
def test_pad_features_unmasked(tile_size, img, expected):
    img_loaded = src.detector.data_prep.png_to_features(img)
    img_padded = src.detector.data_prep.pad(img_loaded, tile_size)
    n_rows_added = expected.shape[0] - img_loaded.shape[0]
    n_cols_added = expected.shape[1] - img_loaded.shape[1]
    assert type(img_padded) == ma.masked_array
    assert type(img_loaded) == ma.masked_array
    assert np.array_equal(img_padded.data, expected.data)
    assert np.array_equal(img_padded.mask, expected.mask)
    assert img_padded.shape[0] % tile_size[0] == 0
    assert img_padded.shape[1] % tile_size[1] == 0
    if n_rows_added > 0:
        assert img_padded.data[-n_rows_added, :, ...].all() == 0
        assert img_padded.mask[-n_rows_added, :, ...].all() == 1
    if n_cols_added > 0:
        assert img_padded.data[:, -n_cols_added, ...].all() == 0
        assert img_padded.mask[:, -n_cols_added, ...].all() == 1


def test_pad_features_assertion_failure(blocks_both_pad, blocks_no_pad):
    img_loaded = src.detector.data_prep.png_to_features("tests/tmp/blocks_missing_both.png")
    img_padded = src.detector.data_prep.pad(img_loaded, (3, 3))
    img_equal = blocks_both_pad
    img_unequal = src.detector.data_prep.png_to_features("tests/tmp/blocks_no_pad.png")
    assert np.array_equal(img_padded, img_equal)
    with pytest.raises(AssertionError):
        assert np.array_equal(img_padded, img_unequal)


# TODO: Sort out that this fixture writes the same png as fixture all_127_no_mask(), which can lead to conflicts
@pytest.fixture
def padded_all_127_mask_none():
    tst_dim = (4, 8)
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("tests/tmp/all_grey.png", all_grey)
    mask_png = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("tests/tmp/mask_none.png", mask_png)
    padded_array = np.pad(np.ones((4, 8)), ((0, 2), (0, 1)), 'constant', constant_values=(0,))
    mask_unpadded = np.zeros(tst_dim)
    mask_array = np.pad(mask_unpadded, ((0, 2), (0, 1)), 'constant', constant_values=(1,))
    masked_array = ma.masked_array(padded_array, mask=mask_array)
    return masked_array


@pytest.fixture
def padded_all_127_mask_bottom():
    tst_dim = (4, 8)
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("tests/tmp/all_grey.png", all_grey)
    mask_png = np.concatenate((np.ones((2, 8)) * 127, np.zeros((2, 8))), axis=0).astype("uint8")
    imageio.imwrite("tests/tmp/mask_bottom.png", mask_png)
    padded_array = np.pad(np.ones((4, 8)), ((0, 2), (0, 1)), 'constant', constant_values=(0,))
    mask_unpadded = np.concatenate((np.zeros((2, 8)), np.ones((2, 8))), axis=0)
    mask_array = np.pad(mask_unpadded, ((0, 2), (0, 1)), 'constant', constant_values=(1,))
    masked_array = ma.masked_array(padded_array, mask=mask_array)
    return masked_array


@pytest.fixture
def padded_all_127_mask_left():
    tst_dim = (4, 8)
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("tests/tmp/all_grey.png", all_grey)
    mask_png = np.concatenate((np.zeros((4, 4)), np.ones((4, 4)) * 127), axis=1).astype("uint8")
    imageio.imwrite("tests/tmp/mask_left.png", mask_png)
    padded_array = np.pad(np.ones((4, 8)), ((0, 2), (0, 1)), 'constant', constant_values=(0,))
    mask_unpadded = np.concatenate((np.ones((4, 4)), np.zeros((4, 4))), axis=1)
    mask_array = np.pad(mask_unpadded, ((0, 2), (0, 1)), 'constant', constant_values=(1,))
    masked_array = ma.masked_array(padded_array, mask=mask_array)
    return masked_array


@pytest.fixture
def padded_all_127_mask_all():
    tst_dim = (4, 8)
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("tests/tmp/all_grey.png", all_grey)
    mask_png = np.zeros(tst_dim).astype("uint8")
    imageio.imwrite("tests/tmp/mask_all.png", mask_png)
    padded_array = np.pad(np.ones((4, 8)), ((0, 2), (0, 1)), 'constant', constant_values=(0,))
    mask_unpadded = np.ones(tst_dim)
    mask_array = np.pad(mask_unpadded, ((0, 2), (0, 1)), 'constant', constant_values=(1,))
    masked_array = ma.masked_array(padded_array, mask=mask_array)
    return masked_array


# TODO: Add different images
@pytest.mark.parametrize("tile_size,img,expected", [
    ((3, 3), "tests/tmp/all_grey.png", pytest.lazy_fixture("padded_all_127_mask_none"))
])
def test_pad_labels_unmasked(tile_size, img, expected):
    img_loaded = src.detector.data_prep.png_to_labels(img)
    img_padded = src.detector.data_prep.pad(img_loaded, tile_size)
    n_rows_added = expected.shape[0] - img_loaded.shape[0]
    n_cols_added = expected.shape[1] - img_loaded.shape[1]
    assert np.array_equal(img_padded.data, expected.data)
    assert np.array_equal(img_padded.mask, expected.mask)
    assert img_padded.shape[0] % tile_size[0] == 0
    assert img_padded.shape[1] % tile_size[1] == 0
    if n_rows_added > 0:
        assert img_padded.data[-n_rows_added, :, ...].all() == 0
    if n_cols_added > 0:
        assert img_padded.data[:, -n_cols_added, ...].all() == 0
    if n_rows_added > 0:
        assert img_padded.mask[-n_rows_added, :, ...].all() == 1
    if n_cols_added > 0:
        assert img_padded.mask[:, -n_cols_added, ...].all() == 1


@pytest.mark.parametrize("tile_size,img,mask,expected", [
    ((3, 3), "tests/tmp/all_grey.png", 'tests/tmp/mask_none.png', pytest.lazy_fixture("padded_all_127_mask_none")),
    ((3, 3), "tests/tmp/all_grey.png", 'tests/tmp/mask_all.png', pytest.lazy_fixture("padded_all_127_mask_all")),
    ((3, 3), "tests/tmp/all_grey.png", 'tests/tmp/mask_left.png', pytest.lazy_fixture("padded_all_127_mask_left")),
    ((3, 3), "tests/tmp/all_grey.png", 'tests/tmp/mask_bottom.png', pytest.lazy_fixture("padded_all_127_mask_bottom"))
])
def test_pad_labels_masked(tile_size, img, mask, expected):
    img_loaded = src.detector.data_prep.png_to_labels(img, mask)
    img_padded = src.detector.data_prep.pad(img_loaded, tile_size)
    n_rows_added = expected.shape[0] - img_loaded.shape[0]
    n_cols_added = expected.shape[1] - img_loaded.shape[1]
    print("\nimg", img_padded.data)
    print("\nexpected", expected.data)
    assert np.array_equal(img_padded.data, expected.data)
    assert np.array_equal(img_padded.mask, expected.mask)
    assert img_padded.shape[0] % tile_size[0] == 0
    assert img_padded.shape[1] % tile_size[1] == 0
    if n_rows_added > 0:
        assert img_padded.data[-n_rows_added, :, ...].all() == 0
        assert img_padded.mask[-n_rows_added, :, ...].all() == 1
    if n_cols_added > 0:
        assert img_padded.data[:, -n_cols_added, ...].all() == 0
        assert img_padded.mask[:, -n_cols_added, ...].all() == 1

########################################################################################################################
# Tiling & Sampling                                                                                                    #
########################################################################################################################

# 1) Make a coordinate list that identifies tiles alongside tile_size
# 2) Mark the tiles that have at least one slum pixel.
# 3) Sample separately from slum and non-slum tiles, then combine into the two buckets into test, validation, train sets
# 4)


########################################################################################################################
# Integration                                                                                                          #
########################################################################################################################
