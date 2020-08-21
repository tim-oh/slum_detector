import pytest
import src.detector.data_prep
import numpy as np
import numpy.ma as ma
import imageio

tst_dim = (2, 8)
tile_size = (3, 3)

########################################################################################################################
# Loading & conversion                                                                                                 #
########################################################################################################################
# TODO: Refactor boilerplate fixtures, perhaps with a fixture generator



# TODO: Array of masked black pixels wrongly evaluates same as unmasked one.
# TODO: Tests fail unless tests/xxx.png exists already, so have to py.test twice. Fix via fixture scopes?
@pytest.mark.parametrize("labels,mask,expected", [
    ("tests/all_black.png", "tests/mask_none.png", pytest.lazy_fixture("all_0_no_mask")),
    ("tests/all_black.png", "tests/mask_all.png", pytest.lazy_fixture("all_0_no_mask")),
    ("tests/all_grey.png", "tests/mask_none.png", pytest.lazy_fixture("all_127_no_mask")),
    ("tests/all_grey.png", "tests/mask_all.png", pytest.lazy_fixture("all_127_full_mask")),
    ("tests/all_grey.png", "tests/grey_top_black_bottom.png", pytest.lazy_fixture("all_127_bottom_mask"))
])
def test_load_labels_masked(labels, mask, expected):
    loaded_mask = src.detector.data_prep.png_to_labels(labels, mask)
    assert np.array_equal(loaded_mask, expected)
    assert type(loaded_mask) == ma.masked_array


def test_load_labels_wrong_mask_size(all_127_no_mask, mask_small):
    with pytest.raises(ValueError):
        src.detector.data_prep.png_to_labels("tests/all_grey.png", "tests/mask_small.png")


@pytest.mark.parametrize("labels,expected", [
    ("tests/all_grey.png", pytest.lazy_fixture("all_127_no_mask")),
    ("tests/all_black.png", pytest.lazy_fixture("all_0_no_mask"))
])
def test_load_labels_no_mask(labels, expected):
    loaded = src.detector.data_prep.png_to_labels(labels)
    assert np.array_equal(loaded, expected)
    assert type(loaded) == ma.masked_array

# TODO: Include in parametrized test above by generating proper expected outcome from fixture(?)
def test_load_labels_no_mask_mixed(mixed_values):
    loaded = src.detector.data_prep.png_to_labels("tests/mixed_values.png")
    expected = np.concatenate((np.zeros((32, 2)), np.ones((32, 2))), axis=0)
    assert np.array_equal(loaded, expected)
    assert type(loaded) == ma.masked_array


def test_load_labels_failure(all_0_no_mask, all_127_no_mask):
    with pytest.raises(AssertionError):
        all_black_png = src.detector.data_prep.png_to_labels("tests/all_black.png")
        assert np.array_equal(all_black_png, all_127_no_mask)


# TODO: Test_all_slum: Convert a file with all 127 to get mask_none, but throw warning.
# TODO: Test_no_slum: Convert a file with all 0 to get mask_all, but throw warning.
# TODO: Test_slum_left_no_slum_right: Convert a file that has a mix and see the correct result.


@pytest.mark.parametrize("labels,expected", [
    ("tests/all_black.png", pytest.lazy_fixture("mask_none")), # Array of 0s turns into array of 1s
    ("tests/all_grey.png", pytest.lazy_fixture("mask_all")), # Array of 127s turns into array of 0s
    ("tests/grey_top_black_bottom.png", pytest.lazy_fixture("mask_top"))
    ])
def test_convert_labels(labels, expected):
    loaded = imageio.imread(labels)
    converted = src.detector.data_prep.convert_labels(loaded)
    assert np.array_equal(converted, expected)


@pytest.mark.parametrize("mask,expected", [
    ("tests/mask_all.png", pytest.lazy_fixture("mask_all")),
    ("tests/mask_none.png", pytest.lazy_fixture("mask_none")),
    ("tests/grey_top_black_bottom.png", pytest.lazy_fixture("mask_bottom")),
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


# TODO: Warning not working properly: fix that and add a test that doesn't throw a warning when there are two values
def test_uniform_mask_value_warning(mask_none):
    with pytest.warns(UserWarning):
        src.detector.data_prep.convert_mask(mask_none)


@pytest.mark.xfail
@pytest.mark.parametrize("tile_size,img,expected", [
    ((3, 3), "tests/tmp/blocks_no_pad.png", pytest.lazy_fixture("blocks_no_pad")),
    ((3, 3), "tests/tmp/blocks_bottom_pad.png", pytest.lazy_fixture("blocks_bottom_pad")),
    ((3, 3), "tests/tmp/blocks_right_pad.png", pytest.lazy_fixture("blocks_right_pad")),
    ((3, 3), "tests/tmp/blocks_both_pad.png", pytest.lazy_fixture("blocks_both_pad"))
])
def test_load_features_unmasked(tile_size, img, expected):
    img_loaded = src.detector.data_prep.png_to_features(img)
    img_padded = src.detector.data_prep.pad_features(img_loaded, tile_size)
    assert np.array_equal(img_padded, expected)


@pytest.mark.xfail
@pytest.mark.parametrize("img,mask,expected", [
    ("IMG PATH", "MASK PATH", pytest.lazy_fixture("FIXTURE")),
    ("IMG PATH", "MASK PATH", pytest.lazy_fixture("FIXTURE"))
])
def test_load_features_masked(img, mask, expected):
    img_loaded = src.detector.data_prep.load_image(img, mask)
    img_padded = src.detector.data_prep.pad_image(img_loaded, tile_size)
    assert np.array_equal(img_padded, expected)

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
    imageio.imwrite("tests/blocks_no_pad.png", blocks_rgb)
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
    imageio.imwrite("tests/block_features_bottom.png", blocks_rgb)
    return blocks_rgb


def test_png_to_features_failure(blocks_no_pad, blocks_bottom_pad):
    with pytest.raises(AssertionError):
        features = src.detector.data_prep.png_to_features("tests/blocks_no_pad.png")
        assert np.array_equal(features, blocks_bottom_pad)


@pytest.fixture(scope="module")
def blocks_mask_leftmost():
    mask_png_1d = np.concatenate((np.zeros((6,1)), np.ones((6, 8)) * 127), axis=1).astype('uint8')
    mask_png_3d = np.dstack((mask_png_1d, mask_png_1d, mask_png_1d))
    imageio.imwrite("tests/blocks_mask_leftmost.png", mask_png_3d)
    mask_one_layer = np.concatenate((np.ones((6,1)), np.zeros((6, 8))), axis=1)
    mask_array = np.dstack((mask_one_layer, mask_one_layer, mask_one_layer))
    return mask_array


def test_png_to_features_masked(blocks_no_pad, blocks_mask_leftmost):
    masked_blocks = src.detector.data_prep.png_to_features(
        "tests/blocks_no_pad.png",
        "tests/blocks_mask_leftmost.png")
    assert type(masked_blocks) == ma.masked_array
    assert np.array_equal(masked_blocks.data, blocks_no_pad.data)
    assert np.array_equal(masked_blocks.mask, blocks_mask_leftmost)


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


def test_load_features_negative_pixel_value(blocks_negative_value):
    with pytest.raises(ValueError):
        src.detector.data_prep.convert_features(blocks_negative_value)


@pytest.fixture
def blocks_excessive_value():
    block_digits = np. array([
        [1, 312, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ])
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    return blocks_rgb


def test_load_features_excessive_pixel_value(blocks_excessive_value):
    with pytest.raises(ValueError):
        src.detector.data_prep.convert_features(blocks_excessive_value)


def test_convert_features(blocks_no_pad):
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

# WRITE TEST FOR WARNING MESSAGE THAT ALL VALUES ARE THE SAME
# Take a .png and turn it into a masked numpy array.
# The mask is all zeros if no file is provided, or like the evaluation one.
# Test: Correct array size, type,

# def png_to_features(feature_png, mask). Note that png_to_labels already exists, model it on that.
# It's the same mask for labels and features.

########################################################################################################################
# Padding                                                                                                              #
########################################################################################################################

# Pad features and labels to get consistent tile size, but mask the padded bits.


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
