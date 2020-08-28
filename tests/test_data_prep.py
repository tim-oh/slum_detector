import pytest
import src.detector.data_prep
import numpy as np
import numpy.ma as ma
import imageio

# TODO: Turn integration tests (e.g. for  proper unit tests for the

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


def test_png_to_features_failure(blocks_no_pad, blocks_bottom_pad):
    with pytest.raises(AssertionError):
        features = src.detector.data_prep.png_to_features("tests/tmp/blocks_no_pad.png")
        assert np.array_equal(features, blocks_bottom_pad)


def test_png_to_features_masked(blocks_no_pad, blocks_mask_leftmost):
    masked_blocks = src.detector.data_prep.png_to_features(
        "tests/tmp/blocks_no_pad.png",
        "tests/tmp/blocks_mask_leftmost.png")
    assert type(masked_blocks) == ma.masked_array
    assert np.array_equal(masked_blocks.data, blocks_no_pad.data)
    assert np.array_equal(masked_blocks.mask, blocks_mask_leftmost)


def test_png_to_features_wrong_mask_size(blocks_no_pad, blocks_small_mask):
    with pytest.raises(ValueError):
        src.detector.data_prep.png_to_features("tests/tmp/blocks_no_pad.png", "tests/tmp/blocks_small_mask.png")


def test_convert_features_negative_pixel_value(blocks_negative_value):
    with pytest.raises(ValueError):
        src.detector.data_prep.convert_features(blocks_negative_value)


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


# TODO: Make test_pad_features_masked
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


# TODO: Test for different tile sizes and labels
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


# TODO: Fixtures for different tile sizes
@pytest.mark.parametrize("tile_size,img,mask,expected", [
    ((3, 3),
     "tests/tmp/blocks_no_pad_tstpad.png",
     "tests/tmp/blocks_no_pad_mask_leftmost.png",
     pytest.lazy_fixture("blocks_no_pad_mask_leftmost")),
    ((3, 3),
     "tests/tmp/blocks_missing_bottom_tstpad.png",
     "tests/tmp/blocks_bottom_pad_mask_top.png",
     pytest.lazy_fixture("blocks_bottom_pad_mask_top")),
    ((3, 3),
     "tests/tmp/blocks_missing_right_tstpad.png",
     "tests/tmp/blocks_right_pad_mask_bottom.png",
     pytest.lazy_fixture("blocks_right_pad_mask_bottom")),
    ((3, 3),
     "tests/tmp/blocks_missing_both_tstpad.png",
     "tests/tmp/blocks_both_pad_mask_right.png",
     pytest.lazy_fixture("blocks_both_pad_mask_right"))
])
def test_pad_features_masked(tile_size, img, mask, expected):
    img_loaded = src.detector.data_prep.png_to_features(img, mask)
    img_padded = src.detector.data_prep.pad(img_loaded, tile_size)
    n_rows_added = expected.shape[0] - img_loaded.shape[0]
    n_cols_added = expected.shape[1] - img_loaded.shape[1]
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
# Tiling & Splitting                                                                                                    #
########################################################################################################################
# TODO: Add asserts for correct array sizing, types, consistent tile size etc
@pytest.mark.parametrize("tile_size,features,expected", [
    ((3, 3),
     pytest.lazy_fixture("features_6x12_masknone"),
     pytest.lazy_fixture("coordinates_6x12_tile_3x3_mask_none")),
    ((2, 2),
     pytest.lazy_fixture("features_6x12_masknone"),
     pytest.lazy_fixture("coordinates_6x12_tile_2x2_mask_none")),
    ((2, 2),
     pytest.lazy_fixture("features_6x12_mask_topleft6x4"),
     pytest.lazy_fixture("coordinates_6x12_tile_2x2_mask_none"))
])
def test_tile_coordinates(tile_size, features, expected):
    coordinates = src.detector.data_prep.tile_coordinates(features, tile_size)
    assert np.array_equal(coordinates, expected)


# TODO: Parametrize and add a small fixture with two different tile sizes, add test below that gives same result
@pytest.mark.parametrize("image,coordinates,expected", [
    (pytest.lazy_fixture("features_6x12_masknone"),
     pytest.lazy_fixture("coordinates_6x12_tile_3x3_mask_none"),
     pytest.lazy_fixture("features_6x12_masknone_tiled_3x3")),
    (pytest.lazy_fixture("features_6x12_mask_topleft6x4"),
     pytest.lazy_fixture("coordinates_6x12_tile_3x3_mask_none"),
     pytest.lazy_fixture("features_6x12_mask_topleft6x4_tiled_3x3"))
])
def test_stack_nonmasked_tiles(image, coordinates, expected):
    tiled = src.detector.data_prep.stack_tiles(image, coordinates)
    assert np.array_equal(tiled.data, expected.data)
    print("expected", expected.mask[:, :, 1, 4])
    print("tiled", tiled.mask[:, :, 1, 4])
    assert np.array_equal(tiled.mask, expected.mask)



# @pytest.mark.xfail
# @pytest.mark.parametrize("tile_size,features,mask,expected", [
#     ((3, 3), PADDED_FEATURE_ARRAY_FIXTURE, PADDED_MASK_ARRAY_FIXTURE, EXPECTED_ARRAY_FIXTURE),
#     ((3, 3), PADDED_FEATURE_ARRAY_FIXTURE, PADDED_MASK_ARRAY_FIXTURE, EXPECTED_ARRAY_FIXTURE)
# ])
# def test_tile_masked_nolabel_nosplit_nosave(tile_size, features, mask, expected):
#     tiled_image = src.detector.data_prep.tile(features, tile_size)
#     assert np.array_equal(tiled_image, expected)
#
#
# @pytest.mark.xfail
# @pytest.mark.parametrize("tile_size,features,labels,expected", [
#     ((3, 3), PADDED_FEATURE_ARRAY_FIXTURE, EXPECTED_ARRAY_FIXTURE),
#     ((3, 3), PADDED_FEATURE_ARRAY_FIXTURE, EXPECTED_ARRAY_FIXTURE)
# ])
# def test_tile_nomask_labelled_nosplit_nosave(tile_size, features, labels, expected):
#     tiled_image = src.detector.data_prep.tile(features, tile_size)
#     assert np.array_equal(tiled_image, expected)
#
#
# @pytest.mark.xfail
# @pytest.mark.parametrize("tile_size,features,labels,mask,expected", [
#     ((3, 3), PADDED_FEATURE_ARRAY_FIXTURE, EXPECTED_ARRAY_FIXTURE),
#     ((3, 3), PADDED_FEATURE_ARRAY_FIXTURE, EXPECTED_ARRAY_FIXTURE)
# ])
# def test_tile_masked_labelled_nosplit_nosave(tile_size, features, labels, mask, expected):
#     tiled_image = src.detector.data_prep.tile(feature_array, tile_size)
#     assert np.array_equal(tiled_image, expected)
#
#
# @pytest.mark.xfail
# @pytest.mark.parametrize("tile_size,features,mask,labels,splits,expected", [
#     ((3, 3), PADDED_FEATURE_ARRAY_FIXTURE, EXPECTED_ARRAY_FIXTURE),
#     ((3, 3), PADDED_FEATURE_ARRAY_FIXTURE, EXPECTED_ARRAY_FIXTURE)
# ])
# def test_tile_masked_labelled_splits_nosave(tile_size, features, mask, labels, splits, expected):
#     tiled_image = src.detector.data_prep.tile(feature_array, tile_size)
#     assert np.array_equal(tiled_image, expected)
#
#
# @pytest.mark.xfail
# @pytest.mark.parametrize("tile_size,features,mask,labels,splits,path,expected", [
#     ((3, 3), PADDED_FEATURE_ARRAY_FIXTURE, EXPECTED_ARRAY_FIXTURE),
#     ((3, 3), PADDED_FEATURE_ARRAY_FIXTURE, EXPECTED_ARRAY_FIXTURE)
# ])
# def test_tile_masked_labelled_splits_saves(tile_size, features, mask, labels, splits, path, expected):
#     tiled_image = src.detector.data_prep.tile(feature_array, tile_size)
#     assert np.array_equal(tiled_image, expected)
#
#
# @pytest.mark.xfail
# @pytest.mark.parametrize("tile_size,features,mask,path,expected", [
#     ((3, 3), PADDED_FEATURE_ARRAY_FIXTURE, EXPECTED_ARRAY_FIXTURE),
#     ((3, 3), PADDED_FEATURE_ARRAY_FIXTURE, EXPECTED_ARRAY_FIXTURE)
# ])
# def test_tile_masked_nolabel_noplit_saves(tile_size, features, mask, path, expected):
#     tiled_image = src.detector.data_prep.tile(features, tile_size)
#     assert np.array_equal(tiled_image, expected)

# def test_tile
# Behaviour:
# Input case I
# Inputs: image, tile_size, kwarg: labels, mask, train/val/test_split, path
# Output I: padded, masked features in tile_size with mask_none, if only image (1 file)
# Output II: padded, masked features in tile_size with given mask, if mask= (1 file)
# Output III: padded, masked features and labels in tile_size, if labels= (2 files)
# Output IV: padded, masked features and labels in tile_size, if labels= and mask= (2 files)
# If train/val/test split: only permitted if features are provided, error otherwise.
# If path: np.savez outputs to that path.
# Feature: No all-masked tiles.
# Feature: Stratified sampling.

# Input: padded png
# Tiling function - command: tile(img, tile_size), output: list of top left corner coordinates, w/o fully masked tiles
# Split function - command: split(coordinates, splits), output:


# 3) Sample separately from slum and non-slum tiles, then combine into the two buckets into test, validation, train sets
# 4)


########################################################################################################################
# Integration                                                                                                          #
########################################################################################################################
