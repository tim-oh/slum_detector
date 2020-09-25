import pytest
from pytest_lazyfixture import lazy_fixture
import src.detector.data_prep
import numpy as np
import numpy.ma as ma
import imageio
import os

########################################################################################################################
# Loading & conversion                                                                                                 #
########################################################################################################################
# TODO: These tests fail unless tests/xxx.png exists already, so have to py.test twice.
@pytest.mark.parametrize("labels,mask,expected", [
    (lazy_fixture("all_0_no_mask_png"), lazy_fixture("mask_none_png"), lazy_fixture("all_0_no_mask_array")),
    (lazy_fixture("all_0_mask_all_png"), lazy_fixture("mask_all_png"), lazy_fixture("all_0_mask_all_array")),
    (lazy_fixture("all_127_no_mask_png"), lazy_fixture("mask_none_png"), lazy_fixture("all_127_no_mask_array")),
    (lazy_fixture("all_127_full_mask_png"), lazy_fixture("mask_all_png"), lazy_fixture("all_127_full_mask_array")),
    (lazy_fixture("all_127_bottom_mask_png"), lazy_fixture("mask_bottom_png"),lazy_fixture("all_127_bottom_mask_array"))
])
def test_load_labels_masked(labels, mask, expected):
    loaded_mask = src.detector.data_prep._png_to_labels(labels, mask)
    assert np.array_equal(loaded_mask.data, expected.data)
    assert np.array_equal(loaded_mask.mask, expected.mask)
    assert type(loaded_mask) == type(expected) == ma.masked_array


def test_load_labels_wrong_mask_size(all_127_no_mask_png, mask_small_png):
    with pytest.raises(ValueError):
        src.detector.data_prep._png_to_labels(all_127_no_mask_png, mask_small_png)


@pytest.mark.parametrize("labels,expected", [
    (lazy_fixture("all_127_no_mask_png"), lazy_fixture("all_127_no_mask_array")),
    (lazy_fixture("all_0_no_mask_png"), lazy_fixture("all_0_no_mask_array"))
])
def test_load_labels_no_mask(labels, expected):
    loaded = src.detector.data_prep._png_to_labels(labels)
    assert np.array_equal(loaded, expected)
    assert type(loaded) == ma.masked_array


# TODO: Include in parametrized test above by adjusting mixed_values fixture
def test_load_labels_no_mask_mixed(mixed_values_png):
    loaded = src.detector.data_prep._png_to_labels(mixed_values_png)
    expected = np.concatenate((np.zeros((32, 2)), np.ones((32, 2))), axis=0)
    assert np.array_equal(loaded, expected)
    assert type(loaded) == ma.masked_array


def test_load_labels_failure(all_0_no_mask_png, all_127_no_mask_array):
    with pytest.raises(AssertionError):
        all_black_png = src.detector.data_prep._png_to_labels(all_0_no_mask_png)
        assert np.array_equal(all_black_png, all_127_no_mask_array)


@pytest.mark.parametrize("labels,expected", [
    (lazy_fixture("mask_none_png"), lazy_fixture("mask_all_array")),  # Array of 0s turns into array of 1s
    (lazy_fixture("mask_none_png"), lazy_fixture("mask_all_array")),  # Array of 127s turns into array of 0s
    (lazy_fixture("mask_bottom_png"), lazy_fixture("mask_top_array"))
    ])
def test_convert_labels(labels, expected):
    loaded = imageio.imread(labels)
    converted = src.detector.data_prep._convert_labels(loaded)
    assert np.array_equal(converted, expected)


@pytest.mark.parametrize("mask,expected", [
    (lazy_fixture("mask_all_png"), lazy_fixture("mask_all_array")),
    (lazy_fixture("mask_none_png"), lazy_fixture("mask_none_array")),
    (lazy_fixture("mask_bottom_png"), lazy_fixture("mask_bottom_array")),
    ])
def test_convert_masks(mask, expected):
    loaded = imageio.imread(mask)
    converted = src.detector.data_prep._convert_mask(loaded)
    assert np.array_equal(converted, expected)


def test_wrong_label_value():
    label_wrong_value = np.arange(122, 130).reshape(2, 4).astype("uint8")
    with pytest.raises(ValueError):
        src.detector.data_prep._convert_labels(label_wrong_value)


def test_uniform_label_value_warning(all_0_no_mask_array):
    with pytest.warns(UserWarning):
        src.detector.data_prep._convert_labels(all_0_no_mask_array)


def test_wrong_mask_value(mask_all_array):
    with pytest.raises(ValueError):
        src.detector.data_prep._convert_mask(mask_all_array)


def test_uniform_mask_value_warning(mask_none_array):
    with pytest.warns(UserWarning):
        src.detector.data_prep._convert_mask(mask_none_array)


def test_png_to_features_failure(blocks_no_pad_png, blocks_bottom_pad_array):
    with pytest.raises(AssertionError):
        features = src.detector.data_prep._png_to_features(blocks_no_pad_png)
        assert np.array_equal(features, blocks_bottom_pad_array)


def test_png_to_features_masked(blocks_no_pad_png,
                                blocks_mask_leftmost_png,
                                blocks_no_pad_array,
                                blocks_mask_leftmost_array):
    masked_blocks = src.detector.data_prep._png_to_features(
        blocks_no_pad_png,
        blocks_mask_leftmost_png)
    assert type(masked_blocks) == ma.masked_array
    assert np.array_equal(masked_blocks.data, blocks_no_pad_array.data)
    assert np.array_equal(masked_blocks.mask, blocks_mask_leftmost_array)


def test_png_to_features_wrong_mask_size(blocks_no_pad_png, blocks_small_mask_png):
    with pytest.raises(ValueError):
        src.detector.data_prep._png_to_features(blocks_no_pad_png, blocks_small_mask_png)


def test_convert_features_negative_pixel_value(blocks_negative_value_array):
    with pytest.raises(ValueError):
        src.detector.data_prep._convert_features(blocks_negative_value_array)


def test_convert_features_excessive_pixel_value(blocks_excessive_value_array):
    with pytest.raises(ValueError):
        src.detector.data_prep._convert_features(blocks_excessive_value_array)


def test_convert_features_basic_functioning(blocks_no_pad_array):
    converted_blocks = src.detector.data_prep._convert_features(blocks_no_pad_array)
    assert np.array_equal(converted_blocks, blocks_no_pad_array)


def test_unique_feature_value_warning():
    grey_rgb_array = np.tile(127, (6, 9, 3))
    with pytest.warns(UserWarning):
        src.detector.data_prep._convert_features(grey_rgb_array)


def test_no_unique_feature_value_warning(blocks_no_pad_array):
    with pytest.warns(None) as record:
        print(np.unique(blocks_no_pad_array))
        src.detector.data_prep._convert_features(blocks_no_pad_array)
    assert not record


########################################################################################################################
# Padding                                                                                                              #
########################################################################################################################


@pytest.mark.parametrize("tile_size,img,expected", [
    ((3, 3), lazy_fixture("blocks_no_pad_tstpad_png"), lazy_fixture("blocks_no_pad_tstpad_array")),
    ((3, 3), lazy_fixture("blocks_bottom_pad_tstpad_png"), lazy_fixture("blocks_bottom_pad_tstpad_array")),
    ((3, 3), lazy_fixture("blocks_right_pad_tstpad_png"), lazy_fixture("blocks_right_pad_tstpad_array")),
    ((3, 3), lazy_fixture("blocks_both_pad_tstpad_png"), lazy_fixture("blocks_both_pad_tstpad_array")),
    ((6, 6), lazy_fixture("blocks_no_pad_tstpad_png"), lazy_fixture("blocks_large_width_tstpad_array")),
    ((2, 4), lazy_fixture("blocks_no_pad_tstpad_png"), lazy_fixture("blocks_large_width_tstpad_array")),
])
def test_pad_features_unmasked(tile_size, img, expected):
    img_loaded = src.detector.data_prep._png_to_features(img)
    img_padded = src.detector.data_prep._pad(img_loaded, tile_size)
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


def test_pad_features_assertion_failure(blocks_both_pad_png, blocks_both_pad_array, blocks_no_pad_png):
    img_loaded = src.detector.data_prep._png_to_features(blocks_both_pad_png)
    img_padded = src.detector.data_prep._pad(img_loaded, (3, 3))
    img_equal = blocks_both_pad_array
    img_unequal = src.detector.data_prep._png_to_features(blocks_no_pad_png)
    assert np.array_equal(img_padded, img_equal)
    with pytest.raises(AssertionError):
        assert np.array_equal(img_padded, img_unequal)


# TODO: Add more images
@pytest.mark.parametrize("tile_size,img,expected", [
    ((3, 3), lazy_fixture("padded_all_127_mask_none_png"), lazy_fixture("padded_all_127_mask_none_array"))
])
def test_pad_labels_unmasked(tile_size, img, expected):
    img_loaded = src.detector.data_prep._png_to_labels(img)
    img_padded = src.detector.data_prep._pad(img_loaded, tile_size)
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
    ((3, 3),
     lazy_fixture("all_grey_png"),
     lazy_fixture("mask_none_png"),
     lazy_fixture("padded_all_127_mask_none_array")),
    ((3, 3),
     lazy_fixture("all_grey_png"),
     lazy_fixture("mask_all_png"),
     lazy_fixture("padded_all_127_mask_all_array")),
    ((3, 3),
     lazy_fixture("all_grey_png"),
     lazy_fixture("mask_left_png"),
     lazy_fixture("padded_all_127_mask_left_array")),
    ((3, 3),
     lazy_fixture("all_grey_png"),
     lazy_fixture("mask_bottom_png"),
     lazy_fixture("padded_all_127_mask_bottom_array"))
])
def test_pad_labels_masked(tile_size, img, mask, expected):
    img_loaded = src.detector.data_prep._png_to_labels(img, mask)
    img_padded = src.detector.data_prep._pad(img_loaded, tile_size)
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
     lazy_fixture("blocks_bottom_pad_tstpad_png"),
     lazy_fixture("mask_top_png"),
     lazy_fixture("blocks_bottom_pad_mask_top_array")),
    ((3, 3),
     lazy_fixture("blocks_missing_right_tstpad_png"),
     lazy_fixture("mask_bottom_tstpad_png"),
     lazy_fixture("blocks_right_pad_mask_bottom_array"))
])
def test_pad_features_masked(tile_size, img, mask, expected):
    img_loaded = src.detector.data_prep._png_to_features(img, mask)
    img_padded = src.detector.data_prep._pad(img_loaded, tile_size)
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
# Tiling & Splitting                                                                                                   #
########################################################################################################################
# TODO: Add asserts for correct array sizing, types, consistent tile size etc
@pytest.mark.parametrize("tile_size,features,expected", [
    ((3, 3),
     lazy_fixture("features_6x12_masknone"),
     lazy_fixture("coordinates_6x12_tile_3x3")),
    ((2, 2),
     lazy_fixture("features_6x12_masknone"),
     lazy_fixture("coordinates_6x12_tile_2x2")),
    ((2, 2),
     lazy_fixture("features_6x12_mask_topleft6x4"),
     lazy_fixture("coordinates_6x12_tile_2x2"))
])
def test_tile_coordinates(tile_size, features, expected):
    coordinates = src.detector.data_prep._tile_coordinates(features, tile_size)
    assert np.array_equal(coordinates, expected)


# TODO: Add a small fixture with a different tile size
@pytest.mark.parametrize("image,coordinates,expected", [
    (lazy_fixture("features_6x12_masknone"),
     lazy_fixture("coordinates_6x12_tile_3x3"),
     lazy_fixture("features_6x12_masknone_tiled_3x3")),
    (lazy_fixture("features_6x12_mask_topleft6x4"),
     lazy_fixture("coordinates_6x12_tile_3x3"),
     lazy_fixture("features_6x12_mask_topleft6x4_tiled_3x3")),
    (lazy_fixture("labels_6x12_mask_topleft6x4"),
     lazy_fixture("coordinates_6x12_tile_3x3"),
     lazy_fixture("labels_6x12_masked_6x4_tiled_3x3"))
])
def test_stack_tiles(image, coordinates, expected):
    tiled = src.detector.data_prep._stack_tiles(image, coordinates)
    assert np.array_equal(tiled.data, expected.data)
    assert np.array_equal(tiled.mask, expected.mask)


def test_clean_stack_features(features_6x12_mask_topleft6x4_tiled_3x3, features_6x12_masked_6x4_cleaned_tiled_3x3):
    stack_all = features_6x12_mask_topleft6x4_tiled_3x3
    expected = features_6x12_masked_6x4_cleaned_tiled_3x3
    stack_cleaned = src.detector.data_prep._clean_stack(stack_all)
    assert np.array_equal(stack_cleaned.data, expected.data)
    assert np.array_equal(stack_cleaned.mask, expected.mask)


def test_clean_stack_labels(labels_6x12_masked_6x4_tiled_3x3, labels_6x12_masked_6x4_cleaned_tiled_3x3):
    stack_all = labels_6x12_masked_6x4_tiled_3x3
    expected = labels_6x12_masked_6x4_cleaned_tiled_3x3
    stack_cleaned = src.detector.data_prep._clean_stack(stack_all)
    assert np.array_equal(stack_cleaned.data, expected.data)
    assert np.array_equal(stack_cleaned.mask, expected.mask)


def test_mark_slum_tiles(labels_6x12_masked_6x4_cleaned_tiled_3x3, slum_tile_marker):
    slum_tiles = src.detector.data_prep._mark_slum_tiles(labels_6x12_masked_6x4_cleaned_tiled_3x3)
    expected = slum_tile_marker
    assert np.array_equal(slum_tiles, expected)


@pytest.mark.parametrize("tiles,splits,expected_set_sizes", [
    (lazy_fixture("labels_6x12_masked_6x4_cleaned_tiled_3x3"), (0.6, 0.2, 0.2), (4, 1, 1)),
    (lazy_fixture("features_6x12_masked_6x4_cleaned_tiled_3x3"), (1, 0, 0), (6, 0, 0)),
    (lazy_fixture("labels_6x12_masked_6x4_cleaned_tiled_3x3"), (0, 1, 0), (0, 6, 0)),
    (lazy_fixture("features_6x12_masked_6x4_cleaned_tiled_3x3"), (0, 0, 1), (0, 0, 6)),
    (lazy_fixture("labels_6x12_masked_6x4_cleaned_tiled_3x3"), (0.5, 0.5, 0.0), (3, 3, 0)),
    (lazy_fixture("features_6x12_masked_6x4_cleaned_tiled_3x3"), (0.5, 0, 0.5), (3, 0, 3)),
    (lazy_fixture("labels_6x12_masked_6x4_cleaned_tiled_3x3"), (0, 0.5, 0.5), (0, 3, 3)),
    (lazy_fixture("features_6x12_masked_6x4_cleaned_tiled_3x3"), (0.9, 0, 0.1), (5, 0, 1))
])
def test_split_tiles_basic(tiles, splits, expected_set_sizes):
    n_tiles = tiles.shape[3]
    train_indices, val_indices, test_indices = src.detector.data_prep._split_tiles(n_tiles, splits)
    assert len(train_indices) == expected_set_sizes[0]
    assert len(val_indices) == expected_set_sizes[1]
    assert len(test_indices) == expected_set_sizes[2]
    assert len(train_indices) + len(val_indices) + len(test_indices) == n_tiles


@pytest.mark.parametrize("tiles,splits", [
    (lazy_fixture("labels_6x12_masked_6x4_cleaned_tiled_3x3"), (2, 0, 0)),
    (lazy_fixture("features_6x12_masked_6x4_cleaned_tiled_3x3"), (-0.2, 0.2, 1)),
    (lazy_fixture("labels_6x12_masked_6x4_cleaned_tiled_3x3"), (0.4, 0.4, 0.1)),
    (lazy_fixture("labels_6x12_masked_6x4_cleaned_tiled_3x3"), (0, 0, 0)),
    (lazy_fixture("labels_6x12_masked_6x4_cleaned_tiled_3x3"), (0.4, 0.4, 0.4))
])
def test_split_tiles_wrong_split_values(tiles, splits):
    n_tiles = tiles.shape[3]
    with pytest.raises(ValueError):
        _, _, _ = src.detector.data_prep._split_tiles(n_tiles, splits)


@pytest.mark.parametrize("features,labels,marker,splits", [
    (lazy_fixture("features_6x12_masked_6x4_cleaned_tiled_3x3"),
     lazy_fixture("labels_6x12_masked_6x4_cleaned_tiled_3x3"),
     lazy_fixture("slum_tile_marker"),
     (0.35, 0.35, 0.3))
])
def test_stratified_split(features, labels, marker, splits):
    features_train, features_val, features_test, labels_train, labels_val, labels_test = \
        src.detector.data_prep._split_stratified(features, labels, marker, splits)
    assert features_train.mask.shape == features_train.data.shape
    assert features_val.mask.shape == features_val.data.shape
    assert features_test.mask.shape == features_test.data.shape
    assert labels_train.shape[0] == labels_val.shape[0] == labels_test.shape[0] == features.shape[0]
    assert labels_train.shape[3] + labels_val.shape[3] + labels_test.shape[3] == features.shape[3]


########################################################################################################################
# Integration                                                                                                          #
########################################################################################################################
# TODO: Add test that prepare() throws error if split= without labels=.
def test_prepare_nomask_nolabel_nosplit_nopath(integration_features_png, integration_features_array):
    tiled_features, _, _, _, _, _, _, _ = src.detector.data_prep.prepare(
        integration_features_png,
        (3, 3)
    )
    assert np.array_equal(tiled_features.data, integration_features_array.data)
    assert np.array_equal(tiled_features.mask, integration_features_array.mask)


def test_integration_masked_nolabel_nosplit_nopath(integration_features_png,
                                                   integration_mask_png,
                                                   integration_mask_array):
    tiled_features, _, _, _, _, _, _, _ = src.detector.data_prep.prepare(
        integration_features_png,
        (3, 3),
        mask_png=integration_mask_png
    )
    assert np.array_equal(tiled_features.data, integration_mask_array.data)
    assert np.array_equal(tiled_features.mask, integration_mask_array.mask)


def test_integration_masked_labelled_nosplit_nopath(integration_features_png,
                                                    integration_mask_png,
                                                    integration_labels_png,
                                                    integration_mask_array,
                                                    integration_labels_array
                                                    ):
    tiled_features, tiled_labels, _, _, _, _, _, _ = src.detector.data_prep.prepare(
        integration_features_png,
        (3, 3),
        mask_png=integration_mask_png,
        label_png=integration_labels_png
    )
    assert np.array_equal(tiled_features.data, integration_mask_array.data)
    assert np.array_equal(tiled_features.mask, integration_mask_array.mask)
    assert np.array_equal(tiled_labels.data, integration_labels_array.data)
    assert np.array_equal(tiled_labels.mask, integration_labels_array.mask)


def test_integration_masked_labelled_split_nopath(
        integration_features_png, integration_mask_png, integration_labels_png):
    _, _, features_train, features_val, features_test, labels_train, labels_val, labels_test = \
        src.detector.data_prep.prepare(
            integration_features_png,
            (3, 3),
            mask_png=integration_mask_png,
            label_png=integration_labels_png,
            splits=(0.5, 0.33, 0.17))
    assert type(features_train) == type(labels_train) == ma.masked_array
    assert np.array_equal(np.sort(
        np.hstack((
        np.unique(features_train.data),
        np.unique(features_val.data),
        np.unique(features_test.data))
        )),
        [0, 1, 2, 3, 6, 7, 8])


def test_integration_nomask_labelled_split_nopath(
        integration_features_png, integration_labels_png):
    _, _, features_train, features_val, features_test, labels_train, labels_val, labels_test = \
        src.detector.data_prep.prepare(
            integration_features_png,
            (3, 3),
            label_png=integration_labels_png,
            splits=(0.33, 0.33, 0.34))
    assert type(features_train) == type(labels_train) == ma.masked_array
    assert np.array_equal(np.sort(
        np.hstack((
        np.unique(features_train.data),
        np.unique(features_val.data),
        np.unique(features_test.data))
        )),
        [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]) or np.array_equal(np.sort(
        np.hstack((
        np.unique(features_train.data),
        np.unique(features_val.data),
        np.unique(features_test.data))
        )),
        [0, 1, 2, 3, 4, 5, 6, 7, 8])


# TODO: Implement test on labels, e.g. number of (unmasked & uncleaned) slum tiles via a sum operation
def test_integration_masked_labelled_split_path(integration_features_png,
                                                integration_mask_png,
                                                integration_labels_png,
                                                tmpdir_factory):
    path_npz = str(tmpdir_factory.mktemp("npz"))
    _, _, _, _, _, _, _, _ = src.detector.data_prep.prepare(
        integration_features_png,
        (3, 3),
        mask_png=integration_mask_png,
        label_png=integration_labels_png,
        splits=(0.5, 0.33, 0.17),
        path_npz=path_npz
    )
    loaded_tiles = np.load(path_npz + ".npz")
    features_train_data = loaded_tiles['features_train_data']
    features_train_mask = loaded_tiles['features_train_mask']
    features_train = ma.masked_array(features_train_data, mask=features_train_mask)
    features_val_data = loaded_tiles['features_val_data']
    features_val_mask = loaded_tiles['features_val_mask']
    features_val = ma.masked_array(features_val_data, mask=features_val_mask)
    features_test_data = loaded_tiles['features_test_data']
    features_test_mask = loaded_tiles['features_test_mask']
    features_test = ma.masked_array(features_test_data, mask=features_test_mask)
    labels_train_data = loaded_tiles['labels_train_data']
    labels_train_mask = loaded_tiles['labels_train_mask']
    labels_train = ma.masked_array(labels_train_data, mask=labels_train_mask)
    labels_val_data = loaded_tiles['labels_val_data']
    labels_val_mask = loaded_tiles['labels_val_mask']
    labels_val = ma.masked_array(labels_val_data, mask=labels_val_mask)
    labels_test_data = loaded_tiles['labels_test_data']
    labels_test_mask = loaded_tiles['labels_test_mask']
    labels_test = ma.masked_array(labels_test_data, mask=labels_test_mask)
    assert np.array_equal(np.sort(
        np.hstack((
        np.unique(features_train.data),
        np.unique(features_val.data),
        np.unique(features_test.data))
        )),
        [0, 1, 2, 3, 6, 7, 8])
    assert type(features_train) == type(labels_train) == ma.masked_array


# Test failing after change to tmpdir_factory... though diferent dir doesn't solve the problem. Same issue as next tst.
@pytest.mark.xfail(reason="Possibly a path specification issue (test used to pass)")
def test_save_png_unlabelled_nomask(integration_features_png, tmpdir_factory):
    path_png = str(tmpdir_factory.mktemp("png"))
    _, _, _, _, _, _, _, _ = src.detector.data_prep.prepare(
        integration_features_png,
        (3, 3),
        path_png=path_png
    )
    assert os.path.exists(path_png + "images/image_7.png")
    assert not os.path.exists(path_png + "images/image_8.png")
    assert np.array_equal(imageio.imread(path_png + "images/image_0.png"),
                          np.dstack([np.ones((3, 3))] * 3))
    assert np.array_equal(imageio.imread(path_png + "images/image_3.png"),
                          np.dstack([[[4, 4, 0], [4, 4, 0], [4, 4, 0]]] * 3))
    assert os.path.exists(path_png + "masks/mask_7.png")
    assert not os.path.exists(path_png + "masks/mask_8.png")
    assert np.array_equal(imageio.imread(path_png + "masks/mask_0.png"),
                          np.zeros((3, 3)))
    assert np.array_equal(imageio.imread(path_png + "masks/mask_3.png"),
                          [[0, 0, 1], [0, 0, 1], [0, 0, 1]])


# Test failing after change to tmpdir_factory... though diferent dir doesn't solve the problem. Same issue as next tst.
@pytest.mark.xfail(reason="Possibly a path specification issue (test used to pass)")
def test_save_png_unlabelled_masked(integration_features_png, integration_mask_png, tmpdir_factory):
    path_png = str(tmpdir_factory.mktemp("png"))
    _, _, _, _, _, _, _, _ = src.detector.data_prep.prepare(
        integration_features_png,
        (3, 3),
        mask_png=integration_mask_png,
        path_png=path_png
    )
    assert os.path.exists(path_png + "images/image_5.png")
    assert not os.path.exists(path_png + "images/image_6.png")
    assert np.array_equal(imageio.imread(path_png + "images/image_0.png"),
                          np.dstack([np.ones((3, 3))] * 3))
    assert np.array_equal(imageio.imread(path_png + "images/image_3.png"),
                          np.dstack([np.ones((3, 3)) * 6] * 3))
    assert os.path.exists(path_png + "masks/mask_5.png")
    assert not os.path.exists(path_png + "masks/mask_6.png")
    assert np.array_equal(imageio.imread(path_png + "masks/mask_0.png"),
                          [[0, 0, 0],
                           [0, 0, 0],
                           [1, 1, 1]])
    assert np.array_equal(imageio.imread(path_png + "masks/mask_3.png"),
                          np.zeros((3, 3)))


def test_save_png_labelled_masked_nosplit(integration_features_png,
                                          integration_mask_png,
                                          integration_labels_png,
                                          tmpdir_factory):
    path_png = str(tmpdir_factory.mktemp("png"))
    _, _, _, _, _, _, _, _ = src.detector.data_prep.prepare(
        integration_features_png,
        (3, 3),
        mask_png=integration_mask_png,
        label_png=integration_labels_png,
        path_png=path_png
    )
    assert os.path.exists(path_png + "images/image_5.png")
    assert not os.path.exists(path_png + "images/image_6.png")
    assert np.array_equal(imageio.imread(path_png + "images/image_0.png"),
                          np.dstack([np.ones((3, 3))] * 3))
    assert np.array_equal(imageio.imread(path_png + "images/image_3.png"),
                          np.dstack([np.ones((3, 3)) * 6] * 3))
    assert os.path.exists(path_png + "masks/mask_5.png")
    assert not os.path.exists(path_png + "masks/mask_6.png")
    assert np.array_equal(imageio.imread(path_png + "masks/mask_0.png"),
                          [[0, 0, 0],
                           [0, 0, 0],
                           [1, 1, 1]])
    assert np.array_equal(imageio.imread(path_png + "masks/mask_3.png"),
                          np.zeros((3, 3)))
    assert os.path.exists(path_png + "labels/label_5.png")
    assert not os.path.exists(path_png + "labels/label_6.png")
    assert np.array_equal(imageio.imread(path_png + "labels/label_0.png"),
                          np.zeros((3, 3)))
    assert np.array_equal(imageio.imread(path_png + "labels/label_2.png"),
                          [[0, 0, 0],
                           [0, 0, 0],
                           [0, 1, 1]])


# TODO: Add more meaningful tests, e.g. setting seed and verifying contents of a feature, mask and label png
def test_save_png_labelled_masked_split(integration_features_png,
                                        integration_mask_png,
                                        integration_labels_png,
                                        tmpdir_factory):
    path_png = str(tmpdir_factory.mktemp("png"))
    _, _, _, _, _, _, _, _ = src.detector.data_prep.prepare(
        integration_features_png,
        (3, 3),
        mask_png=integration_mask_png,
        label_png=integration_labels_png,
        splits=(0.5, 0.33, 0.17),
        path_png=path_png
    )
    assert os.path.exists(path_png + "training/images/image_3.png")
    assert not os.path.exists(path_png + "training/images/image_4.png")
    assert os.path.exists(path_png + "training/masks/mask_3.png")
    assert not os.path.exists(path_png + "training/masks/mask_4.png")
    assert os.path.exists(path_png + "training/labels/label_3.png")
    assert not os.path.exists(path_png + "training/labels/label_4.png")