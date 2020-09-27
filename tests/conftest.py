import pytest
import numpy as np
import numpy.ma as ma
import imageio


########################################################################################################################
# Evaluation                                                                                                           #
########################################################################################################################
@pytest.fixture
def mixed_pred():
    preds = np.array([
        [1, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 0]])
    mixed_pred = ma.masked_array(preds, np.zeros((3, 6)))
    return mixed_pred


@pytest.fixture
def mixed_truth():
    truth = np.array([
        [1, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 1]])
    mixed_truth = ma.masked_array(truth, np.zeros((3, 6)))
    return mixed_truth


@pytest.fixture
def conf_map_actual():
    truth = np.array([
        ["tp", "tn", "tn", "fp", "fn", "fn"],
        ["tp", "tn", "fp", "fp", "fn", "fn"],
        ["tp", "tn", "fp", "fp", "fn", "fn"]])
    mixed_pred = ma.masked_array(truth)
    return mixed_pred


@pytest.fixture
def confusion_matrix_actual():
    confusion_matrix = {'fn': 6, 'fp': 5, 'tn': 4, 'tp': 3}
    return confusion_matrix


########################################################################################################################
# Data preparation                                                                                                     #
########################################################################################################################
# TODO: Eliminate creation of identical png files

@pytest.fixture(scope='session')
def all_127_no_mask_png(tmpdir_factory):
    tst_dim = (4, 8)
    all_grey = (np.ones(tst_dim) * 127).astype('uint8')
    path = str(tmpdir_factory.mktemp("png").join('all_grey.png'))
    imageio.imwrite(path, all_grey)
    return path


@pytest.fixture(scope='session')
def all_127_no_mask_array():
    tst_dim = (4, 8)
    all_grey = ma.masked_array(np.ones(tst_dim), mask=np.zeros(tst_dim), dtype=int)
    return all_grey


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


@pytest.fixture(scope="session")
def all_127_full_mask_png(tmpdir_factory):
    tst_dim = (4, 8)
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("all_grey.png"))
    imageio.imwrite(path, all_grey)
    return path


@pytest.fixture(scope="session")
def all_127_full_mask_array():
    tst_dim = (4, 8)
    all_grey = ma.masked_array(np.ones(tst_dim), mask=np.ones(tst_dim), dtype=int)
    return all_grey



@pytest.fixture(scope="session")
def all_0_no_mask_png(tmpdir_factory):
    tst_dim = (4, 8)
    all_black = np.zeros(tst_dim).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("all_black.png"))
    imageio.imwrite(path, all_black)
    return path


@pytest.fixture(scope="session")
def all_0_no_mask_array():
    tst_dim = (4, 8)
    all_black = np.zeros(tst_dim).astype("uint8")
    all_black = ma.masked_array(all_black, mask=np.zeros(tst_dim), dtype=int)
    return all_black


@pytest.fixture(scope="session")
def all_0_mask_all_png(tmpdir_factory):
    tst_dim = (4, 8)
    all_black = np.zeros(tst_dim).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("all_black.png"))
    imageio.imwrite(path, all_black)
    return path


@pytest.fixture(scope="session")
def all_0_mask_all_array():
    tst_dim = (4, 8)
    all_black = np.zeros(tst_dim).astype("uint8")
    all_black = ma.masked_array(all_black, mask=np.ones(tst_dim), dtype=int)
    return all_black


@pytest.fixture(scope="session")
def mask_all_png(tmpdir_factory):
    tst_dim = (4, 8)
    mask_png = np.zeros(tst_dim).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("mask_all.png"))
    imageio.imwrite(path, mask_png)
    return path


@pytest.fixture(scope="session")
def mask_all_array():
    tst_dim = (4, 8)
    mask_png = np.zeros(tst_dim).astype("uint8")
    mask_array = mask_png + 1
    return mask_array


@pytest.fixture(scope="session")
def mask_bottom_png(tmpdir_factory):
    tst_dim = (4, 8)
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask_png = np.concatenate((np.ones((dim_one, dim_two))*127, np.zeros((dim_one, dim_two))), axis=0).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("grey_top_black_bottom.png"))
    imageio.imwrite(path, mask_png)
    return path


@pytest.fixture(scope="session")
def mask_bottom_array():
    tst_dim = (4, 8)
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask_array = np.concatenate((np.zeros((dim_one, dim_two)), np.ones((dim_one, dim_two))), axis=0)
    return mask_array


@pytest.fixture(scope="session")
def mask_small_png(tmpdir_factory):
    mask_png = np.array([[127, 127, 127], [0, 0, 0]]).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("mask_small.png"))
    imageio.imwrite(path, mask_png)
    return path

@pytest.fixture(scope="session")
def mask_small_array():
    mask_array = np.array([[0, 0, 0], [1, 1, 1]])
    return mask_array


@pytest.fixture(scope="session")
def mask_right_png(tmpdir_factory):
    tst_dim = (4, 8)
    dim_one = tst_dim[0]
    dim_two = tst_dim[1] // 2
    mask_png = np.concatenate((np.ones((dim_one, dim_two))*127, np.zeros((dim_one, dim_two))), axis=1).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("grey_left_black_right.png"))
    imageio.imwrite(path, mask_png)
    return path


@pytest.fixture(scope="session")
def mask_right_array():
    tst_dim = (4, 8)
    dim_one = tst_dim[0]
    dim_two = tst_dim[1] // 2
    mask_array = np.concatenate((np.zeros((dim_one, dim_two)), np.ones((dim_one, dim_two))), axis=1)
    return mask_array


@pytest.fixture(scope="session")
def mixed_values_png(tmpdir_factory):
    mixed_values = np.arange(0, 128).reshape(64, 2).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("mixed_values.png"))
    imageio.imwrite(path, mixed_values)
    return path


@pytest.fixture(scope="session")
def mixed_values_array():
    mixed_values = np.arange(0, 128).reshape(64, 2).astype("uint8")
    mixed_values = ma.masked_array(mixed_values, mask=np.zeros(mixed_values.shape), dtype=int)
    return mixed_values


@pytest.fixture(scope="session")
def mask_top_array():
    tst_dim = (4, 8)
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask_array = np.concatenate((np.ones((dim_one, dim_two)), np.zeros((dim_one, dim_two))), axis=0) # np.ma format
    return mask_array


@pytest.fixture(scope="session")
def mask_none_png(tmpdir_factory):
    tst_dim = (4, 8)
    mask_png = (np.ones(tst_dim) * 127).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("mask_none.png"))
    imageio.imwrite(path, mask_png)
    return path


@pytest.fixture(scope="session")
def mask_none_array():
    tst_dim = (4, 8)
    mask_png = (np.ones(tst_dim) * 127).astype("uint8")
    mask_array = mask_png - 127 # Convert to masked_array convention of unmasked == 0
    return mask_array


@pytest.fixture(scope="session")
def blocks_no_pad_png(tmpdir_factory):
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    path = str(tmpdir_factory.mktemp("png").join("blocks_no_pad.png"))
    imageio.imwrite(path, blocks_rgb)
    return path


@pytest.fixture(scope="session")
def blocks_no_pad_array():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    return blocks_rgb


@pytest.fixture(scope="session")
def blocks_bottom_pad_png(tmpdir_factory):
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    path = str(tmpdir_factory.mktemp("png").join("blocks_missing_bottom.png"))
    imageio.imwrite(path, blocks_rgb)
    return path


@pytest.fixture(scope="session")
def blocks_bottom_pad_array():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    bottom_row = np.dstack([np.zeros((1, 9))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, bottom_row), axis=0)
    return blocks_rgb


@pytest.fixture(scope="session")
def blocks_right_pad_png(tmpdir_factory):
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    path = str(tmpdir_factory.mktemp("png").join("blocks_missing_right.png"))
    imageio.imwrite(path, blocks_rgb)
    return path


@pytest.fixture(scope="session")
def blocks_right_pad_array():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    right_rows = np.dstack([np.zeros((6, 2))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, right_rows), axis=1)
    return blocks_rgb


@pytest.fixture(scope="session")
def blocks_both_pad_png(tmpdir_factory):
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    path = str(tmpdir_factory.mktemp("png").join("blocks_missing_both.png"))
    imageio.imwrite(path, blocks_rgb)
    return path


@pytest.fixture(scope="session")
def blocks_both_pad_array():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    right_rows = np.dstack([np.zeros((5, 2))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, right_rows), axis=1)
    bottom_row = np.dstack([np.zeros((1, 9))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, bottom_row), axis=0)
    return blocks_rgb


@pytest.fixture
def blocks_negative_value_array(tmpdir_factory):
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


@pytest.fixture(scope="session")
def blocks_mask_leftmost_png(tmpdir_factory):
    mask_png = np.concatenate((np.zeros((6, 1)), np.ones((6, 8)) * 127), axis=1).astype('uint8')
    path = str(tmpdir_factory.mktemp("png").join("blocks_mask_leftmost.png"))
    imageio.imwrite(path, mask_png)
    return path


@pytest.fixture(scope="session")
def blocks_mask_leftmost_array():
    mask_one_layer = np.concatenate((np.ones((6, 1)), np.zeros((6, 8))), axis=1)
    mask_array = np.dstack([mask_one_layer] * 3)
    return mask_array


@pytest.fixture
def blocks_excessive_value_array():
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


@pytest.fixture(scope="session")
def blocks_small_mask_png(tmpdir_factory):
    mask_png = np.concatenate((np.ones((3, 8)) * 127, np.zeros((3, 8))), axis=0)
    path = str(tmpdir_factory.mktemp("png").join("blocks_small_mask.png"))
    imageio.imwrite(path, mask_png)
    return path


@pytest.fixture
def blocks_large_width_tstpad_array():
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


@pytest.fixture(scope="session")
def blocks_no_pad_tstpad_png(tmpdir_factory):
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    path = str(tmpdir_factory.mktemp("png").join("blocks_no_pad_tstpad.png"))
    imageio.imwrite(path, blocks_rgb)
    return path


@pytest.fixture(scope="session")
def blocks_no_pad_tstpad_array():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    padded_mask = np.zeros((6, 9))
    mask_rgb = np.dstack([padded_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks


@pytest.fixture(scope="session")
def blocks_bottom_pad_tstpad_png(tmpdir_factory):
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    path = str(tmpdir_factory.mktemp("png").join("blocks_missing_bottom_tstpad.png"))
    imageio.imwrite(path, blocks_rgb)
    return path


@pytest.fixture(scope="session")
def blocks_bottom_pad_tstpad_array():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    bottom_row = np.dstack([np.zeros((1, 9))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, bottom_row), axis=0)
    padded_mask = np.concatenate((np.zeros((5, 9)), np.ones((1, 9))), axis=0)
    mask_rgb = np.dstack([padded_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks


@pytest.fixture(scope="session")
def blocks_right_pad_tstpad_png(tmpdir_factory):
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    path = str(tmpdir_factory.mktemp("png").join("blocks_missing_right_tstpad.png"))
    imageio.imwrite(path, blocks_rgb)
    return path


@pytest.fixture(scope="session")
def blocks_right_pad_tstpad_array():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    right_rows = np.dstack([np.zeros((6, 2))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, right_rows), axis=1)
    padded_mask = np.concatenate((np.zeros((6, 7)), np.ones((6, 2))), axis=1)
    mask_rgb = np.dstack([padded_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks


@pytest.fixture(scope="session")
def blocks_both_pad_tstpad_png(tmpdir_factory):
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    path = str(tmpdir_factory.mktemp("png").join("blocks_missing_both_tstpad.png"))
    imageio.imwrite(path, blocks_rgb)
    return path


@pytest.fixture(scope="session")
def blocks_both_pad_tstpad_array():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    right_rows = np.dstack([np.zeros((5, 2))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, right_rows), axis=1)
    bottom_row = np.dstack([np.zeros((1, 9))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, bottom_row), axis=0)
    unpadded_mask = np.zeros((5, 7))
    padded_mask = np.pad(unpadded_mask, ((0, 1), (0, 2)), 'constant', constant_values=(1,))
    mask_rgb = np.dstack([padded_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks


# TODO: Possible conflict from this fixture writing a png of same name as fixture all_127_no_mask()
@pytest.fixture(scope="session")
def padded_all_127_mask_none_png(tmpdir_factory):
    tst_dim = (4, 8)
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("all_grey.png"))
    imageio.imwrite(path, all_grey)
    return path


@pytest.fixture(scope="session")
def padded_all_127_mask_none_array():
    tst_dim = (4, 8)
    padded_array = np.pad(np.ones((4, 8)), ((0, 2), (0, 1)), 'constant', constant_values=(0,))
    mask_unpadded = np.zeros(tst_dim)
    mask_array = np.pad(mask_unpadded, ((0, 2), (0, 1)), 'constant', constant_values=(1,))
    masked_array = ma.masked_array(padded_array, mask=mask_array)
    return masked_array


@pytest.fixture(scope="session")
def all_grey_png(tmpdir_factory):
    tst_dim = (4, 8)
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("all_grey.png"))
    imageio.imwrite(path, all_grey)
    return path

@pytest.fixture(scope="session")
def mask_bottom_png(tmpdir_factory):
    mask_png = np.concatenate((np.ones((2, 8)) * 127, np.zeros((2, 8))), axis=0).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("mask_bottom.png"))
    imageio.imwrite(path, mask_png)
    return path

@pytest.fixture(scope="session")
def padded_all_127_mask_bottom_array():
    padded_array = np.pad(np.ones((4, 8)), ((0, 2), (0, 1)), 'constant', constant_values=(0,))
    mask_unpadded = np.concatenate((np.zeros((2, 8)), np.ones((2, 8))), axis=0)
    mask_array = np.pad(mask_unpadded, ((0, 2), (0, 1)), 'constant', constant_values=(1,))
    masked_array = ma.masked_array(padded_array, mask=mask_array)
    return masked_array


@pytest.fixture(scope="session")
def mask_left_png(tmpdir_factory):
    mask_png = np.concatenate((np.zeros((4, 4)), np.ones((4, 4)) * 127), axis=1).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("mask_left.png"))
    imageio.imwrite(path, mask_png)
    return path


@pytest.fixture(scope="session")
def padded_all_127_mask_left_array():
    padded_array = np.pad(np.ones((4, 8)), ((0, 2), (0, 1)), 'constant', constant_values=(0,))
    mask_unpadded = np.concatenate((np.ones((4, 4)), np.zeros((4, 4))), axis=1)
    mask_array = np.pad(mask_unpadded, ((0, 2), (0, 1)), 'constant', constant_values=(1,))
    masked_array = ma.masked_array(padded_array, mask=mask_array)
    return masked_array


@pytest.fixture(scope="session")
def mask_all_png(tmpdir_factory):
    tst_dim = (4, 8)
    mask_png = np.zeros(tst_dim).astype("uint8")
    path = str(tmpdir_factory.mktemp("png").join("mask_all.png"))
    imageio.imwrite(path, mask_png)
    return path

@pytest.fixture(scope="session")
def padded_all_127_mask_all_array():
    tst_dim = (4, 8)
    padded_array = np.pad(np.ones((4, 8)), ((0, 2), (0, 1)), 'constant', constant_values=(0,))
    mask_unpadded = np.ones(tst_dim)
    mask_array = np.pad(mask_unpadded, ((0, 2), (0, 1)), 'constant', constant_values=(1,))
    masked_array = ma.masked_array(padded_array, mask=mask_array)
    return masked_array



@pytest.fixture(scope="session")
def mask_leftmost_png(tmpdir_factory):
    unconverted_mask = np.concatenate((np.zeros((6, 1)), np.ones((6, 8)) * 127), axis=1).astype('uint8')
    path = str(tmpdir_factory.mktemp("png").join("blocks_no_pad_mask_leftmost.png"))
    imageio.imwrite(path, unconverted_mask)
    return path


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def blocks_bottom_pad_mask_top_png(tmpdir_factory):
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    path = str(tmpdir_factory.mktemp("png").join("blocks_missing_bottom_tstpad.png"))
    imageio.imwrite(tmpdir_factory.mktemp("png").join("blocks_missing_bottom_tstpad.png"), blocks_rgb)
    return path

@pytest.fixture(scope="session")
def mask_top_png(tmpdir_factory):
    unconverted_mask = np.concatenate((np.zeros((2, 9)), np.ones((3, 9)) * 127), axis=0).astype('uint8')
    path = str(tmpdir_factory.mktemp("png").join("blocks_bottom_pad_mask_top.png"))
    imageio.imwrite(path, unconverted_mask)
    return path


@pytest.fixture(scope="session")
def blocks_bottom_pad_mask_top_array():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    bottom_row = np.dstack([np.zeros((1, 9))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, bottom_row), axis=0)
    padded_mask = np.concatenate((np.ones((2,9)), np.zeros((3, 9)), np.ones((1, 9))), axis=0)
    mask_rgb = np.dstack([padded_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks


@pytest.fixture(scope="session")
def blocks_missing_right_tstpad_png(tmpdir_factory):
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    path = str(tmpdir_factory.mktemp("png").join("blocks_missing_right_tstpad.png"))
    imageio.imwrite(path, blocks_rgb)
    return path


@pytest.fixture(scope="session")
def mask_bottom_tstpad_png(tmpdir_factory):
    unconverted_mask = np.concatenate((np.ones((3, 7)) * 127, np.zeros((3, 7))), axis=0).astype('uint8')
    path = str(tmpdir_factory.mktemp("png").join("blocks_right_pad_mask_bottom.png"))
    imageio.imwrite(path, unconverted_mask)
    return path


@pytest.fixture(scope="session")
def blocks_right_pad_mask_bottom_array():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    right_rows = np.dstack([np.zeros((6, 2))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, right_rows), axis=1)
    unpadded_mask = np.concatenate((np.zeros((3, 7)), np.ones((3, 7))), axis=0)
    padded_mask = np.concatenate((unpadded_mask, np.ones((6, 2))), axis=1)
    mask_rgb = np.dstack([padded_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks


@pytest.fixture(scope="session")
def blocks_both_pad_mask_right(tmpdir_factory):
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6]
    ]).astype('uint8')
    blocks_rgb = np.dstack((block_digits, block_digits, block_digits))
    imageio.imwrite(tmpdir_factory.mktemp("png").join("blocks_missing_both_tstpad.png"), blocks_rgb)
    unconverted_mask = np.concatenate((np.ones((5, 4)) * 127, np.zeros((5, 3))), axis=1).astype('uint8')
    imageio.imwrite(tmpdir_factory.mktemp("png").join("blocks_both_pad_mask_right.png"), unconverted_mask)
    right_rows = np.dstack([np.zeros((5, 2))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, right_rows), axis=1)
    bottom_row = np.dstack([np.zeros((1, 9))] * 3)
    blocks_rgb = np.concatenate((blocks_rgb, bottom_row), axis=0)
    unpadded_mask = np.concatenate((np.zeros((5, 4)), np.ones((5, 3))), axis=1)
    padded_mask = np.pad(unpadded_mask, ((0, 1), (0, 2)), 'constant', constant_values=(1,))
    mask_rgb = np.dstack([padded_mask] * 3)
    masked_padded_blocks = ma.masked_array(blocks_rgb, mask=mask_rgb)
    return masked_padded_blocks


@pytest.fixture
def features_6x12_masknone():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
    ])
    block_digits_rgb = np.dstack([block_digits] * 3)
    mask = np.zeros(block_digits_rgb.shape)
    masked_blocks = ma.masked_array(block_digits_rgb, mask=mask)
    return masked_blocks


@pytest.fixture
def features_6x12_mask_topleft6x4():
    block_digits = np. array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
    ])
    block_digits_rgb = np.dstack([block_digits] * 3)
    mask= np. array([
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    mask_rgb = np.dstack([mask] * 3)
    masked_blocks = ma.masked_array(block_digits_rgb, mask=mask_rgb)
    return masked_blocks


@pytest.fixture
def coordinates_6x12_tile_3x3():
    coordinates = np.dstack((
        [[0, 0], [2, 2]],
        [[0, 3], [2, 5]],
        [[0, 6], [2, 8]],
        [[0, 9], [2, 11]],
        [[3, 0], [5, 2]],
        [[3, 3], [5, 5]],
        [[3, 6], [5, 8]],
        [[3, 9], [5, 11]]
    ))
    return coordinates


@pytest.fixture
def coordinates_6x12_tile_2x2():
    coordinates = np.dstack((
        [[0, 0], [1, 1]],
        [[0, 2], [1, 3]],
        [[0, 4], [1, 5]],
        [[0, 6], [1, 7]],
        [[0, 8], [1, 9]],
        [[0, 10], [1, 11]],
        [[2, 0], [3, 1]],
        [[2, 2], [3, 3]],
        [[2, 4], [3, 5]],
        [[2, 6], [3, 7]],
        [[2, 8], [3, 9]],
        [[2, 10], [3, 11]],
        [[4, 0], [5, 1]],
        [[4, 2], [5, 3]],
        [[4, 4], [5, 5]],
        [[4, 6], [5, 7]],
        [[4, 8], [5, 9]],
        [[4, 10], [5, 11]]
    ))
    return coordinates


@pytest.fixture
def features_6x12_masknone_tiled_3x3():
    tile_data = np.stack((
        np.dstack([np.ones((3, 3))] * 3),
        np.dstack([np.ones((3, 3)) * 2] * 3),
        np.dstack([np.ones((3, 3)) * 3] * 3),
        np.dstack([np.ones((3, 3)) * 4] * 3),
        np.dstack([np.ones((3, 3)) * 5] * 3),
        np.dstack([np.ones((3, 3)) * 6] * 3),
        np.dstack([np.ones((3, 3)) * 7] * 3),
        np.dstack([np.ones((3, 3)) * 8] * 3)), axis=3)
    tile_mask = np.zeros(tile_data.shape)
    masked_tiles = ma.masked_array(tile_data, mask=tile_mask)
    return masked_tiles



@pytest.fixture
def features_6x12_mask_topleft6x4_tiled_3x3():
    tile_data = np.stack((
        np.dstack([np.ones((3, 3))] * 3),
        np.dstack([np.ones((3, 3)) * 2] * 3),
        np.dstack([np.ones((3, 3)) * 3] * 3),
        np.dstack([np.ones((3, 3)) * 4] * 3),
        np.dstack([np.ones((3, 3)) * 5] * 3),
        np.dstack([np.ones((3, 3)) * 6] * 3),
        np.dstack([np.ones((3, 3)) * 7] * 3),
        np.dstack([np.ones((3, 3)) * 8] * 3)), axis=3)
    lowlefttiles = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    tile_mask = np.stack((
        np.dstack([np.ones((3, 3))] * 3),
        np.dstack([np.ones((3, 3))] * 3),
        np.dstack([np.zeros((3, 3))] * 3),
        np.dstack([np.zeros((3, 3))] * 3),
        np.dstack([lowlefttiles] * 3),
        np.dstack([lowlefttiles] * 3),
        np.dstack([np.zeros((3, 3))] * 3),
        np.dstack([np.zeros((3, 3))] * 3)), axis=3)
    masked_tiles = ma.masked_array(tile_data, mask=tile_mask)
    return masked_tiles


@pytest.fixture
def features_6x12_masked_6x4_cleaned_tiled_3x3():
    tile_data = np.stack((
        np.dstack([np.ones((3, 3)) * 3] * 3),
        np.dstack([np.ones((3, 3)) * 4] * 3),
        np.dstack([np.ones((3, 3)) * 5] * 3),
        np.dstack([np.ones((3, 3)) * 6] * 3),
        np.dstack([np.ones((3, 3)) * 7] * 3),
        np.dstack([np.ones((3, 3)) * 8] * 3)), axis=3)
    lowlefttiles = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    tile_mask = np.stack((
        np.dstack([np.zeros((3, 3))] * 3),
        np.dstack([np.zeros((3, 3))] * 3),
        np.dstack([lowlefttiles] * 3),
        np.dstack([lowlefttiles] * 3),
        np.dstack([np.zeros((3, 3))] * 3),
        np.dstack([np.zeros((3, 3))] * 3)), axis=3)
    masked_tiles = ma.masked_array(tile_data, mask=tile_mask)
    return masked_tiles


@pytest.fixture
def register_6x12_mask_topleft6x4_tiled_3x3():
    register = np.array(
        ['masked', 'masked', 'predict', 'predict', 'predict', 'predict', 'predict', 'predict'], dtype='object')
    return register


@pytest.fixture
def register_slum():
    register = np.array(
        ['masked', 'non-slum', 'slum', 'slum', 'slum', 'slum', 'slum', 'slum'], dtype='object')
    return register


# 2x2 tiling. 12 of the 18 remain after mask cleaning. 6 have labels.
# Masked labels should be ignored
@pytest.fixture
def labels_6x12_mask_topleft6x4():
    block_digits = np. array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    mask= np. array([
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    masked_blocks = ma.masked_array(block_digits, mask=mask)
    return masked_blocks

@pytest.fixture
def labels_6x12_masked_6x4_tiled_3x3():
    tiles_3d = np.stack((
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 0]],
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
        [[1, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]), axis=2)
    tiles_4d = tiles_3d[:, :, np.newaxis, :]
    lowlefttiles = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    tile_mask = np.stack((
        np.ones((3, 3)),
        np.ones((3, 3)),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        lowlefttiles,
        lowlefttiles,
        np.zeros((3, 3)),
        np.zeros((3, 3))), axis=2)
    tile_mask = tile_mask[:, :, np.newaxis, :]
    masked_tiles = ma.masked_array(tiles_4d, mask=tile_mask)
    return masked_tiles


@pytest.fixture
def labels_6x12_masked_6x4_cleaned_tiled_3x3():
    tiles_3d = np.stack((
        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 0]],
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
        [[1, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]), axis=2)
    tiles_4d = tiles_3d[:, :, np.newaxis, :]
    lowlefttiles = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    tile_mask = np.stack((
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        lowlefttiles,
        lowlefttiles,
        np.zeros((3, 3)),
        np.zeros((3, 3))), axis=2)
    tile_mask = tile_mask[:, :, np.newaxis, :]
    masked_tiles = ma.masked_array(tiles_4d, mask=tile_mask)
    return masked_tiles


@pytest.fixture
def slum_tile_marker():
    boolean_array = np.array([True, True, True, False, False, False])
    return boolean_array


@pytest.fixture(scope="session")
def integration_features_png(tmpdir_factory):
    block_digits = np.array([
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8],
        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8]
    ]).astype('uint8')
    blocks_rgb = np.dstack([block_digits] * 3)
    path = str(tmpdir_factory.mktemp("png").join("integration_features.png"))
    imageio.imwrite(path, blocks_rgb)
    return path


@pytest.fixture(scope="session")
def integration_features_array():
    tile_four = [[4, 4, 0], [4, 4, 0], [4, 4, 0]]
    tile_eight = [[8, 8, 0], [8, 8, 0], [8, 8, 0]]
    padded_digits_rgb = np.stack((
        np.dstack([np.ones((3, 3))] * 3),
        np.dstack([np.ones((3, 3)) * 2] * 3),
        np.dstack([np.ones((3, 3)) * 3] * 3),
        np.dstack([tile_four] * 3),
        np.dstack([np.ones((3, 3)) * 5] * 3),
        np.dstack([np.ones((3, 3)) * 6] * 3),
        np.dstack([np.ones((3, 3)) * 7] * 3),
        np.dstack([tile_eight] * 3)), axis=3)
    mask = np.zeros(padded_digits_rgb.shape)
    mask[:, 2, :, 3] = 1
    mask[:, 2, :, 7] = 1
    masked_blocks = ma.masked_array(padded_digits_rgb, mask=mask)
    return masked_blocks


@pytest.fixture(scope="session")
def integration_mask_png(tmpdir_factory):
    mask_png = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    ]).astype('uint8') * 127
    path = str(tmpdir_factory.mktemp("png").join("integration_mask.png"))
    imageio.imwrite(path, mask_png)
    return path


@pytest.fixture(scope="session")
def integration_mask_array():
    tile_eight = [[8, 8, 0], [8, 8, 0], [8, 8, 0]]
    padded_digits_rgb = np.stack((
        np.dstack([np.ones((3, 3))] * 3),
        np.dstack([np.ones((3, 3)) * 2] * 3),
        np.dstack([np.ones((3, 3)) * 3] * 3),
        np.dstack([np.ones((3, 3)) * 6] * 3),
        np.dstack([np.ones((3, 3)) * 7] * 3),
        np.dstack([tile_eight] * 3)), axis=3)
    mask_array = np.stack((
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 1]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[1, 1, 1],
         [0, 0, 1],
         [0, 0, 1]]), axis=2)
    mask_4d = mask_array[:, :, np.newaxis, :]
    mask_rgb = np.repeat(mask_4d, 3, axis=2)
    masked_blocks = ma.masked_array(padded_digits_rgb, mask=mask_rgb)
    return masked_blocks


@pytest.fixture(scope="session")
def integration_labels_png(tmpdir_factory):
    labels_png = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    ]).astype('uint8') * 127
    path = str(tmpdir_factory.mktemp("png").join("integration_labels.png"))
    imageio.imwrite(path, labels_png)
    return path


@pytest.fixture(scope="session")
def integration_labels_array():
    padded_labels = np.stack((
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 1]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 1, 1],
         [0, 1, 1],
         [0, 1, 1]],
        [[1, 1, 0],
         [1, 1, 0],
         [1, 1, 0]]), axis=2)
    labels_4d = padded_labels[:, :, np.newaxis, :]
    mask_array = np.stack((
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 1]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[1, 1, 1],
         [0, 0, 1],
         [0, 0, 1]]), axis=2)
    mask_4d = mask_array[:, :, np.newaxis, :]
    masked_labels = ma.masked_array(labels_4d, mask=mask_4d)
    return masked_labels
