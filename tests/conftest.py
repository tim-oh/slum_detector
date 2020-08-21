import pytest
import numpy as np
import numpy.ma as ma
import imageio


tst_dim = (2, 80) # Should be even numbers

# TODO: Refactor boilerplate fixtures, perhaps with a fixture generator

@pytest.fixture(scope="session")
def all_127_no_mask():
    all_grey = (np.ones(tst_dim) * 127).astype("uint8") # imageio.imwrite scales max (non-uint8) pixel values to 255
    imageio.imwrite("./tests/all_grey.png", all_grey)
    all_grey = ma.masked_array(np.ones(tst_dim), mask=np.zeros(tst_dim), dtype=int)
    return all_grey


@pytest.fixture(scope="session")
def all_127_bottom_mask():
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("./tests/all_grey.png", all_grey)
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask = np.concatenate((np.zeros((dim_one, dim_two)), np.ones((dim_one, dim_two))), axis=0)
    all_grey = ma.masked_array(np.ones(tst_dim), mask=mask, dtype=int)
    return all_grey




@pytest.fixture(scope="session")
def all_127_full_mask():
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("tests/all_grey.png", all_grey)
    all_grey = ma.masked_array(np.ones(tst_dim), mask=np.ones(tst_dim), dtype=int)
    return all_grey


@pytest.fixture(scope="session")
def all_0_no_mask():
    all_black = np.zeros(tst_dim).astype("uint8")
    imageio.imwrite("./tests/all_black.png", all_black)
    all_black = ma.masked_array(all_black, mask=np.zeros(tst_dim), dtype=int)
    return all_black


@pytest.fixture(scope="session")
def mask_all():
    mask_png = np.zeros(tst_dim).astype("uint8")
    imageio.imwrite("./tests/mask_all.png", mask_png) # Write mask to disk that follows slums-world conventions
    mask_array = mask_png + 1 # Convert to masked_array convention of masked == 1
    return mask_array


@pytest.fixture(scope="session")
def mask_bottom():
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask_png = np.concatenate((np.ones((dim_one, dim_two))*127, np.zeros((dim_one, dim_two))), axis=0).astype("uint8")
    imageio.imwrite("./tests/grey_top_black_bottom.png", mask_png) # slums-world format
    mask_array = np.concatenate((np.zeros((dim_one, dim_two)), np.ones((dim_one, dim_two))), axis=0) # np.ma format
    return mask_array


@pytest.fixture
def mask_small():
    mask_png = np.array([[127, 127, 127], [0, 0, 0]]).astype("uint8")
    imageio.imwrite("./tests/mask_small.png", mask_png) # Write mask to disk that follows slums-world conventions
    mask_array = np.array([[0, 0, 0], [1, 1, 1]]) # Convert to masked_array convention of masked == 1, unmasked == 0
    return mask_array


@pytest.fixture(scope="session")
def mask_right():
    dim_one = tst_dim[0]
    dim_two = tst_dim[1] // 2
    mask_png = np.concatenate((np.ones((dim_one, dim_two))*127, np.zeros((dim_one, dim_two))), axis=1).astype("uint8")
    imageio.imwrite("./tests/grey_left_black_right.png", mask_png) # slums-world format
    mask_array = np.concatenate((np.zeros((dim_one, dim_two)), np.ones((dim_one, dim_two))), axis=1) # np.ma format
    return mask_array


@pytest.fixture(scope="session")
def mixed_values():
    mixed_values = np.arange(0, 128).reshape(64, 2).astype("uint8")
    imageio.imwrite("tests/mixed_values.png", mixed_values)
    mixed_values = ma.masked_array(mixed_values, mask=np.zeros(mixed_values.shape), dtype=int)
    return mixed_values



@pytest.fixture(scope="session")
def mask_top():
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask_array = np.concatenate((np.ones((dim_one, dim_two)), np.zeros((dim_one, dim_two))), axis=0) # np.ma format
    return mask_array


@pytest.fixture(scope="session")
def mask_none():
    mask_png = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("tests/mask_none.png", mask_png) # Write mask to disk that follows slums-world conventions
    mask_array = mask_png - 127 # Convert to masked_array convention of unmasked == 0
    return mask_array