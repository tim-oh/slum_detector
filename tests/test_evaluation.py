import pytest
import numpy as np
import numpy.ma as ma
import imageio
import src.detector.evaluation as ev

tst_dim = (2, 4) # Pick two even number to avoid breakage of fixtures


########################################################################################################################
# Loading                                                                                                              #
########################################################################################################################
# TODO: Refactor boilerplate fixtures, perhaps with a fixture generator
@pytest.fixture
def all_127_no_mask():
    all_grey = (np.ones(tst_dim) * 127).astype("uint8") # imageio.imwrite scales max (non-uint8) pixel values to 255
    imageio.imwrite("./tests/all_grey.png", all_grey)
    all_grey = ma.masked_array(np.ones(tst_dim), mask=np.zeros(tst_dim), dtype=int)
    return all_grey


@pytest.fixture
def all_127_full_mask():
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("./tests/all_grey.png", all_grey)
    all_grey = ma.masked_array(np.ones(tst_dim), mask=np.ones(tst_dim), dtype=int)
    return all_grey


@pytest.fixture
def all_127_bottom_mask():
    all_grey = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("./tests/all_grey.png", all_grey)
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask = np.concatenate((np.zeros((dim_one, dim_two)), np.ones((dim_one, dim_two))), axis=0)
    all_grey = ma.masked_array(np.ones(tst_dim), mask=mask, dtype=int)
    return all_grey


@pytest.fixture
def all_0_no_mask():
    all_black = np.zeros(tst_dim).astype("uint8")
    imageio.imwrite("./tests/all_black.png", all_black)
    all_black = ma.masked_array(all_black, mask=np.zeros(tst_dim), dtype=int)
    return all_black


@pytest.fixture
def mixed_values():
    mixed_values = np.arange(0, 128).reshape(64, 2).astype("uint8")
    imageio.imwrite("./tests/mixed_values.png", mixed_values)
    mixed_values = ma.masked_array(mixed_values, mask=np.zeros(mixed_values.shape), dtype=int)
    return mixed_values


@pytest.fixture
def mask_all():
    mask_png = np.zeros(tst_dim).astype("uint8")
    imageio.imwrite("./tests/mask_all.png", mask_png) # Write mask to disk that follows slums-world conventions
    mask_array = mask_png + 1 # Convert to masked_array convention of masked == 1
    return mask_array


@pytest.fixture
def mask_bottom():
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask_png = np.concatenate((np.ones((dim_one, dim_two))*127, np.zeros((dim_one, dim_two))), axis=0).astype("uint8")
    imageio.imwrite("./tests/grey_top_black_bottom.png", mask_png) # slums-world format
    mask_array = np.concatenate((np.zeros((dim_one, dim_two)), np.ones((dim_one, dim_two))), axis=0) # np.ma format
    return mask_array


@pytest.fixture
def mask_top():
    dim_one = tst_dim[0] // 2
    dim_two = tst_dim[1]
    mask_array = np.concatenate((np.ones((dim_one, dim_two)), np.zeros((dim_one, dim_two))), axis=0) # np.ma format
    return mask_array


@pytest.fixture
def mask_small():
    mask_png = np.array([[127, 127, 127], [0, 0, 0]]).astype("uint8")
    imageio.imwrite("./tests/mask_small.png", mask_png) # Write mask to disk that follows slums-world conventions
    mask_array = np.array([[0, 0, 0], [1, 1, 1]]) # Convert to masked_array convention of masked == 1, unmasked == 0
    return mask_array


@pytest.fixture
def mask_right():
    dim_one = tst_dim[0]
    dim_two = tst_dim[1] // 2
    mask_png = np.concatenate((np.ones((dim_one, dim_two))*127, np.zeros((dim_one, dim_two))), axis=1).astype("uint8")
    imageio.imwrite("./tests/grey_left_black_right.png", mask_png) # slums-world format
    mask_array = np.concatenate((np.zeros((dim_one, dim_two)), np.ones((dim_one, dim_two))), axis=1) # np.ma format
    return mask_array


@pytest.fixture
def mask_none():
    mask_png = (np.ones(tst_dim) * 127).astype("uint8")
    imageio.imwrite("./tests/mask_none.png", mask_png) # Write mask to disk that follows slums-world conventions
    mask_array = mask_png - 127 # Convert to masked_array convention of unmasked == 0
    return mask_array


# Note: imwrite scales img range (e.g [0, 1]) to [0, 255] unless dtype=uint8, whereas imread always converts 1:1
def test_load_grey_png(all_127_no_mask):
    all_grey_png = ev.png_to_labels("tests/all_grey.png")
    assert np.array_equal(all_grey_png, all_127_no_mask)
    assert type(all_grey_png) == ma.masked_array


def test_load_mixed_png(mixed_values):
    mixed_png = ev.png_to_labels("tests/mixed_values.png")
    converted = np.concatenate((np.zeros((32, 2)), np.ones((32, 2))), axis=0)
    assert np.array_equal(mixed_png, converted)


def test_load_black_png(all_0_no_mask):
    all_black_png = ev.png_to_labels("tests/all_black.png")
    assert np.array_equal(all_black_png, all_0_no_mask)
    assert type(all_black_png) == ma.masked_array


def test_load_fail(all_0_no_mask, all_127_no_mask):
    with pytest.raises(AssertionError):
        all_black_png = ev.png_to_labels("tests/all_black.png")
        assert np.array_equal(all_black_png, all_127_no_mask)


# TODO: Array of masked black pixels wrongly evaluates same as unmasked one.
# TODO: Tests fail unless tests/xxx.png exists already, so have to py.test twice. Fix via fixture scopes?
@pytest.mark.parametrize("pred,mask,expected", [
    ("tests/all_black.png", "tests/mask_none.png", pytest.lazy_fixture("all_0_no_mask")),
    ("tests/all_black.png", "tests/mask_all.png", pytest.lazy_fixture("all_0_no_mask")),
    ("tests/all_grey.png", "tests/mask_none.png", pytest.lazy_fixture("all_127_no_mask")),
    ("tests/all_grey.png", "tests/mask_all.png", pytest.lazy_fixture("all_127_full_mask")),
    ("tests/all_grey.png", "tests/grey_top_black_bottom.png", pytest.lazy_fixture("all_127_bottom_mask"))
])
def test_load_masked_pred(pred, mask, expected):
    expected = expected
    loaded_mask = ev.png_to_labels(pred, mask)
    assert np.array_equal(loaded_mask, expected)


def test_wrong_mask_size(all_127_no_mask, mask_small):
    with pytest.raises(ValueError):
        ev.png_to_labels("tests/all_grey.png", "tests/mask_small.png")


def test_inconsistent_pred_truth_sizes(all_127_no_mask, mask_small):
    with pytest.raises(ValueError):
        ev.conf_map(all_127_no_mask, mask_small)

########################################################################################################################
# Conversion                                                                                                           #
########################################################################################################################
def test_wrong_mask_value(mask_all):
    with pytest.raises(ValueError):
        ev.convert_mask(mask_all)

# TODO: Warning not working properly: fix and add test that DOESN'T throw a warning when there are two values
def test_uniform_mask_value_warning(mask_none):
    with pytest.warns(UserWarning):
        ev.convert_mask(mask_none)


@pytest.mark.parametrize("mask,expected", [
    ("tests/mask_all.png", pytest.lazy_fixture("mask_all")),
    ("tests/mask_none.png", pytest.lazy_fixture("mask_none")),
    ("tests/grey_top_black_bottom.png", pytest.lazy_fixture("mask_bottom")),
    ])
def test_convert_masks(mask, expected):
    loaded = imageio.imread(mask)
    converted = ev.convert_mask(loaded)
    print("converted:\n", converted,)
    print("excpected:\n ")
    assert np.array_equal(converted, expected)

# TODO: Test_all_slum: Convert a file with all 127 to get mask_none, but throw warning.
# TODO: Test_no_slum: Convert a file with all 0 to get mask_all, but throw warning.
# TODO: Test_slum_left_no_slum_right: Convert a file that has a mix and see the correct result.


def test_wrong_pred_value():
    pred_wrong_value = np.arange(122, 130).reshape(2, 4).astype("uint8")
    with pytest.raises(ValueError):
        ev.convert_pred(pred_wrong_value)


def test_uniform_pred_value_warning(all_0_no_mask):
    with pytest.warns(UserWarning):
        ev.convert_pred(all_0_no_mask)

# TODO: missing pngs because these fixtures don't save the images. make fresh fixtures. last one should fail.
# Note: using mask fixtures for convenience, but the conversion is different
@pytest.mark.parametrize("pred,expected", [
    ("tests/all_black.png", pytest.lazy_fixture("mask_none")),
    ("tests/all_grey.png", pytest.lazy_fixture("mask_all")),
    ("tests/grey_top_black_bottom.png", pytest.lazy_fixture("mask_top"))
    ])
def test_convert_pred(pred, expected):
    loaded = imageio.imread(pred)
    converted = ev.convert_pred(loaded)
    assert np.array_equal(converted, expected)


########################################################################################################################
# Evaluation                                                                                                           #
########################################################################################################################
# TODO: Handle case of different masks for pred and truth.
# TODO: Test corner cases: full mask, all slum, no slum
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


def test_mixed_confusion_map(mixed_pred, mixed_truth, conf_map_actual):
    conf_map_calculated = ev.conf_map(mixed_pred, mixed_truth)
    assert np.array_equal(conf_map_calculated, conf_map_actual)


def test_wrong_prediction_value(mixed_truth):
    preds = np.array([
        [-1, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 0]])
    pred_wrong_value = ma.masked_array(preds, np.zeros((3, 6)))
    with pytest.raises(ValueError):
        ev.conf_map(pred_wrong_value, mixed_truth)


def test_wrong_truth_value(mixed_pred):
    truth = np.array([
        [1, 0, 0, 0, 1, 1],
        [1, 0, 0, 4, 1, 1],
        [1, 0, 0, 0, 1, 1]])
    truth_wrong_value = ma.masked_array(truth, np.zeros((3, 6)))
    with pytest.raises(ValueError):
        ev.conf_map(mixed_pred, truth_wrong_value)


@pytest.fixture
def confusion_matrix_actual():
    confusion_matrix = {'fn': 6, 'fp': 5, 'tn': 4, 'tp': 3}
    return confusion_matrix


def test_mixed_confusion_matrix(conf_map_actual, confusion_matrix_actual):
    confusion_matrix_calculated = ev.conf_matrix(conf_map_actual)
    assert np.array_equal(confusion_matrix_calculated, confusion_matrix_actual)

# TODO: Add test that no warning is thrown when all four values are present
def test_confusion_matrix_warning():
    with pytest.warns(UserWarning):
        map = np.array([
            ["tp", "tp", "tp", "fp", "fn", "fn"],
            ["tp", "tp", "fp", "fp", "fn", "fn"],
            ["tp", "tp", "fp", "fp", "fn", "fn"]])
        map_array = ma.masked_array(map)
        ev.conf_matrix(map_array)


# TODO: Refactor test_xx_confusion_matrix tests to mark.parametrize
def test_tp_confusion_matrix():
    tp_map = np.array([
        ["tp", "tp"],
        ["tp", "tp"]])
    conf_matrix = ev.conf_matrix(tp_map)
    assert conf_matrix == {'fn': 0, 'fp': 0, 'tn': 0, 'tp': 4}


def test_tn_confusion_matrix():
    tn_map = np.array([
        ["tn", "tn"],
        ["tn", "tn"]])
    conf_matrix = ev.conf_matrix(tn_map)
    assert conf_matrix == {'fn': 0, 'fp': 0, 'tn': 4, 'tp': 0}


def test_fp_confusion_matrix():
    fp_map = np.array([
        ["fp", "fp"],
        ["fp", "fp"]])
    conf_map = ev.conf_matrix(fp_map)
    assert np.array_equal(conf_map, {'fn': 0, 'fp': 4, 'tn': 0, 'tp': 0})


def test_fn_confusion_matrix():
    fn_map = np.array([
        ["fn", "fn"],
        ["fn", "fn"]])
    conf_map = ev.conf_matrix(fn_map)
    assert np.array_equal(conf_map, {'fn': 4, 'fp': 0, 'tn': 0, 'tp': 0})


def test_pixel_accuracy(confusion_matrix_actual):
    pixel_accuracy_calculated = ev.pixel_acc(confusion_matrix_actual)
    assert pixel_accuracy_calculated == 7/18


def test_precision(confusion_matrix_actual):
    precision_calculated = ev.precision(confusion_matrix_actual)
    assert precision_calculated == 3/8


def test_zero_denominator_precision():
    precision_calculated = ev.precision({'fn': 1, 'fp': 0, 'tn': 1, 'tp': 0})
    assert precision_calculated == 0


def test_recall(confusion_matrix_actual):
    recall_calculated = ev.recall(confusion_matrix_actual)
    assert recall_calculated == 3/9


def test_zero_denominator_recall():
    recall_calculated = ev.recall({'fn': 0, 'fp': 1, 'tn': 1, 'tp': 0})
    assert recall_calculated == 0


def test_f_one(confusion_matrix_actual):
    f_one_calculated = ev.f_one(confusion_matrix_actual)
    assert f_one_calculated == (2 * 3/8 * 3/9) / (3/8 + 3/9)


def test_zero_denominator_f_one():
    f_one_calculated = ev.f_one({'fn': 1, 'fp': 1, 'tn': 1, 'tp': 0})
    assert f_one_calculated == 0


def test_iou(confusion_matrix_actual):
    iou_calculated = ev.iou(confusion_matrix_actual)
    assert iou_calculated == 3/14


def test_zero_denominator_iou():
    iou_calculated = ev.iou({'fn': 0, 'fp': 0, 'tn': 1, 'tp': 0})
    assert iou_calculated == 0


########################################################################################################################
# Integration                                                                                                          #
########################################################################################################################
# TODO: Tests evaluate() with masks
# TODO: Test conf_map() and conf_mat() with masks
# TODO: Make failing test to start with that shows desired behaviour, then build underlying until it fails

def test_compile_metrics(confusion_matrix_actual):
    metrics = ev.compile_metrics(confusion_matrix_actual)
    assert metrics == {
        "Pixel Accuracy": 7/18,
        "Precision": 3/8,
        "Recall": 3/9,
        "F1 Score": (2 * 3/8 * 3/9) / (3/8 + 3/9),
        "Intersection over Union": 3/14}


def test_evaluate_all_correct_nomask(all_127_no_mask):
    results = ev.evaluate("tests/all_grey.png", "tests/all_grey.png")
    assert results == {
        "Pixel Accuracy": 1,
        "Precision": 1,
        "Recall": 1,
        "F1 Score": 1,
        "Intersection over Union": 1}


def test_evaluate_all_wrong_nomask(all_127_no_mask, all_0_no_mask):
    results = ev.evaluate("tests/all_grey.png", "tests/all_black.png")
    assert results == {
        "Pixel Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "Intersection over Union": 0}


def test_evaluate_top_correct_nomask(all_127_no_mask, mask_bottom):
    results = ev.evaluate("tests/all_grey.png", "tests/grey_top_black_bottom.png")
    assert results == {
        "Pixel Accuracy": 0.5,
        "Precision": 0.5,
        "Recall": 1,
        "F1 Score": 1/1.5,
        "Intersection over Union": 0.5}


def test_evaluate_bottom_correct_nomask(all_0_no_mask, mask_bottom):
    results = ev.evaluate("tests/all_black.png", "tests/grey_top_black_bottom.png")
    assert results == {
        "Pixel Accuracy": 0.5,
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "Intersection over Union": 0}


def test_evaluate_bottom_correct_bottom_masked(all_0_no_mask, mask_bottom):
    results = ev.evaluate(
        "tests/all_black.png", "tests/grey_top_black_bottom.png", mask="tests/grey_top_black_bottom.png")
    assert results == {
        "Pixel Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "Intersection over Union": 0}


def test_evaluate_all_true_bottom_masked(all_0_no_mask, mask_bottom):
    results = ev.evaluate("tests/all_grey.png", "tests/all_grey.png", mask="tests/grey_top_black_bottom.png")
    assert results == {
        "Pixel Accuracy": 1,
        "Precision": 1,
        "Recall": 1,
        "F1 Score": 1,
        "Intersection over Union": 1}


def test_evaluate_all_wrong_bottom_masked(all_127_no_mask, all_0_no_mask, mask_bottom):
    results = ev.evaluate("tests/all_grey.png", "tests/all_black.png", mask="tests/grey_top_black_bottom.png")
    assert results == {
        "Pixel Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "Intersection over Union": 0}


def test_evaluate_topleft_bottomright_correct_nomask(mask_right, mask_bottom):
    results = ev.evaluate("tests/grey_left_black_right.png", "tests/grey_top_black_bottom.png")
    assert results == {
        "Pixel Accuracy": 0.5,
        "Precision": 0.5,
        "Recall": 0.5,
        "F1 Score": 0.5,
        "Intersection over Union": 1/3}


def test_evaluate_topleft_bottomright_correct_bottom_masked(mask_right, mask_bottom):
    results = ev.evaluate(
        "tests/grey_left_black_right.png", "tests/grey_top_black_bottom.png", mask="tests/grey_top_black_bottom.png")
    assert results == {
        "Pixel Accuracy": 0.5,
        "Precision": 1,
        "Recall": 0.5,
        "F1 Score": 1/1.5,
        "Intersection over Union": 0.5}