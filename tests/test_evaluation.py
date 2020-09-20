import pytest
import numpy as np
import numpy.ma as ma
import src.detector.evaluation
import src.detector.data_prep


########################################################################################################################
# Loading                                                                                                              #
#######################################################################################################################
def test_inconsistent_pred_truth_sizes(all_127_no_mask_array, mask_small_array):
    with pytest.raises(ValueError):
        src.detector.evaluation._conf_map(all_127_no_mask_array, mask_small_array)


########################################################################################################################
# Evaluation                                                                                                           #
########################################################################################################################
# TODO: Test corner cases: full mask, all slum, no slum.
# TODO: Test conf_map() with masked predictions.
def test_mixed_confusion_map(mixed_pred, mixed_truth, conf_map_actual):
    conf_map_calculated = src.detector.evaluation._conf_map(mixed_pred, mixed_truth)
    assert np.array_equal(conf_map_calculated, conf_map_actual)


def test_wrong_prediction_value(mixed_truth):
    preds = np.array([
        [-1, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 0]])
    pred_wrong_value = ma.masked_array(preds, np.zeros((3, 6)))
    with pytest.raises(ValueError):
        src.detector.evaluation._conf_map(pred_wrong_value, mixed_truth)


def test_wrong_truth_value(mixed_pred):
    truth = np.array([
        [1, 0, 0, 0, 1, 1],
        [1, 0, 0, 4, 1, 1],
        [1, 0, 0, 0, 1, 1]])
    truth_wrong_value = ma.masked_array(truth, np.zeros((3, 6)))
    with pytest.raises(ValueError):
        src.detector.evaluation._conf_map(mixed_pred, truth_wrong_value)


# TODO: Test conf_mat() with masked predictions.
def test_mixed_confusion_matrix(conf_map_actual, confusion_matrix_actual):
    confusion_matrix_calculated = src.detector.evaluation._conf_matrix(conf_map_actual)
    assert np.array_equal(confusion_matrix_calculated, confusion_matrix_actual)


# TODO: Add test that no warning is thrown when all four values are present
def test_confusion_matrix_warning():
    with pytest.warns(UserWarning):
        map = np.array([
            ["tp", "tp", "tp", "fp", "fn", "fn"],
            ["tp", "tp", "fp", "fp", "fn", "fn"],
            ["tp", "tp", "fp", "fp", "fn", "fn"]])
        map_array = ma.masked_array(map)
        src.detector.evaluation._conf_matrix(map_array)


# TODO: Refactor test_xx_confusion_matrix tests to mark.parametrize
def test_tp_confusion_matrix():
    tp_map = np.array([
        ["tp", "tp"],
        ["tp", "tp"]])
    conf_matrix = src.detector.evaluation._conf_matrix(tp_map)
    assert conf_matrix == {'fn': 0, 'fp': 0, 'tn': 0, 'tp': 4}


def test_tn_confusion_matrix():
    tn_map = np.array([
        ["tn", "tn"],
        ["tn", "tn"]])
    conf_matrix = src.detector.evaluation._conf_matrix(tn_map)
    assert conf_matrix == {'fn': 0, 'fp': 0, 'tn': 4, 'tp': 0}


def test_fp_confusion_matrix():
    fp_map = np.array([
        ["fp", "fp"],
        ["fp", "fp"]])
    conf_map = src.detector.evaluation._conf_matrix(fp_map)
    assert np.array_equal(conf_map, {'fn': 0, 'fp': 4, 'tn': 0, 'tp': 0})


def test_fn_confusion_matrix():
    fn_map = np.array([
        ["fn", "fn"],
        ["fn", "fn"]])
    conf_map = src.detector.evaluation._conf_matrix(fn_map)
    assert np.array_equal(conf_map, {'fn': 4, 'fp': 0, 'tn': 0, 'tp': 0})


def test_pixel_accuracy(confusion_matrix_actual):
    pixel_accuracy_calculated = src.detector.evaluation._pixel_acc(confusion_matrix_actual)
    assert pixel_accuracy_calculated == 7/18


def test_precision(confusion_matrix_actual):
    precision_calculated = src.detector.evaluation._precision(confusion_matrix_actual)
    assert precision_calculated == 3/8


def test_zero_denominator_precision():
    precision_calculated = src.detector.evaluation._precision({'fn': 1, 'fp': 0, 'tn': 1, 'tp': 0})
    assert precision_calculated == 0


def test_recall(confusion_matrix_actual):
    recall_calculated = src.detector.evaluation._recall(confusion_matrix_actual)
    assert recall_calculated == 3/9


def test_zero_denominator_recall():
    recall_calculated = src.detector.evaluation._recall({'fn': 0, 'fp': 1, 'tn': 1, 'tp': 0})
    assert recall_calculated == 0


def test_f_one(confusion_matrix_actual):
    f_one_calculated = src.detector.evaluation._f_one(confusion_matrix_actual)
    assert f_one_calculated == (2 * 3/8 * 3/9) / (3/8 + 3/9)


def test_zero_denominator_f_one():
    f_one_calculated = src.detector.evaluation._f_one({'fn': 1, 'fp': 1, 'tn': 1, 'tp': 0})
    assert f_one_calculated == 0


def test_iou(confusion_matrix_actual):
    iou_calculated = src.detector.evaluation._iou(confusion_matrix_actual)
    assert iou_calculated == 3/14


def test_zero_denominator_iou():
    iou_calculated = src.detector.evaluation._iou({'fn': 0, 'fp': 0, 'tn': 1, 'tp': 0})
    assert iou_calculated == 0


########################################################################################################################
# Integration                                                                                                          #
########################################################################################################################
def test_compile_metrics(confusion_matrix_actual):
    metrics = src.detector.evaluation._compile_metrics(confusion_matrix_actual)
    assert metrics == {
        "Pixel Accuracy": 7/18,
        "Precision": 3/8,
        "Recall": 3/9,
        "F1 Score": (2 * 3/8 * 3/9) / (3/8 + 3/9),
        "Intersection over Union": 3/14}


# TODO: Test evaluate() with masks
def test_evaluate_all_correct_nomask(all_127_no_mask_png):
    results = src.detector.evaluation.evaluate(all_127_no_mask_png, all_127_no_mask_png)
    assert results == {
        "Pixel Accuracy": 1,
        "Precision": 1,
        "Recall": 1,
        "F1 Score": 1,
        "Intersection over Union": 1}


def test_evaluate_all_wrong_nomask(all_127_no_mask_png, all_0_no_mask_png):
    results = src.detector.evaluation.evaluate(all_127_no_mask_png, all_0_no_mask_png)
    assert results == {
        "Pixel Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "Intersection over Union": 0}


def test_evaluate_top_correct_nomask(all_127_no_mask_png, mask_bottom_png):
    results = src.detector.evaluation.evaluate(all_127_no_mask_png, mask_bottom_png)
    assert results == {
        "Pixel Accuracy": 0.5,
        "Precision": 0.5,
        "Recall": 1,
        "F1 Score": 1/1.5,
        "Intersection over Union": 0.5}


def test_evaluate_bottom_correct_nomask(all_0_no_mask_png, mask_bottom_png):
    results = src.detector.evaluation.evaluate(all_0_no_mask_png, mask_bottom_png)
    assert results == {
        "Pixel Accuracy": 0.5,
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "Intersection over Union": 0}


def test_evaluate_bottom_correct_bottom_masked(all_0_no_mask_png, mask_bottom_png):
    results = src.detector.evaluation.evaluate(
        all_0_no_mask_png, mask_bottom_png, mask_png=mask_bottom_png)
    assert results == {
        "Pixel Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "Intersection over Union": 0}


def test_evaluate_all_true_bottom_masked(all_127_no_mask_png, mask_bottom_png):
    results = src.detector.evaluation.evaluate(
        all_127_no_mask_png,
        all_127_no_mask_png,
        mask_png=mask_bottom_png)
    assert results == {
        "Pixel Accuracy": 1,
        "Precision": 1,
        "Recall": 1,
        "F1 Score": 1,
        "Intersection over Union": 1}


def test_evaluate_all_wrong_bottom_masked(all_127_no_mask_png, all_0_no_mask_png, mask_bottom_png):
    results = src.detector.evaluation.evaluate(
        all_127_no_mask_png,
        all_0_no_mask_png,
        mask_png=mask_bottom_png)
    assert results == {
        "Pixel Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "Intersection over Union": 0}


def test_evaluate_topleft_bottomright_correct_nomask(mask_right_png, mask_bottom_png):
    results = src.detector.evaluation.evaluate(
        mask_bottom_png,
        mask_right_png)
    assert results == {
        "Pixel Accuracy": 0.5,
        "Precision": 0.5,
        "Recall": 0.5,
        "F1 Score": 0.5,
        "Intersection over Union": 1/3}


def test_evaluate_topleft_bottomright_correct_bottom_masked(mask_right_png, mask_bottom_png):
    results = src.detector.evaluation.evaluate(
        mask_right_png,
        mask_bottom_png,
        mask_png=mask_bottom_png)
    assert results == {
        "Pixel Accuracy": 0.5,
        "Precision": 1,
        "Recall": 0.5,
        "F1 Score": 1/1.5,
        "Intersection over Union": 0.5}