"""
Evaluate the predictive fit of a slum map against the ground truth.

-- Scripts

evaluate(pred, truth, mask): script to convert pngs to masked arrays, compile metrics of fit and print a results table.

Args -- pred: predicted slum map; truth: ground truth slum map

Optional args -- mask: area-of-interest mask.

Example usage without mask:
evaluate("path/to/prediction.png", "path/to/ground_truth.png")

Example usage with mask:
evaluate("path/to/prediction.png", "path/to/ground_truth.png", "path/to/mask.png")

-- Support functions

conf_map(): mark each pixel's prediction as true/false positive/negative on a 'confusion map'.

conf_mat(): aggregate the confusion map into a confusion matrix dictionary and prints the matrix.

pixel_acc(): compute pixel accuracy.

precision(): compute precision.

recall(): compute recall.

f_one(): compute F1 score.

iou(): compute intersection over union.

compile_metrics(): assemble metrics in a dictionary and print as a table.
"""

import numpy as np
import numpy.ma as ma
import warnings
from tabulate import tabulate
import src.detector.data_prep


# TODO: Refactor clunky conditionals, perhaps with a dictionary.
def _conf_map(pred, truth):
    """
    Produce a confusion map of the predictions, meaning a mapping of pixel-level true/false positives/negatives.

    The map serves as a basis for compilation of a standard confusion matrix.
    It can be overlaid on its satellite image to highlight failure cases and differences between model predictions.
    True positive = "tp", false positive = "fp", true negative = "tn", false negative = "fn"

    :param pred: Two-dimensional prediction array of same (x, y) size as satellite image; slum = 1 and non-slum = 1.
    :param truth: Two-dimensional ground truth array of same (x, y) size as satellite image; slum = 1 and non-slum = 1.
    :return: Confusion map array of same (x, y) size as satellite image.
    """
    if not pred.shape == truth.shape:
        raise ValueError("Array sizes: shape of predictions must equal shape of ground truth %r." % str(pred.shape))
    conf_map = ma.array(np.empty(pred.shape), mask=np.zeros(pred.shape)).astype('str')
    for i in np.arange(0, conf_map.shape[0]):
        for j in np.arange(0, conf_map.shape[1]):
            if pred[i, j] == 1 and truth[i, j] == 1:
                conf_map[i, j] = "tp"
            elif pred[i, j] == 0 and truth[i, j] == 0:
                conf_map[i, j] = "tn"
            elif pred[i, j] == 1 and truth[i, j] == 0:
                conf_map[i, j] = "fp"
            elif pred[i, j] == 0 and truth[i, j] == 1:
                conf_map[i, j] = "fn"
            elif pred[i, j] is ma.masked and truth[i, j] is ma.masked:
                conf_map.mask[i, j] = True
            else:
                if not pred[i, j] == 0 or pred[i, j] == 1:
                    raise ValueError("Prediction values: pixels must be 0, 1 or masked, but is %r." % pred[i, j])
                if not truth[i, j] == 0 or truth[i, j] == 1:
                    raise ValueError("Ground truth values: pixels must be 0, 1 or masked but is %r." % truth[i, j])
    return conf_map


def _conf_matrix(conf_map):
    """
    Count sum of pixel-level true positives/false positives/true negatives/false negatives and print results table.

    :param conf_map: Confusion map produced by conf_map().
    :return: Standard confusion matrix, also printed to stdout as a table.
    """
    markers, counts = np.unique(conf_map.data, return_counts=True)
    conf_matrix = dict(zip(markers, counts))
    required_keys = ["fn", "fp", "tn", "tp"]
    for key in required_keys:
        try:
            conf_matrix[key]
        except KeyError:
            warnings.warn("Confusion matrix: no %r." % key, UserWarning)
            conf_matrix[key] = 0
    table_entries = np.array([
        ["Truth: slum", conf_matrix["tp"], conf_matrix["fn"]],
        ["Truth: non-slum", conf_matrix["fp"], conf_matrix["tn"]]
        ])
    headers = ["Confusion matrix", "Prediction: slum", "Prediction: non-slum"]
    print(tabulate(table_entries, headers, tablefmt="rst", numalign="center"))
    return conf_matrix


def _pixel_acc(conf_mat):
    """
    Compute pixel-level prediction accuracy, i.e. the ratio of true positives plus true negatives to number of pixels.

    Answers the question: "Which share of the pixels did the model predict correctly?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return: Pixel accuracy score, ranging from 0 to 1.
    """
    pixel_acc = (conf_mat['tp'] + conf_mat['tn']) / (conf_mat['tp'] + conf_mat['tn'] + conf_mat['fp'] + conf_mat['fn'])
    return pixel_acc


def _precision(conf_mat):
    """
    Compute the precision score, i.e. the ratio of true positives to true positives plus false positives.

    Answers the question: "Which share of the pixels predicted by the model as slum was actually slum?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return: Precision score, ranging from 0 to 1.
    """
    if conf_mat['tp'] + conf_mat['fp'] == 0:
        precision = 0
    else:
        precision = conf_mat['tp'] / (conf_mat['tp'] + conf_mat['fp'])
    return precision


def _recall(conf_mat):
    """
    Compute the recall score, i.e. the ratio of true positives to true positives and false negatives.

    Answers the question: "Which share of the pixels that are actually slum was identified by the model as such?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return: Recall score, ranging from 0 to 1.
    """
    if conf_mat['tp'] + conf_mat['fn'] == 0:
        recall = 0
    else:
        recall = conf_mat['tp'] / (conf_mat['tp'] + conf_mat['fn'])
    return recall


def _f_one(conf_mat):
    """
    Compute harmonic mean of precision and recall.

    Answers the question: "What is the average of precision and recall?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return:F-1 score, ranging from 0 to 1.
    """
    prec = _precision(conf_mat)
    rec = _recall(conf_mat)
    if prec + rec == 0:
        f_one = 0
    else:
        f_one = (2 * prec * rec) / (prec + rec)
    return f_one


def _iou(conf_mat):
    """
    Compute Intersection over Union (IoU) evaluation metric.

    Answers the question: "What share actual and predicted slum pixels was identified correctly?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return: IoU score, ranging from 0 to 1.
    """
    if conf_mat['tp'] + conf_mat['fp'] + conf_mat['fn'] == 0:
        iou = 0
    else:
        iou = conf_mat['tp'] / (conf_mat['tp'] + conf_mat['fp'] + conf_mat['fn'])
    return iou


def _compile_metrics(conf_mat):
    """
    Collate evaluation metrics by calling corresponding functions. Prints table of metrics.

    :param conf_mat: Confusion matrix produced by conf_mat()
    :return: Dictionary of evaluation metrics.
    """
    metrics = {
        "Pixel Accuracy": _pixel_acc(conf_mat),
        "Precision": _precision(conf_mat),
        "Recall": _recall(conf_mat),
        "F1 Score": _f_one(conf_mat),
        "Intersection over Union": _iou(conf_mat)}
    metrics_list = list(metrics.items())
    headers = ["Metric", "Value"]
    print(tabulate(metrics_list, headers, tablefmt="rst", numalign="center", floatfmt=".4f"))
    return metrics


def evaluate(pred_png, truth_png, mask_png=None):
    """
    Orchestrate evaluation of predictions versus ground truth by calling computation functions.

    Prints confusion matrix and evaluation metrics to stdout.
    Coding of pngs needs to match slums world conventions: 63 < slum < 128, 0 = <non-slum =< 63; masked=127, non-mask=0.

    :param pred_png: Prediction png of (x, y) size matching underlying satellite image.
    :param truth_png: Ground truth png of (x, y) size matching underlying satellite image.
    :param mask_png: Mask png of (x, y) size matching underlying satellite image.
    :return: Dictionary of evaluation metrics.
    """
    preds = src.detector.data_prep._png_to_labels(pred_png, mask=mask_png)
    truth = src.detector.data_prep._png_to_labels(truth_png, mask=mask_png)
    confusion_map = _conf_map(preds, truth)
    confusion_matrix = _conf_matrix(confusion_map)
    results = _compile_metrics(confusion_matrix)
    return results