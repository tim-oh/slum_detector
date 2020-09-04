import numpy as np
import numpy.ma as ma
import warnings
from tabulate import tabulate
import src.detector.data_prep


# TODO: Refactor clunky conditionals, perhaps with a dictionary.
def conf_map(pred, truth):
    """
    Produces a confusion map of the predictions, meaning a mapping of pixel-level true/false positives/negatives.
    The map serves as a basis for compilation of a standard confusion matrix.
    The map can be overlaid on its satellite image to highlight failure cases and differences between model predictions.


    :param pred: Two-dimensional prediction array of same (x, y) size as satellite image; slum = 1 and non-slum = 1.
    :param truth: Two-dimensional ground truth array of same (x, y) size as satellite image; slum = 1 and non-slum = 1.
    :return: Confusion map array of same (x, y) size as satellite image;
    True positive = "tp", false positive = "fp", true negative = "tn", false negative = "fn".
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


def conf_matrix(conf_map):
    """
    Counts sum of pixel-level true positives, false positives, true negatives and false negatives. Prints results table.

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


def pixel_acc(conf_mat):
    """
    Computes pixel-level prediction accuracy, i.e. the ratio of true positives plus true negatives to number of pixels.
    Answers the question: "Which share of the pixels did the model predict correctly?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return: Pixel accuracy score, ranging from 0 to 1.
    """
    pixel_acc = (conf_mat['tp'] + conf_mat['tn']) / (conf_mat['tp'] + conf_mat['tn'] + conf_mat['fp'] + conf_mat['fn'])
    return pixel_acc


def precision(conf_mat):
    """
    Computes the precision score, i.e. the ratio of true positives to true positives plus false positives.
    Answers the question: "Which share of the pixels predicted by the model as slum was actually slum?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return: Precision score, ranging from 0 to 1.
    """
    if conf_mat['tp'] + conf_mat['fp'] == 0:
        precision = 0
    else:
        precision = conf_mat['tp'] / (conf_mat['tp'] + conf_mat['fp'])
    return precision


def recall(conf_mat):
    """
    Computes the recall score, i.e. the ratio of true positives to true positives and false negatives.
    Answers the question: "Which share of the pixels that are actually slum was identified by the model as such?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return: Recall score, ranging from 0 to 1.
    """
    if conf_mat['tp'] + conf_mat['fn'] == 0:
        recall = 0
    else:
        recall = conf_mat['tp'] / (conf_mat['tp'] + conf_mat['fn'])
    return recall


def f_one(conf_mat):
    """
    Harmonic mean of precision and recall. Answers the question: "What is the average of precision and recall?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return:F-1 score, ranging from 0 to 1.
    """
    prec = precision(conf_mat)
    rec = recall(conf_mat)
    if prec + rec == 0:
        f_one = 0
    else:
        f_one = (2 * prec * rec) / (prec + rec)
    return f_one


def iou(conf_mat):
    """
    Computes Intersection over Union (IoU) evaluation metric.
    Answers the question: "What share actual and predicted slum pixels was identified correctly?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return: IoU score, ranging from 0 to 1.
    """
    if conf_mat['tp'] + conf_mat['fp'] + conf_mat['fn'] == 0:
        iou = 0
    else:
        iou = conf_mat['tp'] / (conf_mat['tp'] + conf_mat['fp'] + conf_mat['fn'])
    return iou


def compile_metrics(conf_mat):
    """
    Collates evaluation metrics by calling corresponding functions. Prints table of metrics.

    :param conf_mat: Confusion matrix produced by conf_mat()
    :return: Dictionary of evaluation metrics.
    """
    metrics = {
        "Pixel Accuracy": pixel_acc(conf_mat),
        "Precision": precision(conf_mat),
        "Recall": recall(conf_mat),
        "F1 Score": f_one(conf_mat),
        "Intersection over Union": iou(conf_mat)}
    metrics_list = list(metrics.items())
    headers = ["Metric", "Value"]
    print(tabulate(metrics_list, headers, tablefmt="rst", numalign="center", floatfmt=".4f"))
    return metrics


def evaluate(pred_png, truth_png, mask_png=None):
    """
    Script to orchestrate evaluation of predictions versus ground truth by calling computation functions.
    Prints confusion matrix and evaluation metrics to stdout.
    Coding of pngs needs to match slums world conventions: 63 < slum < 128, 0 = <non-slum =< 63; masked=127, non-mask=0.

    :param pred_png: Prediction png of (x, y) size matching underlying satellite image.
    :param truth_png: Ground truth png of (x, y) size matching underlying satellite image.
    :param mask_png: Mask png of (x, y) size matching underlying satellite image.
    :return: Dictionary of evaluation metrics.
    """
    if mask_png:
        preds = src.detector.data_prep.png_to_labels(pred_png, mask_png)
        truth = src.detector.data_prep.png_to_labels(truth_png, mask_png)
    else:
        preds = src.detector.data_prep.png_to_labels(pred_png)
        truth = src.detector.data_prep.png_to_labels(truth_png)
    confusion_map = conf_map(preds, truth)
    confusion_matrix = conf_matrix(confusion_map)
    results = compile_metrics(confusion_matrix)
    return results


def evaluate2(pred_png, truth_png, mask_png=None):
    """Temporary function to deal with a missing column in a slums-world prediction, otherwise same as evaluate."""
    if mask_png:
        preds = src.detector.data_prep.png_to_labels(pred_png, mask_png)
        truth = src.detector.data_prep.png_to_labels(truth_png, mask_png)
    else:
        preds = src.detector.data_prep.png_to_labels(pred_png)
        truth = src.detector.data_prep.png_to_labels(truth_png)
    confusion_map = conf_map(preds, truth[:, 1:]) # Remove the first column
    confusion_matrix = conf_matrix(confusion_map)
    results = compile_metrics(confusion_matrix)
    return results


# Usage of running evaluate() on the slums-world prediction vs Mumbai ground truth:
# evaluate2("./../predictions/slums-world_17082020/pred_y.png",
#           "./../predictions/slums-world_17082020/true_y.png")
# Optional mask argument not working as mask is the wrong size:
# # TODO: Fix by creating new mask.png
# evaluate2("./../predictions/slums-world_17082020/pred_y.png",
#           "./../predictions/slums-world_17082020/true_y.png",
#           "./../predictions/slums-world_17082020/mask.png")
