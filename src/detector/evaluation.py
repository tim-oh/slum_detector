import numpy as np
import numpy.ma as ma
import imageio
import warnings
from tabulate import tabulate
import src.detector.data_prep

# Consider evaluation for multiple files. How to aggregate predictions from different cities?
# A utility to stitch predictions for neighbouring areas together could be useful, depending on satellite imagery.
# Also consider a comparison utility that shows the differences between two predictions


# TODO: Refactor clunky conditionals, insert docstrings
def conf_map(pred, truth):
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
    pixel_acc = (conf_mat['tp'] + conf_mat['tn']) / (conf_mat['tp'] + conf_mat['tn'] + conf_mat['fp'] + conf_mat['fn'])
    return pixel_acc


def precision(conf_mat):
    if conf_mat['tp'] + conf_mat['fp'] == 0:
        precision = 0
    else:
        precision = conf_mat['tp'] / (conf_mat['tp'] + conf_mat['fp'])
    return precision


def recall(conf_mat):
    if conf_mat['tp'] + conf_mat['fn'] == 0:
        recall = 0
    else:
        recall = conf_mat['tp'] / (conf_mat['tp'] + conf_mat['fn'])
    return recall


def f_one(conf_mat):
    prec = precision(conf_mat)
    rec = recall(conf_mat)
    if prec + rec == 0:
        f_one = 0
    else:
        f_one = (2 * prec * rec) / (prec + rec)
    return f_one


def iou(conf_mat):
    if conf_mat['tp'] + conf_mat['fp'] + conf_mat['fn'] == 0:
        iou = 0
    else:
        iou = conf_mat['tp'] / (conf_mat['tp'] + conf_mat['fp'] + conf_mat['fn'])
    return iou


def compile_metrics(conf_matrix):
    metrics = {
        "Pixel Accuracy": pixel_acc(conf_matrix),
        "Precision": precision(conf_matrix),
        "Recall": recall(conf_matrix),
        "F1 Score": f_one(conf_matrix),
        "Intersection over Union": iou(conf_matrix)}
    metrics_list = list(metrics.items())
    headers = ["Metric", "Value"]
    print(tabulate(metrics_list, headers, tablefmt="rst", numalign="center", floatfmt=".4f"))
    return metrics


def evaluate(pred_png, truth_png, mask=[]):
    """
    ENTER DESCRIPTION

    :param pred_png:
    :param truth_png:
    :param mask_png:
    :return:
    """
    preds = src.detector.data_prep.png_to_labels(pred_png, mask)
    truth = src.detector.data_prep.png_to_labels(truth_png, mask)
    confusion_map = conf_map(preds, truth)
    confusion_matrix = conf_matrix(confusion_map)
    results = compile_metrics(confusion_matrix)
    return results

# TEMPORARY FUNCTION TO DEAL WITH A MISSING (FIRST?) COLUMN IN THE PREDICTION ARRAY
def evaluate2(pred_png, truth_png, mask=[]):
    """
    ENTER DESCRIPTION

    :param pred_png:
    :param truth_png:
    :param mask_png:
    :return:
    """
    preds = src.detector.data_prep.png_to_labels2(pred_png, mask)
    truth = src.detector.data_prep.png_to_labels(truth_png, mask)
    confusion_map = conf_map(preds, truth[:, 1:]) # Remove the first column
    confusion_matrix = conf_matrix(confusion_map)
    results = compile_metrics(confusion_matrix)
    return results

# Run evaluate() on the slums-world prediction vs Mumbai ground truth
# evaluate2("./../predictions/slums-world_17082020/pred_y.png",
#           "./../predictions/slums-world_17082020/true_y.png",
#           "./../predictions/slums-world_17082020/mask.png")



