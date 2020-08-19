import numpy as np
import numpy.ma as ma
import imageio
import warnings
from tabulate import tabulate

# Note: consider evaluation for multiple files. How to aggregate predictions from different cities?
# A utility to stitch predictions for neighbouring areas together could be useful, depending on satellite imagery.
# Also consider a comparison utility that shows the differences between two predictions


def png_to_labels(png, mask=[]):
    """
    Turns a png label file into a masked numpy array with converted coding.

    :param png: Label file path relative to working directory.
    :param mask: Optional path to area-of-interest mask corresponding to png; all pixels unmasked if none.
    :return: Masked label array.
    """
    pred_array = imageio.imread("./" + png)
    if mask == []:
        mask_array = np.ones(pred_array.shape) * 127
    else:
        mask_array = imageio.imread("./" + mask)
    pred_converted = convert_pred(pred_array)
    mask_converted = convert_mask(mask_array)
    if not pred_converted.shape == mask_converted.shape:
        raise ValueError(
            "Mask size: mask array size does not match prediction array size %r." % str(pred_converted.shape))
    masked_labels = ma.masked_array(pred_converted, mask_converted)
    return masked_labels


def convert_pred(pred_array):
    """
    Converts slum_detection_lib greyscale label coding [0:63 slum, 64:127 no slum] to [0 no slum, 1 slum].

    :param pred_array: Numpy array of imported pixel labels.
    :return: Numpy array of converted pixel labels.
    """
    valid = np.arange(0, 128)
    if not np.isin(pred_array, valid).all():
        raise ValueError("Label values: all elements must be one of %r." % valid)
    if np.unique(pred_array).ndim == 1:
        warnings.warn("Label values: all elements are set to %r." % pred_array[0, 0], UserWarning)
    pred_array[pred_array <= 63] = 0
    pred_array[pred_array > 63] = 1
    return pred_array

# TODO: Refactor ambiguous use of valid: notation might suggest a range, but it's a set unless notation is a:b
def convert_mask(mask_array):
    """
    Converts slum_detection_lib greyscale pixel coding [127: area of interest, 0: mask] to [0: AOI, 1: mask].

    :param mask_array: Numpy array of imported mask values.
    :return: Numpy array of converted mask values.
    """
    valid = [0, 127]
    if not np.isin(mask_array, valid).all():
        raise ValueError("Mask values: must all be one of %r." % valid)
    if np.unique(mask_array).ndim == 1:
        warnings.warn("Mask values: all elements are set to %r." % mask_array[0, 0], UserWarning)
    mask_array[mask_array == 0] = 1
    mask_array[mask_array == 127] = 0
    return mask_array


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
    preds = png_to_labels(pred_png, mask)
    truth = png_to_labels(truth_png, mask)
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
    preds = png_to_labels2(pred_png, mask)
    truth = png_to_labels(truth_png, mask)
    confusion_map = conf_map(preds, truth[:, 1:]) # Remove the first column
    confusion_matrix = conf_matrix(confusion_map)
    results = compile_metrics(confusion_matrix)
    return results

# Temporary function adjusted for missing (first?) column in prediction array
def png_to_labels2(png, mask):
    """
    Turns a png label file into a masked numpy array with converted coding.

    :param png: Label file path relative to working directory.
    :param mask: Optional path to area-of-interest mask corresponding to png; all pixels unmasked if none.
    :return: Masked label array.
    """
    pred_array = imageio.imread("./" + png)
    mask_array = imageio.imread("./" + mask)
    mask_array = mask_array[:, 1:]
    pred_converted = convert_pred(pred_array)
    mask_converted = convert_mask(mask_array)
    if not pred_converted.shape == mask_converted.shape:
        raise ValueError(
            "Mask size: mask array size does not match prediction array size %r." % str(pred_converted.shape))
    masked_labels = ma.masked_array(pred_converted, mask_converted)
    return masked_labels


# Run evaluate() on the slums-world prediction vs Mumbai ground truth
# evaluate2("./../predictions/slums-world_17082020/pred_y.png",
#           "./../predictions/slums-world_17082020/true_y.png",
#           "./../predictions/slums-world_17082020/mask.png")


