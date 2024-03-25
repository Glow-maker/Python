import numpy as np
import pandas as pd
from sklearn.utils import assert_all_finite, check_consistent_length
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import type_of_target
from sklearn.utils.extmath import stable_cumsum
from sklearn.metrics import multilabel_confusion_matrix, balanced_accuracy_score, unique_labels
from sklearn.preprocessing import OneHotEncoder
from inspect import signature



def _nanaverage(a, weights=None):
    """
    Compute the weighted average, ignoring NaNs.
    
    Parameters
    ----------
    a : ndarray
        Array containing data to be averaged.
    weights : array-like, default=None
        An array of weights associated with the values in `a`. Each value in `a`
        contributes to the average according to its associated weight. The
        weights array can either be 1-D of the same shape as `a`. If weights=None,
        then all data in `a` are assumed to have a weight equal to one.
    
    Returns
    -------
    weighted_average : float
        The weighted average.
    
    Notes
    -----
    This function combines `numpy.average` and `numpy.nanmean`, so
    that NaN values are ignored from the average and weights can
    be passed. Note that when possible, this function delegates to the prime methods.
    """
    if len(a) == 0:
        return np.nan
    
    mask = np.isnan(a)
    if mask.all():
        return np.nan
    
    if weights is None:
        return np.nanmean(a)
    
    weights = np.array(weights, copy=False)
    a, weights = a[~mask], weights[~mask]
    try:
        return np.average(a, weights=weights)
    except ZeroDivisionError:
        # This occurs when all weights are zero, then ignore them
        return np.average(a)



def binary_cif_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """
    Calculate true and false positives per binary classification threshold.
    This function is adapted from sklearn.metrics._ranking module.
    
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True targets of binary classification.
    y_score : ndarray of shape (n_samples,)
        Estimated probabilities or output of a decision function.
    pos_label : int, float, bool or str, default=None
        The label of the positive class.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    
    Returns
    -------
    fps : ndarray of shape (n_thresholds,)
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : ndarray of shape (n_thresholds,)
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : ndarray of shape (n_thresholds,)
        Decreasing score values.
    """
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true, input_name="y_true")
    if not (y_type == "binary" or (y_type == "multiclass" and pos_label is not None)):
        raise ValueError(f"{y_type} format is not supported")

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    # Filter out zero-weighted samples, as they should not impact the result
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        nonzero_weight_mask = sample_weight != 0
        y_true = y_true[nonzero_weight_mask]
        y_score = y_score[nonzero_weight_mask]
        sample_weight = sample_weight[nonzero_weight_mask]

    # Make y_true a boolean vector
    y_true = y_true == pos_label

    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1

    # Extract indices associated with distinct values
    distinct_value_indices = np.where(np.diff(y_score) != 0)[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps

    return fps, tps, y_score[threshold_idxs]


def gain_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True, metric="precision"):
    """
    Compute Gain Curve for binary classification.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int, float, bool or str, default=None
        The label of the positive class.
        When 'pos_label=None', if y_true is in {-1, 1} or {0, 1},
        pos_label is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    drop_intermediate : bool, default=True
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.

    metric : str, default='precision'
        The metric to use as y-axis. Must be in {'precision', 'lift', 'recall'}.

    Returns
    -------
    y_axis : ndarray
        The values for the Y-axis of the gain curve based on the specified metric.

    pct : ndarray
        The percentage of samples that have been classified up to each threshold.

    thresholds : ndarray
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. 'thresholds[0]' represents no instances being predicted
        and is arbitrarily set to 'np.inf'.
    """
    # This line is intended to call a function '_binary_cif_curve' which seems to be a typo or a placeholder.
    # You should replace '_binary_cif_curve' with the correct function call, for example, 'binary_cif_curve'.
    fps, tps, thresholds = binary_cif_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(
            np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True]
        )[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, thresholds]

    if metric == "precision":
        with np.errstate(divide='ignore', invalid='ignore'):
            y_axis = np.divide(tps, (tps + fps))
            y_axis[np.isnan(y_axis)] = 0
    elif metric == "lift":
        denominator = (tps[-1] / (tps[-1] + fps[-1]))
        y_axis = np.divide(tps, (tps + fps)) / denominator
    elif metric == "recall":
        y_axis = tps / tps[-1]
    else:
        raise ValueError("Invalid metric for y-axis.")

    pct = (tps + fps) / (tps[-1] + fps[-1])

    return y_axis, pct, thresholds




def mcm_based_score(y_true, y_pred, sample_weight=None):
    """
    Calculate various metrics based on the multi-label confusion matrix.
    """
    # Calculate tp, fp, tn, fn, pred_p, true_p, true_n, all_p_n
    MCM = multilabel_confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    tp = MCM[:, 1, 1]
    fp = MCM[:, 0, 1]
    tn = MCM[:, 0, 0]
    fn = MCM[:, 1, 0]
    pred_p = tp + fp
    true_p = tp + fn
    true_n = tn + fp
    all_p_n = tp + tn + fp + fn

    # Calculate metrics
    precision = np.divide(tp, pred_p, out=np.zeros_like(tp), where=pred_p!=0)
    recall = np.divide(tp, true_p, out=np.zeros_like(tp), where=true_p!=0)
    fpr = np.divide(fp, true_n, out=np.zeros_like(fp), where=true_n!=0)
    accuracy = np.divide(tp + tn, all_p_n, out=np.zeros_like(tp + tn), where=all_p_n!=0)
    f1_score = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall)!=0)
    lift = np.divide(precision, np.divide(true_p, all_p_n, out=np.zeros_like(true_p), where=all_p_n!=0), out=np.zeros_like(precision), where=np.divide(true_p, all_p_n, out=np.zeros_like(true_p), where=all_p_n!=0)!=0)

    # Per-class scores
    columns = ["precision", "recall", "fpr", "accuracy", "f1_score", "lift"]
    labels = unique_labels(y_true, y_pred)
    index = [f"class_{i}" for i in labels]
    scores = pd.DataFrame(np.column_stack((precision, recall, fpr, accuracy, f1_score, lift)), columns=columns, index=index)

    # Weighted and macro scores
    scores_weighted = [_nanaverage(scores[var], weights=true_p) for var in scores.columns]
    scores_macro = [_nanaverage(scores[var], weights=None) for var in scores.columns]
    scores.loc["weighted", :] = scores_weighted
    scores.loc["macro", :] = scores_macro

    # Micro scores
    tp_sum = tp.sum()
    pred_p_sum = pred_p.sum()
    true_n_sum = true_n.sum()
    precision_m = tp_sum / pred_p_sum if pred_p_sum > 0 else 0
    recall_m = tp_sum / true_p.sum() if true_p.sum() > 0 else 0
    f1_score_m = 2 * precision_m * recall_m / (precision_m + recall_m) if (precision_m + recall_m) > 0 else 0
    scores_micro = [precision_m, recall_m, fp.sum() / true_n_sum if true_n_sum > 0 else 0, balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight), f1_score_m, precision_m * len(labels) / len(labels)]
    scores.loc["micro", :] = scores_micro

    return scores

def multiclass_curve_wrapper(curve_func, y_true, y_score, sample_weight=None, labels=None):
    if labels is None:
        labels = unique_labels(y_true)
    res = {}
    for i, label in enumerate(labels):
        axis = curve_func(y_true == label, y_score[:, i], sample_weight=sample_weight)
        res[f"class_{label}"] = list(zip(axis[0], axis[1]))
    return res

def multiclass_score_wrapper(score_func, y_true, y_score, sample_weight=None, labels=None):
    if labels is None:
        labels = np.unique(y_true).tolist()
    averages = ["weighted", "macro", "micro"]
    kwargs = {"average": None, "sample_weight": sample_weight}
    
    if "multi_class" in signature(score_func).parameters:
        kwargs["multi_class"] = "ovr"
    
    res = []
    for i, label in enumerate(labels):
        kwargs['labels'] = [label]  # Ensure score is calculated per class
        score = score_func(y_true == label, y_score[:, i], **kwargs)
        res.append(score)
    
    for average in averages:
        kwargs["average"] = average
        score = score_func(y_true, y_score, **kwargs)
        res.append(score)
    
    res_df = pd.DataFrame({"score": res}, index=[f"class_{label}" for label in labels] + averages)
    return res_df

def correlation(features, target, weight):
    """
    Calculate correlation between features and target, which is one-hot encoded.

    Parameters:
    - features: Pandas DataFrame, features to calculate correlation against target.
    - target: array-like, target variable to be one-hot encoded.
    - weight: array-like, weights for each sample.
    
    Returns:
    - Pandas DataFrame of correlation coefficients between each feature and target class.
    """
    encoder = OneHotEncoder(sparse_output=False,
                            feature_name_combiner=lambda f,c: f"class_{c}")
    target = np.asarray(target)
    if target.ndim == 1:
        target = target[:, None]

    class_data = encoder.fit_transform(target)
    class_mean = np.average(class_data, axis=0, weights=weight)
    class_std = np.sqrt(np.average((class_data - class_mean) ** 2, axis=0, weights=weight))
    feature_data = features.to_numpy()

    feature_mean = np.average(feature_data, axis=0, weights=weight)
    feature_std = np.sqrt(np.average((feature_data - feature_mean) ** 2, axis=0, weights=weight))
    weighted_sum = np.sum(weight)
    covariance = (feature_data.T * weight[:, None]).dot(class_data) / weighted_sum - np.outer(feature_mean, class_mean)
    corr = covariance / np.outer(feature_std, class_std)
    res = pd.DataFrame(corr, columns=encoder.get_feature_names_out())
    res["feature"] = features.columns

    return res

def add_total(mtx):
    """
    Add a total sum row at the bottom and a total sum column on the right side of the matrix.
    
    Parameters:
    - mtx: numpy.ndarray, the matrix to add totals to.
    
    Returns:
    - numpy.ndarray with added totals row and column.
    """
    # Add a total sum row at the bottom
    mtx = np.row_stack((mtx, mtx.sum(axis=0)))
    
    # Add a total sum column on the right
    mtx = np.column_stack((mtx, mtx.sum(axis=1)))
    
    return mtx

def auc_curve_from_roc(x, y):
    """
    Calculate the cumulative Area Under the Curve (AUC) from ROC curve coordinates.
    
    Parameters:
    - x: array-like, False Positive Rate (FPR) coordinates of the ROC curve.
    - y: array-like, True Positive Rate (TPR) coordinates of the ROC curve.
    
    Returns:
    - List of tuples (x, cumulative AUC) up to each point.
    """
    x = column_or_1d(x)
    y = column_or_1d(y)
    
    # Calculate the differences between consecutive x values
    dx = np.diff(x)
    
    # Compute the cumulative AUC using the Trapezoidal rule
    cum_auc = np.cumsum(dx * (y[:-1] + y[1:]) * 0.5)
    
    return list(zip(x, np.r_[0, cum_auc]))
