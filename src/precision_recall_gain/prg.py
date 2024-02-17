from functools import partial

import numpy as np
from sklearn.metrics._base import _average_binary_score
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.utils.multiclass import type_of_target


def area_under_precision_recall_gain_score(
    y_true, y_score, *, average="macro", pos_label=1, sample_weight=None
):
    """Compute average precision (AP) from prediction scores.

    AP summarizes a precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous
    threshold used as the weight:

    .. math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n

    where :math:`P_n` and :math:`R_n` are the precision and recall at the nth
    threshold [1]_. This implementation is not interpolated and is different
    from computing the area under the precision-recall curve with the
    trapezoidal rule, which uses linear interpolation and can be too
    optimistic.

    Note: this implementation is restricted to the binary classification task
    or multilabel classification task.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,) or (n_samples, n_classes)
        True binary labels or binary label indicators.

    y_score : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by :term:`decision_function` on some classifiers).

    average : {'micro', 'samples', 'weighted', 'macro'} or None, \
            default='macro'
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    pos_label : int or str, default=1
        The label of the positive class. Only applied to binary ``y_true``.
        For multilabel-indicator ``y_true``, ``pos_label`` is fixed to 1.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    average_precision : float

    See Also
    --------
    roc_auc_score : Compute the area under the ROC curve.
    precision_recall_curve : Compute precision-recall pairs for different
        probability thresholds.

    Notes
    -----
    .. versionchanged:: 0.19
      Instead of linearly interpolating between operating points, precisions
      are weighted by the change in recall since the last operating point.

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
           <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
           oldid=793358396#Average_precision>`_

    Examples
    --------
    >>> import numpy as np
    >>> from precision_recall_gain import average_precision_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> average_precision_score(y_true, y_scores)
    0.83...
    """

    def _binary_uninterpolated_average_precision(
        y_true, y_score, pos_label=1, sample_weight=None
    ):
        precision_gain, recall_gain = precision_recall_gain_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
        )
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        # TODO compute integral correct?
        return -np.sum(np.diff(recall_gain) * np.array(precision_gain)[:-1])

    y_type = type_of_target(y_true)
    if y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError(
            "Parameter pos_label is fixed to 1 for "
            "multilabel-indicator y_true. Do not set "
            "pos_label or set pos_label to 1."
        )
    elif y_type == "binary":
        # Convert to Python primitive type to avoid NumPy type / Python str
        # comparison. See https://github.com/numpy/numpy/issues/6784
        present_labels = np.unique(y_true).tolist()
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError(
                f"pos_label={pos_label} is not a valid label. It should be "
                f"one of {present_labels}"
            )
    average_precision = partial(
        _binary_uninterpolated_average_precision, pos_label=pos_label
    )
    # Average a binary metric for multilabel classification.
    average_precision = _average_binary_score(
        average_precision, y_true, y_score, average, sample_weight=sample_weight
    )
    return average_precision


def precision_recall_gain(precisions, recalls, proportion_of_positives):
    """
    Converts precision and recall into precision-gain and recall-gain.


    Parameters
    ----------
    proportion_of_positives: float. Proportion of positives. Termed π in the paper.
    precisions : ndarray
    recalls: ndarray
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        prec_gain = (precisions - proportion_of_positives) / (
            (1 - proportion_of_positives) * precisions
        )
        rec_gain = (recalls - proportion_of_positives) / (
            (1 - proportion_of_positives) * recalls
        )

    return prec_gain, rec_gain


def precision_recall_gain_curve(y_true, probas_pred, pos_label=1, sample_weight=None):
    """Compute precision-recall pairs for different probability thresholds.

    Note: this implementation is restricted to the binary classification task.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold. This ensures that the graph starts on the
    y axis.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    probas_pred : ndarray of shape (n_samples,)
        Estimated probabilities or output of a decision function.

    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    precision : ndarray of shape (n_thresholds + 1,)
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.

    recall : ndarray of shape (n_thresholds + 1,)
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.

    thresholds : ndarray of shape (n_thresholds,)
        Increasing thresholds on the decision function used to compute
        precision and recall. n_thresholds <= len(np.unique(probas_pred)).

    See Also
    --------
    plot_precision_recall_curve : Plot Precision Recall Curve for binary
        classifiers.
    PrecisionRecallDisplay : Precision Recall visualization.
    average_precision_score : Compute average precision from prediction scores.
    det_curve: Compute error rates for different probability thresholds.
    roc_curve : Compute Receiver operating characteristic (ROC) curve.

    Examples
    --------
    >>> import numpy as np
    >>> from precision_recall_gain import precision_recall_curve
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> precision, recall, thresholds = precision_recall_curve(
    ...     y_true, y_scores)
    >>> precision
    array([0.66666667, 0.5       , 1.        , 1.        ])
    >>> recall
    array([1. , 0.5, 0.5, 0. ])
    >>> thresholds
    array([0.35, 0.4 , 0.8 ])

    """
    if pos_label != 1:
        raise NotImplementedError("Have not implemented non-binary targets")
    if sample_weight is not None:
        raise NotImplementedError

    # calc true and false poitives per binary classification thresh
    fps, tps, thresholds = _binary_clf_curve(
        y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight
    )

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # equivalent to slice [last_ind:None:-1]
    precision, recall, thresholds = (
        np.r_[precision[sl], 1],
        np.r_[recall[sl], 0],
        thresholds[sl],
    )

    # everything above is taken from sklearn.metrics._ranking.precision_recall_curve

    # logic taken from sklearn.metrics._ranking.det_curve
    # fns = tps[-1] - tps
    p_count = tps[-1]
    n_count = fps[-1]
    proportion_of_positives = p_count / n_count

    precision_gains, recall_gains = precision_recall_gain(
        precisions=precision,
        recalls=recall,
        proportion_of_positives=proportion_of_positives,
    )

    return precision_gains, recall_gains


"""
Source:
https://github.com/meeliskull/prg/blob/master/Python_package/prg/prg.py
"""


def precision(tp, fn, fp, tn):
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / (tp + fp)


def recall(tp, fn, fp, tn):
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / (tp + fn)


def precision_gain(tp, fn, fp, tn):
    """Calculates Precision Gain from the contingency table

    This function calculates Precision Gain from the entries of the contingency
    table: number of true positives (TP), false negatives (FN), false positives
    (FP), and true negatives (TN). More information on Precision-Recall-Gain
    curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.
    """
    n_pos = tp + fn
    n_neg = fp + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_gain = 1.0 - (n_pos / n_neg) * (fp / tp)
    if np.alen(prec_gain) > 1:
        prec_gain[tn + fn == 0] = 0
    elif tn + fn == 0:
        prec_gain = 0
    return prec_gain


def recall_gain(tp, fn, fp, tn):
    """Calculates Recall Gain from the contingency table

    This function calculates Recall Gain from the entries of the contingency
    table: number of true positives (TP), false negatives (FN), false positives
    (FP), and true negatives (TN). More information on Precision-Recall-Gain
    curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.

    Args:
        tp (float) or ([float]): True Positives
        fn (float) or ([float]): False Negatives
        fp (float) or ([float]): False Positives
        tn (float) or ([float]): True Negatives
    Returns:
        (float) or ([float])
    """
    n_pos = tp + fn
    n_neg = fp + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        rg = 1.0 - (n_pos / n_neg) * (fn / tp)
    if np.alen(rg) > 1:
        rg[tn + fn == 0] = 1
    elif tn + fn == 0:
        rg = 1
    return rg
