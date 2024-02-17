"""
https://github.com/scikit-learn/scikit-learn/pull/24121
"""

# ruff: noqa: E501
import numpy as np
from sklearn.metrics._classification import (
    _check_set_wise_labels,
    _check_zero_division,
    _prf_divide,
    _warn_prf,
    multilabel_confusion_matrix,
)
from sklearn.utils.multiclass import unique_labels


def _precision_recall_fscore_support(
    y_true,
    y_pred,
    *,
    beta=1.0,
    labels=None,
    pos_label=1,
    average=None,
    warn_for=("precision", "recall", "f-score"),
    sample_weight=None,
    zero_division="warn",
    return_in_gain_space=False,
    class_distribution=None,
):
    """Compute precision, recall, F-measure and support for each class.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label a negative sample as
    positive.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.

    The support is the number of occurrences of each class in ``y_true``.

    If ``pos_label is None`` and in binary classification, this function
    returns the average precision, recall and F-measure if ``average``
    is one of ``'micro'``, ``'macro'``, ``'weighted'`` or ``'samples'``.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    beta : float, default=1.0
        The strength of recall versus precision in the F-score.

    labels : array-like, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : {'binary', 'micro', 'macro', 'samples', 'weighted'}, \
            default=None
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division:
           - recall: when there are no positive labels
           - precision: when there are no positive predictions
           - f-score: both

        If set to "warn", this acts as 0, but warnings are also raised.

    return_in_gain_space : bool, default=False
        If True, Precision Gain, Recall Gain and FScore Gain are returned.

    class_distribution : Optional list, default=None
        The proportion that each class makes up in the dataset. It's used only
        when return_in_gain_space=True. If not provided then it's estimated from
        y_true.

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Precision score.

    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Recall score.

    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        F-beta score.

    support : None (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``y_true``.

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined.
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score,
    and ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.

    References
    ----------
    .. [1] `Wikipedia entry for the Precision and recall
           <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

    .. [2] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_.

    .. [3] `Discriminative Methods for Multi-labeled Classification Advances
           in Knowledge Discovery and Data Mining (2004), pp. 22-30 by Shantanu
           Godbole, Sunita Sarawagi
           <http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_recall_fscore_support
    >>> y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    >>> y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
    >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
    (0.22..., 0.33..., 0.26..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
    (0.33..., 0.33..., 0.33..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
    (0.22..., 0.33..., 0.26..., None)

    It is possible to compute per-label precisions, recalls, F1-scores and
    supports instead of averaging:

    >>> precision_recall_fscore_support(y_true, y_pred, average=None,
    ... labels=['pig', 'dog', 'cat'])
    (array([0.        , 0.        , 0.66...]),
     array([0., 0., 1.]), array([0. , 0. , 0.8]),
     array([2, 2, 2]))
    """
    _check_zero_division(zero_division)
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)
    class_distribution = _check_valid_class_distribution(
        class_distribution, y_true, y_pred, average, pos_label
    )

    # Calculate tp_sum, pred_sum, true_sum ###
    samplewise = average == "samples"
    MCM = multilabel_confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=labels,
        samplewise=samplewise,
    )
    tp_sum = MCM[:, 1, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta**2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(
        tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division
    )
    recall = _prf_divide(
        tp_sum, true_sum, "recall", "true", average, warn_for, zero_division
    )

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == "warn" and ("f-score",) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(average, "true nor predicted", "F-score is", len(true_sum))

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.0] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    if return_in_gain_space:
        for class_index, (
            precision_i,
            recall_i,
            f_score_i,
            true_sum_i,
            cm_i,
        ) in enumerate(zip(precision, recall, f_score, true_sum, MCM)):
            class_proportion = (
                (true_sum_i / cm_i.sum())
                if class_distribution is None
                else class_distribution[class_index]
            )
            precision[class_index] = prg_gain_transform(
                precision_i, pi=class_proportion
            )
            recall[class_index] = prg_gain_transform(recall_i, pi=class_proportion)
            f_score[class_index] = prg_gain_transform(f_score_i, pi=class_proportion)

    # Average the results
    if average == "weighted":
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = np.float64(1.0)
            if zero_division in ["warn", 0]:
                zero_division_value = np.float64(0.0)
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            if pred_sum.sum() == 0:
                return (
                    zero_division_value,
                    zero_division_value,
                    zero_division_value,
                    None,
                )
            else:
                return (np.float64(0.0), zero_division_value, np.float64(0.0), None)

    elif average == "samples":
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != "binary" or len(precision) == 1
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = None  # return no support

    return precision, recall, f_score, true_sum


def _check_valid_class_distribution(
    class_distribution, y_true, y_pred, average, pos_label
):
    if class_distribution:
        classes = unique_labels(y_true, y_pred).tolist()
        num_classes = len(classes)
        if len(class_distribution) != num_classes:
            raise ValueError(
                "Class distribution must have the same length as the number of classes"
                f" - {num_classes}."
            )
        if sum(class_distribution) != 1:
            raise ValueError("Class distribution values do not sum to 1.")

        if average == "binary":
            class_distribution = [class_distribution[classes.index(pos_label)]]

    return class_distribution


def f1_gain_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
    class_distribution=None,
):
    """Compute the F1 Gain score, also known as balanced F-Gain score or
    F-Gain measure.

    The F1 Gain score can be interpreted as a arithmetic mean of the precision
    gain and recall gain, where an F1 Gain score reaches its best value at 1 and
    worst score at -Inf. The relative contribution of precision and recall to
    the F1 score are equal. The formula for the F1 score is::

        F1_Gain = (precision_gain + recall_gain) / 2

    In the multi-class and multi-label case, this is the average of the F1 Gain
    score of each class with weighting depending on the ``average`` parameter.
    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : array-like, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

        .. versionchanged:: 0.17
           Parameter `labels` improved for multiclass problem.

    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : {'macro', 'weighted', 'binary'} or None, \
            default='binary'
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative. If set to "warn", this acts as 0,
        but warnings are also raised.

    class_distribution : Optional list, default=None
        The proportion that each class makes up in the dataset. If not
        provided then it's estimated from y_true.

    Returns
    -------
    f1_gain_score : float or array of float, shape = [n_unique_labels]
        F1 Gain score of the positive class in binary classification or weighted
        average of the F1 Gain scores of each class for the multiclass task.

    See Also
    --------
    fbeta_gain_score : Compute the F-Gain beta score.
    precision_recall_fgain_score_support : Compute the precision gain, recall
        gain, F-Gain score, and support.
    jaccard_score : Compute the Jaccard similarity coefficient score.
    multilabel_confusion_matrix : Compute a confusion matrix for each class or
        sample.

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined.
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score,
    and ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.

    References
    ----------
    .. [1] `Precision-Recall-Gain Curves: PR Analysis Done Right (2015) by
            Peter A. Flach and Meelis Kull
           <https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf>`_.
    .. [2] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_.

    Examples
    --------
    >>> from precision_recall_gain import f1_gain_score
    >>> y_true = [0, 1, 2, 0, 1, 2, 2]
    >>> y_pred = [0, 2, 1, 0, 1, 1, 2]
    >>> f1_gain_score(y_true, y_pred, average='macro')
    0.42...
    >>> f1_gain_score(y_true, y_pred, average='weighted')
    0.34...
    >>> f1_gain_score(y_true, y_pred, average=None)
    array([ 1.   ,  0.4  , -0.125])
    >>> y_true = [0, 0, 0, 0, 0, 0]
    >>> y_pred = [0, 0, 0, 0, 0, 0]
    >>> f1_gain_score(y_true, y_pred, zero_division=1)
    1.0
    >>> # multilabel classification
    >>> y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
    >>> y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]
    >>> f1_gain_score(y_true, y_pred, average=None)
    array([0.75, 1.  , 0.  ])
    """
    return fbeta_gain_score(
        y_true,
        y_pred,
        beta=1,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
        class_distribution=class_distribution,
    )


def fbeta_gain_score(
    y_true,
    y_pred,
    *,
    beta,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
    class_distribution=None,
):
    """Compute the F-Gain beta score.

    The F-Gain beta score is the weighted arthimetic mean of precision gain
    and recall gain, reaching its optimal value at 1 and its worst value at
    -Inf.

    The `beta` parameter determines the weight of recall gain in the combined
    score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
    favors recall (``beta -> 0`` considers only precision, ``beta -> +inf``
    only recall).

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    beta : float
        Determines the weight of recall in the combined score.

    labels : array-like, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

        .. versionchanged:: 0.17
           Parameter `labels` improved for multiclass problem.

    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : {'macro', 'weighted', 'binary'} or None, \
            default='binary'
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative. If set to "warn", this acts as 0,
        but warnings are also raised.

    class_distribution : Optional list, default=None
        The proportion that each class makes up in the dataset. If not
        provided then it's estimated from y_true.

    Returns
    -------
    fgain_beta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        F-Gain beta score of the positive class in binary classification or weighted
        average of the F-Gain beta score of each class for the multiclass task.

    See Also
    --------
    precision_recall_fgain_score_support : Compute the precision gain, recall
        gain, F-Gain score, and support.
    multilabel_confusion_matrix : Compute a confusion matrix for each class or
        sample.

    Notes
    -----
    When ``true positive + false positive == 0`` or
    ``true positive + false negative == 0``, f-score returns 0 and raises
    ``UndefinedMetricWarning``. This behavior can be
    modified with ``zero_division``.

    References
    ----------
    .. [1] `Precision-Recall-Gain Curves: PR Analysis Done Right (2015) by
            Peter A. Flach and Meelis Kull
           <https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf>`_.
    .. [2] R. Baeza-Yates and B. Ribeiro-Neto (2011).
           Modern Information Retrieval. Addison Wesley, pp. 327-328.

    .. [3] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_.

    Examples
    --------
    >>> from precision_recall_gain import fbeta_gain_score
    >>> y_true = [0, 1, 2, 0, 1, 2, 2]
    >>> y_pred = [0, 2, 1, 0, 1, 1, 2]
    >>> fbeta_gain_score(y_true, y_pred, average='macro', beta=0.5)
    0.45...
    >>> fbeta_gain_score(y_true, y_pred, average='weighted', beta=0.5)
    0.40...
    >>> fbeta_gain_score(y_true, y_pred, average=None, beta=0.5)
    array([1.  , 0.28, 0.1 ])
    """

    _, _, f, _ = precision_recall_fgain_score_support(
        y_true,
        y_pred,
        beta=beta,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("f-score",),
        sample_weight=sample_weight,
        zero_division=zero_division,
        class_distribution=class_distribution,
    )
    return f


def precision_recall_fgain_score_support(
    y_true,
    y_pred,
    *,
    class_distribution=None,
    beta=1.0,
    labels=None,
    pos_label=1,
    average=None,
    warn_for=("precision", "recall", "f-score"),
    sample_weight=None,
    zero_division="warn",
):
    """Compute precision gain, recall gain, F-Gain measure and support for each
    class.

    All three measures are derrived by applying the following transform to their
    respective vanilla metric values.

        f(x) = (x - pi) / ((1 - pi) * x)

            pi = proportion of positives

    The vanilla metrics prior to transformation are defined as follows:

        The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number
        of true positives and ``fp`` the number of false positives. The
        precision is intuitively the ability of the classifier not to label a
        negative sample as positive.

        The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
        true positives and ``fn`` the number of false negatives. The recall is
        intuitively the ability of the classifier to find all the positive
        samples.

        The F-beta score can be interpreted as a weighted harmonic mean of the
        precision and recall, where an F-beta score reaches its best value at 1
        and worst score at 0.

        The F-beta score weights recall more than precision by a factor of
        ``beta``. ``beta == 1.0`` means recall and precision are equally
        important.

    The support is the number of occurrences of each class in ``y_true``.

    If ``pos_label is None`` and in binary classification, this function returns
    the average precision gain, recall gain and F-gain measure if ``average`` is
    one of ``'micro'``, ``'macro'``, ``'weighted'`` or ``'samples'``.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    beta : float, default=1.0
        The strength of recall versus precision in the F-score.

    labels : array-like, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : {'binary', 'macro', 'weighted'}, \
            default=None
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division:
           - recall: when there are no positive labels
           - precision: when there are no positive predictions
           - f-score: both

        If set to "warn", this acts as 0, but warnings are also raised.

    class_distribution : Optional list, default=None
        The proportion that each class makes up in the dataset. If not
        provided then it's estimated from y_true.

    Returns
    -------
    precision_gain : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Precision Gain score.

    recall_gain : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Recall Gain score.

    f_gain_beta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        F-beta Gain score.

    support : None (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``y_true``.

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined.
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score,
    and ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.

    References
    ----------
    .. [1] `Precision-Recall-Gain Curves: PR Analysis Done Right (2015) by Peter
            A. Flach and Meelis Kull
           <https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf>`_.
    .. [2] `Wikipedia entry for the Precision and recall
           <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

    .. [3] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_.

    .. [4] `Discriminative Methods for Multi-labeled Classification Advances in
           Knowledge Discovery and Data Mining (2004), pp. 22-30 by Shantanu
           Godbole, Sunita Sarawagi
           <http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from precision_recall_gain import precision_recall_fgain_score_support
    >>> y_true = np.array(['cat', 'dog', 'pig', 'dog', 'cat', 'pig', 'pig'])
    >>> y_pred = np.array(['cat', 'pig', 'dog', 'dog', 'cat', 'dog', 'pig'])

    It is possible to compute per-label precisions, recalls, F1-scores and
    supports instead of averaging:

    >>> precision_recall_fgain_score_support(y_true, y_pred, average=None,
    ... labels=['pig', 'dog', 'cat'])
    (array([0.25, 0.2 , 1.  ]), array([-0.5,  0.6,  1. ]), array([-0.125,  0.4  ,  1.   ]), array([3, 2, 2]))
    """
    average_options = (None, "binary", "macro", "weighted")
    if average not in average_options:
        raise ValueError("average has to be one of " + str(average_options))

    return _precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        beta=beta,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=warn_for,
        sample_weight=sample_weight,
        zero_division=zero_division,
        return_in_gain_space=True,
        class_distribution=class_distribution,
    )


def precision_gain_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
    class_distribution=None,
):
    """Compute the precision Gain.

    The metric is derrived by applying the following transform to precision:

        f(x) = (x - pi) / ((1 - pi) * x)

            pi = proportion of positives

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is -Inf.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : array-like, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

        .. versionchanged:: 0.17
           Parameter `labels` improved for multiclass problem.

    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : {'macro', 'weighted', 'binary'} or None, \
            default='binary'
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    class_distribution : Optional list, default=None
        The proportion that each class makes up in the dataset. If not
        provided then it's estimated from y_true.

    Returns
    -------
    precision_gain : float (if average is not None) or array of float of shape \
                (n_unique_labels,)
        Precision of the positive class in binary classification or weighted
        average of the precision of each class for the multiclass task.

    See Also
    --------
    precision_recall_fgain_score_support : Compute precision, recall, F-measure and
        support for each class.
    recall_gain_score :  Compute the ratio ``tp / (tp + fn)`` where ``tp`` is the
        number of true positives and ``fn`` the number of false negatives.
    PrecisionRecallDisplay.from_estimator : Plot precision-recall curve given
        an estimator and some data.
    PrecisionRecallDisplay.from_predictions : Plot precision-recall curve given
        binary class predictions.
    multilabel_confusion_matrix : Compute a confusion matrix for each class or
        sample.

    Notes
    -----
    When ``true positive + false positive == 0``, precision returns 0 and
    raises ``UndefinedMetricWarning``. This behavior can be
    modified with ``zero_division``.

    References
    ----------
    .. [1] `Precision-Recall-Gain Curves: PR Analysis Done Right (2015) by Peter
            A. Flach and Meelis Kull
           <https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf>`_.

    Examples
    --------
    >>> from precision_recall_gain import precision_gain_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> int(precision_gain_score(y_true, y_pred, average='macro'))
    -333333333333333
    >>> int(precision_gain_score(y_true, y_pred, average='weighted'))
    -333333333333333
    >>> precision_gain_score(y_true, y_pred, average=None)
    array([ 7.5e-01, -5.0e+14, -5.0e+14])
    >>> y_pred = [0, 0, 0, 0, 0, 0]
    >>> precision_gain_score(y_true, y_pred, average=None)
    array([ 0.e+00, -5.e+14, -5.e+14])
    >>> precision_gain_score(y_true, y_pred, average=None, zero_division=1)
    array([0., 1., 1.])
    >>> # multilabel classification
    >>> y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
    >>> y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]
    >>> precision_gain_score(y_true, y_pred, average=None)
    array([0.5, 1. , 1. ])
    """
    p, _, _, _ = precision_recall_fgain_score_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("precision",),
        sample_weight=sample_weight,
        zero_division=zero_division,
        class_distribution=class_distribution,
    )
    return p


def recall_gain_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
    class_distribution=None,
):
    """Compute the recall Gain.

    The metric is derrived by applying the following transform to precision:

        f(x) = (x - pi) / ((1 - pi) * x)

            pi = proportion of positives

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is -Inf.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : array-like, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

        .. versionchanged:: 0.17
           Parameter `labels` improved for multiclass problem.

    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : {'macro', 'weighted', 'binary'} or None, \
            default='binary'
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall. Weighted recall
            is equal to accuracy.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    class_distribution : Optional list, default=None
        The proportion that each class makes up in the dataset. If not
        provided then it's estimated from y_true.

    Returns
    -------
    recall : float (if average is not None) or array of float of shape \
             (n_unique_labels,)
        Recall of the positive class in binary classification or weighted
        average of the recall of each class for the multiclass task.

    See Also
    --------
    precision_recall_fgain_score_support : Compute precision, recall, F-measure and
        support for each class.
    precision_gain_score : Compute the ratio ``tp / (tp + fp)`` where ``tp`` is the
        number of true positives and ``fp`` the number of false positives.
    balanced_accuracy_score : Compute balanced accuracy to deal with imbalanced
        datasets.
    multilabel_confusion_matrix : Compute a confusion matrix for each class or
        sample.
    PrecisionRecallDisplay.from_estimator : Plot precision-recall curve given
        an estimator and some data.
    PrecisionRecallDisplay.from_predictions : Plot precision-recall curve given
        binary class predictions.

    Notes
    -----
    When ``true positive + false negative == 0``, recall returns 0 and raises
    ``UndefinedMetricWarning``. This behavior can be modified with
    ``zero_division``.

    References
    ----------
    .. [1] `Precision-Recall-Gain Curves: PR Analysis Done Right (2015) by Peter
            A. Flach and Meelis Kull
           <https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf>`_.

    Examples
    --------
    >>> from precision_recall_gain import recall_gain_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> int(recall_gain_score(y_true, y_pred, average='macro'))
    -333333333333333
    >>> int(recall_gain_score(y_true, y_pred, average='weighted'))
    -333333333333333
    >>> recall_gain_score(y_true, y_pred, average=None)
    array([ 1.e+00, -5.e+14, -5.e+14])
    >>> y_true = [0, 0, 0, 0, 0, 0]
    >>> recall_gain_score(y_true, y_pred, average=None)
    array([-inf,  nan,  nan])
    >>> recall_gain_score(y_true, y_pred, average=None, zero_division=1)
    array([-inf,   1.,   1.])
    >>> # multilabel classification
    >>> y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
    >>> y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]
    >>> recall_gain_score(y_true, y_pred, average=None)
    array([ 1.,  1., -1.])
    """
    _, r, _, _ = precision_recall_fgain_score_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("recall",),
        sample_weight=sample_weight,
        zero_division=zero_division,
        class_distribution=class_distribution,
    )
    return r


def prg_gain_transform(x, *, pi):
    """Transfrom from Precision Recall space into Precision Recall Gain space.

    Parameters
    ----------
    x : scaler or 1d array-like
        The metric, either precision, recall or F-score to be transformed into
        PRG space.
    pi : scaler
        The proportion of datapoints belonging to the positive class in the
        dataset.

    Returns
    -------
    x' : scaler or 1d array-like
        The transformed metric in PRG space.

    References
    ----------
    .. [1] `Precision-Recall-Gain Curves: PR Analysis Done Right (2015) by Peter
            A. Flach and Meelis Kull
           <https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf>`_.
    """
    if x == pi == 1:
        return 1
    elif x == pi == 0:
        # if no positive class in true or predicted labels, return NaN
        return np.nan
    # note: if x == 0, then the metric value is -Inf
    # and if x<pi, then the metric value is negative
    # for our purposes we will add a small value to x
    # to avoid division by zero and so that the metric
    # value is not nan if one of the classes have a precision
    # or recall of 0
    x = min(1, x + 1e-15)
    # we have to also adjust pi for cases when pi is 0
    pi = min(1, pi + 1e-15)
    return (x - pi) / ((1 - pi) * x)
