import numpy as np
import pytest
from sklearn import datasets, svm
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_no_warnings,
    ignore_warnings,
)
from sklearn.utils.validation import check_random_state

from precision_recall_gain import (
    f1_gain_score,
    fbeta_gain_score,
    precision_gain_score,
    precision_recall_fgain_score_support,
    recall_gain_score,
)

###############################################################################
# Utilities for testing


def make_prediction(dataset=None, binary=False):
    """Make some classification predictions on a toy dataset using a SVC

    If binary is True restrict to a binary classification problem instead of a
    multiclass classification problem
    """

    if dataset is None:
        # import some data to play with
        dataset = datasets.load_iris()

    X = dataset.data
    y = dataset.target

    if binary:
        # restrict to a binary classification task
        X, y = X[y < 2], y[y < 2]

    n_samples, n_features = X.shape
    p = np.arange(n_samples)

    rng = check_random_state(37)
    rng.shuffle(p)
    X, y = X[p], y[p]
    half = int(n_samples / 2)

    # add noisy features to make the problem harder and avoid perfect results
    rng = np.random.RandomState(0)
    X = np.c_[X, rng.randn(n_samples, 200 * n_features)]

    # run classifier, get class probabilities and label predictions
    clf = svm.SVC(kernel="linear", probability=True, random_state=0)
    probas_pred = clf.fit(X[:half], y[:half]).predict_proba(X[half:])

    if binary:
        # only interested in probabilities of the positive case
        # XXX: do we really want a special API for the binary case?
        probas_pred = probas_pred[:, 1]

    y_pred = clf.predict(X[half:])
    y_true = y[half:]
    return y_true, y_pred, probas_pred


###############################################################################
# Tests


def test_precision_recall_f1_gain_score_averages():
    # Test Precision Recall and F1 Score for binary classification task
    y_true, y_pred, _ = make_prediction(binary=True)

    # binary average
    p, r, f, s = precision_recall_fgain_score_support(y_true, y_pred, average="binary")
    assert_array_almost_equal(p, 0.82, 2)
    assert_array_almost_equal(r, 0.53, 2)
    assert_array_almost_equal(f, 0.68, 2)

    # macro average
    p, r, f, s = precision_recall_fgain_score_support(y_true, y_pred, average="macro")
    assert_array_almost_equal(p, 0.73, 2)
    assert_array_almost_equal(r, 0.70, 2)
    assert_array_almost_equal(f, 0.72, 2)

    # Test Precision Recall and F1 Score for multi classification task
    y_true, y_pred, _ = make_prediction(binary=False)

    # weighted average
    p, r, f, s = precision_recall_fgain_score_support(
        y_true, y_pred, average="weighted"
    )
    assert_array_almost_equal(p, 0.25, 2)
    assert_array_almost_equal(r, -1.77, 2)
    assert_array_almost_equal(f, -0.76, 2)


def test_precision_recall_f1_gain_score_class_dist():
    # Test Precision Recall and F1 Score for binary classification task
    y_true, y_pred, _ = make_prediction(binary=True)

    # binary average
    p, r, f, s = precision_recall_fgain_score_support(
        y_true, y_pred, average="binary", class_distribution=[0.4, 0.6]
    )
    assert_array_almost_equal(p, 0.74, 2)
    assert_array_almost_equal(r, 0.29, 2)
    assert_array_almost_equal(f, 0.51, 2)

    # macro average
    p, r, f, s = precision_recall_fgain_score_support(
        y_true, y_pred, average="macro", class_distribution=[0.4, 0.6]
    )
    assert_array_almost_equal(p, 0.75, 2)
    assert_array_almost_equal(r, 0.60, 2)
    assert_array_almost_equal(f, 0.67, 2)

    # Test Precision Recall and F1 Score for multi classification task
    y_true, y_pred, _ = make_prediction(binary=False)

    # weighted average
    p, r, f, s = precision_recall_fgain_score_support(
        y_true, y_pred, average="weighted", class_distribution=[0.4, 0.2, 0.4]
    )
    assert_array_almost_equal(p, 0.50, 2)
    assert_array_almost_equal(r, -0.04, 2)
    assert_array_almost_equal(f, 0.23, 2)


def test_precision_recall_f1_gain_score_binary():
    # Test Precision Recall and F1 Score for binary classification task
    y_true, y_pred, _ = make_prediction(binary=True)

    # detailed measures for each class
    p, r, f, s = precision_recall_fgain_score_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.64, 0.82], 2)
    assert_array_almost_equal(r, [0.86, 0.53], 2)
    assert_array_almost_equal(f, [0.75, 0.68], 2)
    assert_array_equal(s, [25, 25])

    # individual scoring function that can be used for grid search: in the
    # binary class case the score is the value of the measure for the positive
    # class (e.g. label == 1). This is deprecated for average != 'binary'.
    for kwargs, my_assert in [
        ({}, assert_no_warnings),
        ({"average": "binary"}, assert_no_warnings),
    ]:
        ps = my_assert(precision_gain_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(ps, 0.82, 2)

        rs = my_assert(recall_gain_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(rs, 0.53, 2)

        fs = my_assert(f1_gain_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(fs, 0.68, 2)

        beta = 2
        assert_almost_equal(
            my_assert(fbeta_gain_score, y_true, y_pred, beta=beta, **kwargs),
            (ps + ((beta**2) * rs)) / (1 + (beta**2)),
            2,
        )


@ignore_warnings
def test_precision_recall_f_gain_binary_single_class():
    # Test precision, recall and F-scores behave with a single positive or
    # negative class. Such a case may occur with non-stratified cross-validation
    assert 1.0 == precision_gain_score([1, 1], [1, 1])
    assert 1.0 == recall_gain_score([1, 1], [1, 1])
    assert 1.0 == f1_gain_score([1, 1], [1, 1])
    assert 1.0 == fbeta_gain_score([1, 1], [1, 1], beta=0)
    assert 1.0 == f1_gain_score([2, 2], [2, 2], pos_label=2)

    # test case when no positive class present in true or predicted labels
    assert np.isnan(precision_gain_score([2, 2], [2, 2]))
    assert np.isnan(precision_gain_score([-1, -1], [-1, -1]))
    assert np.isnan(recall_gain_score([-1, -1], [-1, -1]))
    assert np.isnan(f1_gain_score([-1, -1], [-1, -1]))
    assert np.isnan(fbeta_gain_score([-1, -1], [-1, -1], beta=float("inf")))
    assert np.isnan(fbeta_gain_score([-1, -1], [-1, -1], beta=1e5))

    # test case when true labels all positive
    assert precision_gain_score([1, 1], [1, 0]) == 1
    assert precision_gain_score([1, 1], [0, 1]) == 1
    assert recall_gain_score([1, 1], [1, 0]) == -np.inf
    assert recall_gain_score([1, 1], [0, 1]) == -np.inf
    assert f1_gain_score([1, 1], [1, 0]) == -np.inf
    assert f1_gain_score([1, 1], [0, 1]) == -np.inf

    # test case when predicted labels all positive
    assert precision_gain_score([1, 0], [1, 1]) == 0
    assert precision_gain_score([0, 1], [1, 1]) == 0
    assert recall_gain_score([1, 0], [1, 1]) == 1
    assert recall_gain_score([0, 1], [1, 1]) == 1
    assert_array_almost_equal(f1_gain_score([1, 0], [1, 1]), 0.5)
    assert_array_almost_equal(f1_gain_score([0, 1], [1, 1]), 0.5)


@ignore_warnings
def test_precision_recall_fgain_score_support_errors():
    y_true, y_pred, _ = make_prediction(binary=True)

    # Bad beta
    with pytest.raises(ValueError):
        precision_recall_fgain_score_support(y_true, y_pred, beta=-0.1)

    # Bad pos_label
    with pytest.raises(ValueError):
        precision_recall_fgain_score_support(
            y_true, y_pred, pos_label=2, average="binary"
        )

    # Bad average option 1
    with pytest.raises(ValueError):
        precision_recall_fgain_score_support([0, 1, 2], [1, 2, 0], average="mega")

    # Bad average option 2
    with pytest.raises(ValueError):
        precision_recall_fgain_score_support([0, 1, 2], [1, 2, 0], average="micro")

    # Bad class_distribution dimension
    with pytest.raises(ValueError):
        precision_recall_fgain_score_support(
            [0, 1, 2], [1, 2, 0], class_distribution=[3]
        )

    # Bad class_distribution values
    with pytest.raises(ValueError):
        precision_recall_fgain_score_support(
            [0, 1, 2], [1, 2, 0], class_distribution=[0.4, 0.6, 0.1]
        )


def test_precision_recall_f1_gain_score_multiclass():
    # Test Precision Recall and F1 Score for multiclass classification task
    y_true, y_pred, _ = make_prediction(binary=False)

    # compute scores with default labels introspection
    p, r, f, s = precision_recall_fgain_score_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.9, -0.41, 0.49], 2)
    assert_array_almost_equal(r, [0.88, -5.58, 0.96], 2)
    assert_array_almost_equal(f, [0.89, -2.99, 0.73], 2)
    assert_array_equal(s, [24, 31, 20])

    # averaging tests
    ps = precision_gain_score(y_true, y_pred, average="macro")
    assert_array_almost_equal(ps, 0.33, 2)

    rs = recall_gain_score(y_true, y_pred, average="macro")
    assert_array_almost_equal(rs, -1.25, 2)

    fs = f1_gain_score(y_true, y_pred, average="macro")
    assert_array_almost_equal(fs, -0.46, 2)

    ps = precision_gain_score(y_true, y_pred, average="weighted")
    assert_array_almost_equal(ps, 0.25, 2)

    rs = recall_gain_score(y_true, y_pred, average="weighted")
    assert_array_almost_equal(rs, -1.77, 2)

    fs = f1_gain_score(y_true, y_pred, average="weighted")
    assert_array_almost_equal(fs, -0.76, 2)

    with pytest.raises(ValueError):
        precision_gain_score(y_true, y_pred, average="samples")
    with pytest.raises(ValueError):
        recall_gain_score(y_true, y_pred, average="samples")
    with pytest.raises(ValueError):
        f1_gain_score(y_true, y_pred, average="samples")
    with pytest.raises(ValueError):
        fbeta_gain_score(y_true, y_pred, average="samples", beta=0.5)

    # same prediction but with and explicit label ordering
    p, r, f, s = precision_recall_fgain_score_support(
        y_true, y_pred, labels=[0, 2, 1], average=None
    )
    assert_array_almost_equal(p, [0.9, 0.49, -0.41], 2)
    assert_array_almost_equal(r, [0.88, 0.96, -5.58], 2)
    assert_array_almost_equal(f, [0.89, 0.73, -2.99], 2)
    assert_array_equal(s, [24, 20, 31])


def test_precision_gain_score_docs():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    assert precision_gain_score(y_true, y_pred, average="macro") < -1e14
    assert precision_gain_score(y_true, y_pred, average="weighted") < -1e14

    result = precision_gain_score(y_true, y_pred, average=None)
    assert np.isclose(result[0], 0.75)
    assert np.all(result[1:] < -1e14)

    y_pred = [0, 0, 0, 0, 0, 0]
    with pytest.warns(UndefinedMetricWarning):
        result = precision_gain_score(y_true, y_pred, average=None)
    assert np.isclose(result[0], 0)
    assert np.all(result[1:] < -1e14)

    assert_array_almost_equal(
        precision_gain_score(y_true, y_pred, average=None, zero_division=1),
        [0.0, 1.0, 1.0],
        2,
    )

    # multilabel classification
    y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
    y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]
    # this one is correct
    assert_array_almost_equal(
        precision_gain_score(y_true, y_pred, average=None), [0.5, 1.0, 1.0], 2
    )
    assert_array_almost_equal(
        recall_gain_score(y_true, y_pred, average=None), [1.0, 1.0, -1.0], 2
    )

    # binary classification
    y_pred = [0, 0, 1, 0]
    y_true = [0, 1, 1, 0]
    result = precision_recall_fgain_score_support(y_true, y_pred, average="binary")
    assert_almost_equal(result[:3], [1, 0, 0.5])
    assert result[3] is None


def test_recall_gain_docs():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]

    assert recall_gain_score(y_true, y_pred, average="macro") < -1e14
    assert recall_gain_score(y_true, y_pred, average="weighted") < -1e14
    result = recall_gain_score(y_true, y_pred, average=None)
    assert np.isclose(result[0], 1)
    assert np.all(result[1:] < -1e14)

    y_true = [0, 0, 0, 0, 0, 0]

    with pytest.warns((UndefinedMetricWarning, RuntimeWarning)):
        result = recall_gain_score(y_true, y_pred, average=None)
    assert_array_almost_equal(result, [-np.inf, np.nan, np.nan], 2)

    with pytest.warns(RuntimeWarning):
        assert_array_almost_equal(
            recall_gain_score(y_true, y_pred, average=None, zero_division=1),
            [-np.inf, 1.0, 1.0],
            2,
        )
