from bitarray import bitarray
import heapq

import numpy as np
from numbers import Real
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier,
    _fit_context,
)
import sklearn.utils
from sklearn.utils._param_validation import Interval, StrOptions, HasMethods
from sklearn.preprocessing import MinMaxScaler
from imblearn.base import BaseSampler


class TIcEImputer(BaseSampler, MetaEstimatorMixin):
    """
    TIcEImputer is a class that performs imputation using the TIcE algorithm.

    Estimate the class prior through decision tree induction.

    Parameters:
    -----------
    estimator : BaseEstimator
        The base estimator used for imputation.
    delta : float, optional
        The delta parameter for TIcE algorithm. Default is None.
    max_bepp : int, optional
        The max-bepp parameter k. Default is 5.
    maxSplits : int, optional
        The maximum number of splits in the decision tree. Default is 500.
    useMostPromisingOnly : bool, optional
        Whether to use only the most promising subset or calculate the maximum lower bound. Default is False.
    minT : int, optional
        The minimum set size to update the lower bound. Default is 10.
    nbIts : int, optional
        The number of times to repeat the estimation process. Default is 2.

    Attributes:
    -----------
    estimator : BaseEstimator
        The base estimator used for imputation.
    delta : float, optional
        The delta parameter for TIcE algorithm.
    max_bepp : int, optional
        The max-bepp parameter k.
    maxSplits : int, optional
        The maximum number of splits in the decision tree.
    useMostPromisingOnly : bool, optional
        Whether to use only the most promising subset or calculate the maximum lower bound.
    minT : int, optional, default=10
        The minimum set size to update the lower bound with.
    nbIts : int, optional
        The number of times to repeat the estimation process. Default 2 (first with c_prior=0.5, then with c_prior=c_estimate).
    """

    _sampling_type = "clean-sampling"
    _parameter_constraints = {
        "estimator": [BaseEstimator],
    }

    def __init__(
        self,
        estimator,
        delta=None,
        max_bepp=5,
        maxSplits=500,
        useMostPromisingOnly=False,
        minT=10,
        nbIts=2,
        cv=5,
    ):
        self.estimator = estimator
        self.delta = delta
        self.max_bepp = max_bepp
        self.maxSplits = maxSplits
        self.useMostPromisingOnly = useMostPromisingOnly
        self.minT = minT
        self.nbIts = nbIts
        self.cv = cv

    def _estimate_c(self, X, y_col):
        c_estimate, c_its_estimates = _tice(
            data=X,
            labels=y_col,
            k=self.max_bepp,
            folds=np.random.randint(self.cv, size=len(X)),  # FIXME
            delta=self.delta,
            nbIterations=self.nbIts,
            maxSplits=self.maxSplits,
            useMostPromisingOnly=self.useMostPromisingOnly,
            minT=self.minT,
        )
        return c_estimate, c_its_estimates

    @_fit_context(
        # Estimator is not validated yet.
        prefer_skip_nested_validation=False,
    )
    def fit(self, X, y, **params):
        """
        Fit the imputer to the data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.
        """
        self._validate_params()
        self.estimator_ = clone(self.estimator)

        self.c_estimates_ = []
        self.c_its_estimates_ = []
        self.alpha_ = []

        for y_col in y.T:
            y_bitcol = bitarray(list(y_col.astype(bool)))
            c_estimate, c_its_estimates = self._estimate_c(X, y_bitcol)
            self.c_estimates_.append(c_estimate)
            self.c_its_estimates_.append(c_its_estimates)

            alpha = 1.0
            if c_estimate > 0:
                pos = float(y_bitcol.count()) / c_estimate
                alpha = max(0.0, min(1.0, pos / len(X)))

            self.alpha_.append(alpha)

        # self.estimator_.fit(X, y)
        return self
    
    def _fit_resample(self, X, y):
        pass


def pick_delta(T):
    return max(0.025, 1 / (1 + 0.004 * T))


def low_c(subset, label, delta, minT, c=0.5):
    T = float(subset.count())
    if T < minT:
        return 0.0
    L = float((subset & label).count())
    clow = L / T - np.sqrt(c * (1 - c) * (1 - delta) / (delta * T))
    return clow


def max_bepp(k):
    def fun(counts):
        return max(
            map(lambda TP: (0 if TP[0] == 0 else float(TP[1]) / (TP[0] + k)), counts)
        )

    return fun


def generate_folds(folds):
    for fold in range(max(folds) + 1):
        tree_train = bitarray(list(folds == fold))
        estimate = ~tree_train
        # tree_train, estimate = estimate, tree_train  # XXX
        yield (tree_train, estimate)


def estimate_label_frequency_lower_bound(
    X,
    y,
    n_folds=5,
    max_bepp=5,
    delta=None,
    n_iter=2,
    max_splits=500,
    most_promising_only=False,
    min_set_size=10,
    random_state=None,
):
    random_state = sklearn.utils.check_random_state(random_state)
    c_estimates = []
    c_its_estimates = []
    scaled_X = MinMaxScaler().fit_transform(X)

    for y_col in y.T:
        y_bit_col = bitarray(list(y_col.astype(bool)))
        c_estimate, c_its_estimate = _tice(
            data=scaled_X,
            labels=y_bit_col,
            k=max_bepp,
            folds=random_state.randint(n_folds, size=len(X)),  # FIXME
            delta=delta,
            nbIterations=n_iter,
            maxSplits=max_splits,
            useMostPromisingOnly=most_promising_only,
            minT=min_set_size,
        )
        c_estimates.append(c_estimate)
        c_its_estimates.append(c_its_estimate)

    return c_estimates, c_its_estimates


def _tice(
    data,
    labels,
    k,
    folds,
    delta=None,
    nbIterations=2,
    maxSplits=500,
    useMostPromisingOnly=False,
    minT=10,
):
    c_its_ests = []
    c_estimate = 0.5

    for it in range(nbIterations):
        c_estimates = []

        global c_cur_best  # global so that it can be used for optimizing queue.
        for tree_train, estimate in generate_folds(folds):
            c_cur_best = low_c(estimate, labels, 1.0, minT, c=c_estimate)
            cur_delta = delta if delta else pick_delta(estimate.count())

            if useMostPromisingOnly:
                c_tree_best = 0.0
                most_promising = estimate
                for tree_subset, estimate_subset in subsetsThroughDT(
                    data,
                    tree_train,
                    estimate,
                    labels,
                    splitCrit=max_bepp(k),
                    minExamples=minT,
                    maxSplits=maxSplits,
                    c_prior=c_estimate,
                    delta=cur_delta,
                ):
                    tree_est_here = low_c(
                        tree_subset, labels, cur_delta, 1, c=c_estimate
                    )
                    if tree_est_here > c_tree_best:
                        c_tree_best = tree_est_here
                        most_promising = estimate_subset

                c_estimates.append(
                    max(
                        c_cur_best,
                        low_c(most_promising, labels, cur_delta, minT, c=c_estimate),
                    )
                )

            else:
                for tree_subset, estimate_subset in subsetsThroughDT(
                    data,
                    tree_train,
                    estimate,
                    labels,
                    splitCrit=max_bepp(k),
                    minExamples=minT,
                    maxSplits=maxSplits,
                    c_prior=c_estimate,
                    delta=cur_delta,
                ):
                    est_here = low_c(
                        estimate_subset, labels, cur_delta, minT, c=c_estimate
                    )
                    c_cur_best = max(c_cur_best, est_here)
                c_estimates.append(c_cur_best)

        c_estimate = sum(c_estimates) / float(len(c_estimates))
        c_its_ests.append(c_estimates)

    return c_estimate, c_its_ests


def subsetsThroughDT(
    data,
    tree_train,
    estimate,
    labels,
    splitCrit=max_bepp(5),
    minExamples=10,
    maxSplits=500,
    c_prior=0.5,
    delta=0.0,
):
    # This learns a decision tree and updates the label frequency lower bound for every tried split.
    # It splits every variable into 4 pieces: [0,.25[ , [.25, .5[ , [.5,.75[ , [.75,1]
    # The input data is expected to have only binary or continues variables with values between 0 and 1. To achieve this, the multivalued variables should be binarized and the continuous variables should be normalized

    # Max: Return all the subsets encountered

    all_data = tree_train | estimate

    borders = [0.25, 0.5, 0.75]

    def makeSubsets(a):
        subsets = []
        options = bitarray(all_data)
        for b in borders:
            X_cond = bitarray(list((data[:, a] < b))) & options
            options &= ~X_cond
            subsets.append(X_cond)
        subsets.append(options)
        return subsets

    conditionSets = [makeSubsets(a) for a in range(data.shape[1])]

    priorityq = []
    heapq.heappush(
        priorityq,
        (
            -low_c(tree_train, labels, delta, 0, c=c_prior),
            -(tree_train & labels).count(),
            tree_train,
            estimate,
            set(range(data.shape[1])),
            0,
        ),
    )
    yield (tree_train, estimate)

    n = 0
    minimumLabeled = 1
    while n < maxSplits and len(priorityq) > 0:
        n += 1
        (
            ppos,
            neg_lab_count,
            subset_train,
            subset_estimate,
            available,
            depth,
        ) = heapq.heappop(priorityq)
        lab_count = -neg_lab_count

        best_a = -1
        best_score = -1
        best_subsets_train = []
        best_subsets_estimate = []
        best_lab_counts = []
        uselessAs = set()

        for a in available:
            subsets_train = map(lambda X_cond: X_cond & subset_train, conditionSets[a])
            subsets_estimate = map(
                lambda X_cond: X_cond & subset_train, conditionSets[a]
            )
            estimate_lab_counts = map(
                lambda subset: (subset & labels).count(), subsets_estimate
            )
            if max(estimate_lab_counts) < minimumLabeled:
                uselessAs.add(a)
            else:
                score = splitCrit(
                    map(
                        lambda subsub: (subsub.count(), (subsub & labels).count()),
                        subsets_train,
                    )
                )
                if score > best_score:
                    best_score = score
                    best_a = a
                    best_subsets_train = subsets_train
                    best_subsets_estimate = subsets_estimate
                    best_lab_counts = estimate_lab_counts

        fake_split = (
            len(list(filter(lambda subset: subset.count() > 0, best_subsets_estimate)))
            == 1
        )

        if best_score > 0 and not fake_split:
            newAvailable = available - set([best_a]) - uselessAs
            for subsub_train, subsub_estimate in zip(
                best_subsets_train, best_subsets_estimate
            ):
                yield (subsub_train, subsub_estimate)
            minimumLabeled = (
                c_prior * (1 - c_prior) * (1 - delta) / (delta * (1 - c_cur_best) ** 2)
            )

            for subsub_lab_count, subsub_train, subsub_estimate in zip(
                best_lab_counts, best_subsets_train, best_subsets_estimate
            ):
                if subsub_lab_count > minimumLabeled:
                    total = subsub_train.count()
                    if (
                        total > minExamples
                    ):  # stop criterion: minimum size for splitting
                        train_lab_count = (subsub_train & labels).count()
                        if (
                            lab_count != 0 and lab_count != total
                        ):  # stop criterion: purity
                            heapq.heappush(
                                priorityq,
                                (
                                    -low_c(subsub_train, labels, delta, 0, c=c_prior),
                                    -train_lab_count,
                                    subsub_train,
                                    subsub_estimate,
                                    newAvailable,
                                    depth + 1,
                                ),
                            )
