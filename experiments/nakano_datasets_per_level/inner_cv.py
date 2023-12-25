from warnings import warn
from collections import defaultdict

from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate
from sklearn.model_selection._search import BaseSearchCV


def cross_validate_nested_scores(estimator, X, y, scoring, *args, **kwargs):
    scoring = check_scoring(scoring)
    results = defaultdict(list)

    results |= cross_validate(
        *args, estimator=estimator, X=X, y=y, scoring=scoring, **kwargs,
    )

    for model, (train, test) in zip(results["estimator"], results["indices"]):
        X_test, y_test = X[test], y[test]
        for internal_model in model.estimators_:
        outer_scores = scoring(model, X_test, y_test)
        internal_scores = estimator.estimator.cv_results_
    scores = []
    
    for estimator in inner_estimators:
        estimator_score = grid_search_cv.score_estimator(estimator)
        scores.append(estimator_score)
    
    return scores
