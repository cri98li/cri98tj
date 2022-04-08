import pytest
from sklearn.utils.estimator_checks import check_estimator

from cri98tj import TemplateClassifier
from cri98tj import TemplateEstimator
from cri98tj import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
