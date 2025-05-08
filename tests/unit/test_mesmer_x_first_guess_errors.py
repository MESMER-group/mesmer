import numpy as np
import pytest

from mesmer.mesmer_x import (
    ConditionalDistribution,
    ConditionalDistributionOptions,
    Expression,
)
from mesmer.mesmer_x._first_guess import FirstGuess


@pytest.fixture
def distrib():
    expr = Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")
    return ConditionalDistribution(expr, ConditionalDistributionOptions(expr))


def test_fg_errors(distrib):
    n = 15  # must be > 10 for smoothing
    with pytest.raises(ValueError, match="nan values in predictors"):
        FirstGuess(
            distrib,
            data_pred=np.ones(n) * np.nan,
            predictor_names=["tas"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(ValueError, match="infinite values in predictors"):
        FirstGuess(
            distrib,
            data_pred=np.ones(n) * np.inf,
            predictor_names=["tas"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(ValueError, match="nan values in target"):
        FirstGuess(
            distrib,
            data_pred=np.ones(n),
            predictor_names=["tas"],
            data_targ=np.ones(n) * np.nan,
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(ValueError, match="infinite values in target"):
        FirstGuess(
            distrib,
            data_pred=np.ones(n),
            predictor_names=["tas"],
            data_targ=np.ones(n) * np.inf,
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(
        ValueError, match="The provided first guess does not have the correct shape:"
    ):
        FirstGuess(
            distrib,
            data_pred=np.ones(n),
            predictor_names=["tas"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2, 3]),
        )
