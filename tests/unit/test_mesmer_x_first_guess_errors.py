import numpy as np
import pytest

from mesmer.mesmer_x import (
    ConditionalDistributionOptions,
    Expression,
)
from mesmer.mesmer_x._first_guess import _FirstGuess as FirstGuess


@pytest.fixture
def expr():
    return Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")


@pytest.fixture
def options():
    return ConditionalDistributionOptions()


def test_fg_errors(expr, options):
    n = 15  # must be > 10 for smoothing
    with pytest.raises(ValueError, match="nan values in predictors"):
        FirstGuess(
            expr,
            options,
            data_pred=np.ones(n) * np.nan,
            predictor_names=["tas"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(ValueError, match="infinite values in predictors"):
        FirstGuess(
            expr,
            options,
            data_pred=np.ones(n) * np.inf,
            predictor_names=["tas"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(ValueError, match="nan values in target"):
        FirstGuess(
            expr,
            options,
            data_pred=np.ones(n),
            predictor_names=["tas"],
            data_targ=np.ones(n) * np.nan,
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(ValueError, match="infinite values in target"):
        FirstGuess(
            expr,
            options,
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
            expr,
            options,
            data_pred=np.ones(n),
            predictor_names=["tas"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2, 3]),
        )
