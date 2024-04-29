import numpy as np
import pytest
import xarray as xr

import mesmer
from mesmer.mesmer_m.power_transformer import (
    PowerTransformerVariableLambda,
    lambda_function
)

def test_lambda_function():
    # Note that we test with normally distributed data
    # which should make lambda close to 1 and
    # the coefficients close to 1 and 0
    # but for the sake of testing, we set the coefficients slighly differently 
    coeffs = [1, 0.001]
    local_yearly_T_test_data = np.random.rand(10)*100 

    # even for random numbers, the lambdas should always be between 0 and 2
    lambdas = lambda_function(coeffs, local_yearly_T_test_data)

    assert np.all(lambdas >= 0) and np.all(lambdas <= 2)

    local_yearly_T_test_data = np.array([-3,-2,-1,0,1,2,3])
    lambdas = lambda_function(coeffs, local_yearly_T_test_data)
    expected_lambdas = np.array([1.0015, 1.001 , 1.0005, 1.    , 0.9995, 0.999 , 0.9985])

