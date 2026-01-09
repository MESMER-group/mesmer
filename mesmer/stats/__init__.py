from mesmer.stats._auto_regression import (
    draw_auto_regression_correlated,
    draw_auto_regression_monthly,
    draw_auto_regression_uncorrelated,
    fit_auto_regression,
    fit_auto_regression_monthly,
    fit_auto_regression_scen_ens,
    select_ar_order,
    select_ar_order_scen_ens,
)
from mesmer.stats._gaspari_cohn import gaspari_cohn, gaspari_cohn_correlation_matrices
from mesmer.stats._harmonic_model import fit_harmonic_model, predict_harmonic_model
from mesmer.stats._linear_regression import LinearRegression
from mesmer.stats._localized_covariance import (
    adjust_covariance_ar1,
    find_localized_empirical_covariance,
    find_localized_empirical_covariance_monthly,
)

from mesmer.stats._principal_component_decomposition import (
    fit_principal_components,
    inverse_transform_principal_components,
    transform_principal_components,
)
from mesmer.stats._regularized_glm import GammaGLMXarray
from mesmer.stats._power_transformer import YeoJohnsonTransformer
from mesmer.stats._smoothing import lowess
from mesmer.stats._xarray_kde import GroupedKDEXarray
from mesmer.stats._xarray_pipelines import XarrayPipeline
from mesmer.stats._xarray_transformers import SklearnXarrayTransformer

__all__ = [
    # auto regression
    "fit_auto_regression_scen_ens",
    "select_ar_order_scen_ens",
    "draw_auto_regression_correlated",
    "draw_auto_regression_uncorrelated",
    "fit_auto_regression",
    "select_ar_order",
    "fit_auto_regression_monthly",
    "draw_auto_regression_monthly",
    # gaspari cohn
    "gaspari_cohn_correlation_matrices",
    "gaspari_cohn",
    # linear regression
    "LinearRegression",
    # localized covariance
    "adjust_covariance_ar1",
    "find_localized_empirical_covariance",
    "find_localized_empirical_covariance_monthly",
    # smoothing
    "lowess",
    # harmonic model
    "fit_harmonic_model",
    "predict_harmonic_model",
    # power transformer
    "YeoJohnsonTransformer",
    "GammaGLMXarray",
    "GroupedKDEXarray",
    "XarrayPipeline",
    "SklearnXarrayTransformer",
    "fit_principal_components",
    "inverse_transform_principal_components",
    "transform_principal_components",
]
