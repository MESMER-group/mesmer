import warnings

import numpy as np
import scipy
import xarray as xr

from mesmer.core.utils import (
    LinAlgWarning,
    _check_dataarray_form,
    _create_equal_dim_names,
    _minimize_local_discrete,
)


def adjust_covariance_ar1(
    covariance: xr.DataArray, ar_coefs: xr.DataArray
) -> xr.DataArray:
    """
    adjust localized empirical covariance matrix for autoregressive process of order one

    Parameters
    ----------
    covariance : 2D xr.DataArray
        Empirical covariance matrix.
    ar_coefs : 1D xr.DataArray
        The coefficients of the autoregressive process of order 1.
        Must have length equal to the size of `covariance`.

    Returns
    -------
    adjusted_covariance : xr.DataArray
        Adjusted empirical covariance matrix.

    Notes
    -----
    - Adjusts ``covariance`` for an AR(1) process according to [1]_, eq (8).

    - The formula is specific for an AR(1) process, see also `#167 (comment)
      <https://github.com/MESMER-group/mesmer/pull/167#discussion_r912481495>`__.

    - According to [2]_ "The multiplication with the ``reduction_factor`` scales the
      empirical standard error under the assumption of an autoregressive process of
      order one [3]_. This accounts for the fact that the variance of an autoregressive
      process is larger than that of the driving white noise process."

    - This formula is wrong in [1]_. However, it is correct in the code. See also [2]_
      and [3]_.

    .. [1] Beusch, L., Gudmundsson, L., and Seneviratne, S. I.: Emulating Earth system model
       temperatures with MESMER: from global mean temperature trajectories to grid-point-
       level realizations on land, Earth Syst. Dynam., 11, 139–159,
       https://doi.org/10.5194/esd-11-139-2020, 2020.

    .. [2] Humphrey, V. and Gudmundsson, L.: GRACE-REC: a reconstruction of climate-driven
       water storage changes over the last century, Earth Syst. Sci. Data, 11, 1153–1170,
       https://doi.org/10.5194/essd-11-1153-2019, 2019.

    .. [3] Cressie, N. and Wikle, C. K.: Statistics for spatio-temporal data, John Wiley
       & Sons, Hoboken, New Jersey, USA, 2011.
    """

    # pass ar_coefs.data - so it will 'just work'
    return _adjust_ecov_ar1_np(covariance, ar_coefs.data)


def _adjust_ecov_ar1_np(covariance, ar_coefs):

    ar_coefs = ar_coefs.squeeze()  # allow n x 1 ar_coeffs
    if ar_coefs.ndim != 1 or ar_coefs.size != covariance.shape[0]:
        raise ValueError(
            "`ar_coefs` must be 1D and have length equal to the size of `covariance`"
        )

    reduction_factor = np.sqrt(1 - ar_coefs**2)
    reduction_factor = np.atleast_2d(reduction_factor)  # so it can be transposed

    # equivalent to ``diag(reduction_factor) @ covariance @ diag(reduction_factor)``
    return reduction_factor * reduction_factor.T * covariance


def find_localized_empirical_covariance(
    data: xr.DataArray,
    weights: xr.DataArray,
    localizer: dict[float | int, xr.DataArray | np.ndarray],
    dim: str,
    k_folds: int,
    equal_dim_suffixes: tuple[str, str] = ("_i", "_j"),
    allow_singluar: bool = False,
) -> xr.Dataset:
    """determine localized empirical covariance by cross validation

    Parameters
    ----------
    data : xr.DataArray
        2D DataArray with data to calculate the covariance for.
    weights : xr.DataArray
        Weights for the individual samples.
    localizer : dict of xr.DataArray
        Dictionary containing the localization radii as keys and the localization matrix
        as values. The localization must be 2D and of shape n_gridpoints x n_gridpoints.
        Currently only the Gaspari-Cohn localizer is implemented in MESMER.
    dim : str
        Dimension along which to calculate the covariance.
    k_folds : int
        Number of folds to use for cross validation.
    equal_dim_suffixes : tuple of str, default: ("_i", "_j")
        Suffixes to add to the the name of ``dim`` for the covariance array
        (xr.DataArray cannot have two dimensions with the same name).
    allow_singluar : bool, default: False
        If True, allow singular matrices to be used in the cross validation. In this case,
        the method of decomposition is switched from cholesky to eigh and a warning is emitted.

    Returns
    -------
    localized_empirical_covariance : xr.Dataset
        Dataset containing three DataArrays:
    localization_radius : float
        Selected localization radius.
    covariance : xr.DataArray
        Empirical covariance matrix.
    localized_covariance : xr.DataArray
        Localized empirical covariance matrix.

    Notes
    -----
    Runs a k-fold cross validation if ``k_folds`` is smaller than the number of samples
    and a leave-one-out cross validation otherwise.
    """

    _check_dataarray_form(data, name="data", ndim=2)

    # ensure data has the right orientation
    data = data.transpose(dim, ...)
    all_dims = data.dims

    (sample_dim,) = set(all_dims) - {dim}
    out_dims = _create_equal_dim_names(sample_dim, equal_dim_suffixes)

    out = xr.apply_ufunc(
        _find_localized_empirical_covariance_np,
        data,
        weights,
        kwargs={
            "localizer": localizer,
            "k_folds": k_folds,
            "allow_singluar": allow_singluar,
        },
        input_core_dims=[all_dims, [dim]],
        output_core_dims=([], out_dims, out_dims),
    )
    localization_radius, covariance, localized_covariance = out

    data_vars = {
        "localization_radius": localization_radius,
        "covariance": covariance,
        "localized_covariance": localized_covariance,
    }

    return xr.Dataset(data_vars)


def find_localized_empirical_covariance_monthly(
    data: xr.DataArray,
    weights: xr.DataArray,
    localizer: dict[float | int, xr.DataArray | np.ndarray],
    dim: str,
    k_folds: int,
    equal_dim_suffixes: tuple[str, str] = ("_i", "_j"),
    allow_singluar: bool = False,
) -> xr.Dataset:
    """determine localized empirical covariance by cross validation for each month. `data`
    should be the residuals of the cyclo-stationary AR(1) process, see
    :func:`fit_auto_regression_monthly <mesmer.stats.fit_auto_regression_monthly>`. Note that here,
    no additional adjustment is necessary.

    Parameters
    ----------
    data : xr.DataArray
        2D DataArray with monthly data to calculate the covariance for (residuals of the AR(1)
        process).
    weights : xr.DataArray
        Weights for the individual samples.
    localizer : dict of DataArray
        Dictionary containing the localization radii as keys and the localization matrix
        as values. The localization must be 2D and of shape n_gridpoints x n_gridpoints.
        Currently only the Gaspari-Cohn localizer is implemented in MESMER.
    dim : str
        Dimension along which to calculate the covariance.
    k_folds : int
        Number of folds to use for cross validation.
    equal_dim_suffixes : tuple of str, default: ("_i", "_j")
        Suffixes to add to the the name of ``dim`` for the covariance array
        (xr.DataArray cannot have two dimensions with the same name).
    allow_singluar : bool, default: False
        If True, allow singular matrices to be used in the cross validation. In this case,
        the method of decomposition is switched from cholesky to eigh and a warning is emitted.

    Returns
    -------
    localized_empirical_covariance : xr.Dataset
        Dataset containing three DataArrays:
    localization_radius : float
        Selected localization radius.
    covariance : xr.DataArray
        Empirical covariance matrix.
    localized_covariance : xr.DataArray
        Localized empirical covariance matrix.

    Notes
    -----
    Runs a k-fold cross validation if ``k_folds`` is smaller than the number of samples
    and a leave-one-out cross validation otherwise.
    """
    localized_ecov = []
    data_grouped = data.groupby(f"{dim}.month")
    weights_grouped = weights.groupby(f"{dim}.month")

    for mon in range(1, 13):
        res = find_localized_empirical_covariance(
            data_grouped[mon],
            weights_grouped[mon],
            localizer,
            dim=dim,
            k_folds=k_folds,
            equal_dim_suffixes=equal_dim_suffixes,
            allow_singluar=allow_singluar,
        )
        localized_ecov.append(res)

    month = xr.DataArray(range(1, 13), dims="month")
    return xr.concat(localized_ecov, dim=month)


def _find_localized_empirical_covariance_np(
    data, weights, localizer, k_folds, allow_singluar
):
    """determine localized empirical covariance by cross validation

    Parameters
    ----------
    data : 2D array
        Data array with shape n_samples x n_gridpoints.
    weights : 1D array
        Weights for the individual samples.
    localizer : dict of array-like
        Dictionary containing the localization radii as keys and the localization matrix
        as values. The localization must be 2D and of shape nr_gridpoints x nr_gridpoints.
        Currently only the Gaspari-Cohn localizer is implemented in MESMER.
    k_folds : int
        Number of folds to use for cross validation.
    allow_singluar : bool
        If True, allow singular matrices to be used in the cross validation. In this case,
        the method of decomposition is switched from cholesky to eigh and a warning is emitted.

    Returns
    -------
    localization_radius : float
        Selected localization radius.
    covariance : ndarray
        Empirical covariance matrix.
    localized_covariance : ndarray
        Localized empirical covariance matrix.

    Notes
    -----
    Runs a k-fold cross validation if ``k_folds`` is smaller than the number of samples
    and a leave-one-out cross validation otherwise.
    """

    if not isinstance(k_folds, int) or k_folds <= 1:
        raise ValueError(f"'k_folds' must be an integer larger than 1, got {k_folds}.")

    if data.shape[0] != weights.size:
        raise ValueError("weights and data have incompatible shape")

    localization_radii = sorted(localizer.keys())

    # find _local_ minimum because
    # experience tells: once we stop selecting larger localization radii, we will not
    # start again. Better to stop once min is reached (to limit computational effort
    # and singular matrices).

    # start with cholesky decomposition
    # function returns method and can switch to eigh() if cov is singular
    localization_radius = _minimize_local_discrete(
        _EcovCrossvalidation(method="cholesky").crossvalidate,
        localization_radii,
        data=data,
        weights=weights,
        localizer=localizer,
        k_folds=k_folds,
        allow_singluar=allow_singluar,
    )

    covariance = np.cov(data, rowvar=False, aweights=weights)
    localized_covariance = localizer[localization_radius] * covariance

    return localization_radius, covariance, localized_covariance


class _EcovCrossvalidation:
    # we organize this as a class so we can store `method` as state
    # ensures we use `eigh` after `cholesky` fails for one localization
    # radius
    def __init__(self, method=None):

        self.method = method or "cholesky"

    def crossvalidate(
        self, localization_radius, *, data, weights, localizer, k_folds, allow_singluar
    ):
        """k-fold crossvalidation for a single localization radius"""

        n_samples, __ = data.shape
        n_iterations = min(n_samples, k_folds)

        nll = 0  # negative log likelihood

        for it in range(n_iterations):

            # every `k_folds` element for validation such that each is used exactly once
            sel = np.ones(n_samples, dtype=bool)
            sel[it::k_folds] = False

            # extract training set
            data_train, weights_train = data[sel, :], weights[sel]

            # extract validation set
            data_cv, weights_cv = data[~sel, :], weights[~sel]

            # compute (localized) empirical covariance
            cov = np.cov(data_train, rowvar=False, aweights=weights_train)
            localized_cov = localizer[localization_radius] * cov

            try:
                # sum log likelihood of all crossvalidation folds
                nll += _get_neg_loglikelihood(
                    data_cv, localized_cov, weights_cv, self.method
                )

            except np.linalg.LinAlgError as e:
                if not allow_singluar:
                    raise np.linalg.LinAlgError(
                        f"Singular matrix for localization_radius of {localization_radius}."
                    )

                if self.method == "eigh":
                    raise e

                # switch to eigh from now on
                self.method = "eigh"
                warnings.warn(
                    f"Singular matrix for localization_radius of {localization_radius}."
                    "\n Switching to eigh().",
                    LinAlgWarning,
                )

                nll += _get_neg_loglikelihood(
                    data_cv, localized_cov, weights_cv, self.method
                )

        return nll


def _get_neg_loglikelihood(data, covariance, weights, method):
    """calculate weighted log likelihood for multivariate normal distribution

    Parameters
    ----------
    data : 2D array
        Data array used for cross validation.
    covariance : 2D array
        Localized empirical variance matrix.
    weights : 1D array
        Sample weights

    Returns
    -------
    weighted_nll : float
        Weighted negative log likelihood

    Raises
    ------
    np.linalg.LinAlgError if a singular covariance matrix is passed. See `#187
    <https://github.com/MESMER-group/mesmer/issues/187>`__ for a discussion.

    Notes
    -----
    The mean is assumed to be zero for all points.
    """

    if method == "cholesky":
        L = np.linalg.cholesky(covariance)
        cov = scipy.stats.Covariance.from_cholesky(L)
    else:
        w, v = np.linalg.eigh(covariance)
        cov = scipy.stats.Covariance.from_eigendecomposition((w, v))

    log_likelihood = scipy.stats.multivariate_normal.logpdf(
        data, cov=cov, allow_singular=True
    )

    # logpdf can return a scalar, which np.average does not like
    log_likelihood = np.atleast_1d(log_likelihood)

    # weighted sum for each cv sample
    # equals `log_likelihood @ weights * weights.size / weights.sum()`
    weighted_nll = -np.average(log_likelihood, weights=weights) * weights.size

    return weighted_nll
