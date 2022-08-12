import warnings

import numpy as np
from scipy.stats import multivariate_normal

from mesmer.core.utils import LinAlgWarning, _minimize_local_discrete


def _adjust_ecov_ar1_np(covariance, ar_coefs):
    """
    adjust localized empirical covariance matrix for autoregressive process of order 1

    Parameters
    ----------
    covariance : 2D np.array
        Empirical covariance matrix.
    ar_coefs : 1D np.array
        The coefficients of the autoregressive process of order 1.
        Must have length equal to the size of `covariance`.

    Returns
    -------
    adjusted_covariance : np.array
        Adjusted empirical covariance matrix.

    Notes
    -----
    - Adjusts ``covariance`` for an AR(1) process according to [1]_, eq (8).

    - The formula is specific for an AR(1) process, see also `#167 (comment)
      https://github.com/MESMER-group/mesmer/pull/167#discussion_r912481495`__.

    - According to [2]_ "The multiplication with the ``reduction_factor`` scales the
      empirical standard error under the assumption of an autoregressive process of
      order 1 [3]_. This accounts for the fact that the variance of an autoregressive
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

    if ar_coefs.ndim != 1 or ar_coefs.size != covariance.shape[0]:
        raise ValueError(
            "`ar_coefs` must be 1D and have length equal to the size of `covariance`"
        )

    reduction_factor = np.sqrt(1 - ar_coefs**2)
    reduction_factor = np.atleast_2d(reduction_factor)  # so it can be transposed

    # equivalent to ``diag(reduction_factor) @ covariance @ diag(reduction_factor)``
    return reduction_factor * reduction_factor.T * covariance


def _find_localized_empirical_covariance_np(data, weights, localizer, k_folds):
    """determine localized empirical covariance by cross validation

    Parameters
    ----------
    data : 2D array
        Data array with shape n_samples x n_gridpoints.
    weights : 1D array
        Weights for the individual samples.
    localizer : dict of array-like
        Dictonary containing the localization radii as keys and the localization matrix
        as values. The localization must be 2D and of shape nr_gridpoints x nr_gridpoints.
        Currently only the Gaspari-Cohn localizer is implemented in MESMER.
    k_folds : int
        Number of folds to use for cross validation.

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
        raise ValueError("'k_folds' must be an integer larger than 1.")

    if data.shape[0] != weights.size:
        raise ValueError("weights and data have incompatible shape")

    localization_radii = sorted(localizer.keys())

    # find _local_ minimum because
    # experience tells: once we stop selecting larger localization radii, we will not
    # start again. Better to stop once min is reached (to limit computational effort
    # and singular matrices).

    localization_radius = _minimize_local_discrete(
        _ecov_crossvalidation,
        localization_radii,
        data=data,
        weights=weights,
        localizer=localizer,
        k_folds=k_folds,
    )

    covariance = np.cov(data, rowvar=False, aweights=weights)
    localized_covariance = localizer[localization_radius] * covariance

    return localization_radius, covariance, localized_covariance


def _ecov_crossvalidation(localization_radius, *, data, weights, localizer, k_folds):
    """k-fold crossvalidation for a single localization radius"""

    n_samples, __ = data.shape
    n_iterations = min(n_samples, k_folds)

    log_likelihood = 0

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
            log_likelihood += _get_neg_loglikelihood(data_cv, localized_cov, weights_cv)
        except np.linalg.LinAlgError:
            warnings.warn(
                f"Singular matrix for localization_radius of {localization_radius}."
                " Skipping this radius.",
                LinAlgWarning,
            )
            return float("inf")

    return log_likelihood


def _get_neg_loglikelihood(data, covariance, weights):
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
    weighted_log_likelihood : float

    Raises
    ------
    np.linalg.LinAlgError if a singular covariance matrix is passed. See `#187
    <https://github.com/MESMER-group/mesmer/issues/187>`__ for a discussion.

    Notes
    -----
    The mean is assumed to be zero for all points.
    """

    # NOTE: 90 % of time is spent in multivariate_normal.logpdf - not much point
    # optimizing the rest

    log_likelihood = multivariate_normal.logpdf(data, cov=covariance)

    # logpdf can return a scalar, which np.average does not like
    log_likelihood = np.atleast_1d(log_likelihood)

    # weighted sum for each cv sample
    # equals `log_likelihood @ weights * weights.size / weights.sum()`
    weighted_llh = np.average(log_likelihood, weights=weights) * weights.size

    return -weighted_llh
