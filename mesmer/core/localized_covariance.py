import numpy as np
from scipy.stats import multivariate_normal


def _adjust_ecov_ar1_np(covariance, ar_coefs):
    """
    adjust localized empirical covariance matrix for autoregressive process of order 1

    Parameters
    ----------
    covariance : 2D np.array
        Empirical covariance matrix.
    ar_coefs : 1D np.array
        The coefficients of the autoregressive process of order 1.

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

    reduction_factor = np.sqrt(1 - ar_coefs**2)
    reduction_factor = np.atleast_2d(reduction_factor)  # so it can be transposed

    # equivalent to ``diag(reduction_factor) @ ecov @ diag(reduction_factor)``
    return reduction_factor * reduction_factor.T * covariance


def _find_localized_empirical_covariance_np(data, weights, localizer, k_folds):
    """determine localized empirical covariance by cross validation

    Parameters
    ----------
    data : 2D array
        Data array with shape nr_samples x nr_gridpoints.
    weights : 1D array
        Weights for the individual samples.
    localizer : dict of array-like
        Dictonary containing the localization radii as keys and the localisation as
        values. The localization must be 2D and of shape nr_gridpoints x nr_gridpoints.
    k_folds : int
        Number of folds to use for cross validation.

    Returns
    -------
    localisation_radius : float
        Selected localisation radius.
    ecov : ndarray
        Empirical covariance matrix.
    loc_ecov : ndarray
        Localized empirical covariance matrix.

    Notes
    -----
    Runs a k-fold cross validation if ``k_folds`` is smaller than the number of samples
    and a leave-one-out cross validation otherwise.
    """

    localisation_radii = sorted(localizer.keys())

    # find _local_ minimum because
    # experience tells: once stop selecting larger localisation radii, will not
    # start again. Better to stop once min is reached (to limit computational effort
    # and singular matrices).

    localisation_radius = _minimize_local_discrete(
        _ecov_crossvalidation,
        localisation_radii,
        data=data,
        weights=weights,
        localizer=localizer,
        k_folds=k_folds,
    )

    ecov = np.cov(data, rowvar=False, aweights=weights)
    loc_ecov = localizer[localisation_radius] * ecov

    return localisation_radius, ecov, loc_ecov


def _ecov_crossvalidation(localisation_radius, *, data, weights, localizer, k_folds):
    """k-fold crossvalidation for a single localisation radius"""

    nr_samples, __ = data.shape
    nr_iterations = min(nr_samples, k_folds)

    log_likelihood = 0

    for it in range(nr_iterations):

        # every `k_folds` element for validation such that each is used exactly once
        sel = np.ones(nr_samples, dtype=bool)
        sel[it::k_folds] = False

        # extract training set
        data_est = data[sel, :]
        weights_est = weights[sel]

        # extract validation set
        data_cv = data[~sel, :]
        weights_cv = weights[~sel]

        # compute (localized) empirical covariance
        ecov = np.cov(data_est, rowvar=False, aweights=weights_est)
        loc_ecov = localizer[localisation_radius] * ecov

        # sum log likelihood of all crossvalidation folds
        log_likelihood += _get_neg_loglikelihood(data_cv, loc_ecov, weights_cv)

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

    Notes
    -----
    The mean is zero for all points.
    """

    # NOTE: 90 % of time is spent here - not much point optimizing the rest
    log_likelihood = multivariate_normal.logpdf(
        data, cov=covariance, allow_singular=True
    )
    # ``allow_singular = True`` because stms ran into singular matrices
    # ESMs eg affected: CanESM2, CanESM5, IPSL-CM5A-LR, MCM-UA-1-0
    # -> reassuring that saw that in these ESMs L values where matrix
    # is not singular yet can end up being selected

    # weighted sum for each cv sample
    weighted_llh = np.average(log_likelihood, weights=weights) * weights.size

    return -weighted_llh


def _minimize_local_discrete(func, sequence, **kwargs):
    """find the local minimum for a function that consumes discrete input

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Should take the elements of ``sequence``
        as input and return a float that is to be minimized.
    sequence : iterable
        An iterable with discrete values to evaluate func for.
    kwargs : Mapping
        Keyword arguments passed to `func`.

    Returns
    -------
    element
        The element from sequence which corresponds to the local minimum.

    Raises
    ------
    ValueError : if `func` returns negative infinity for any input.

    Notes
    -----
    - The function determines the local minimum, i.e., the loop is aborted if
      `func(sequence[i-1]) >= func(sequence[i])`.
    """

    current_min = float("inf")
    for element in sequence:

        res = func(element, **kwargs)

        if np.isneginf(res):
            raise ValueError("`fun` returned `-inf`")
        elif res < current_min:
            current_min = res
        else:
            return element

    # warn if the local minimum is not reached?
    return element
