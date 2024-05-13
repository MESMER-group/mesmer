import numpy as np

# from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize, rosen_der
from sklearn.preprocessing import PowerTransformer, StandardScaler


def lambda_function(coeffs, local_yearly_T):
    return 2 / (1 + coeffs[0] * np.exp(local_yearly_T * coeffs[1]))


class PowerTransformerVariableLambda(PowerTransformer):
    """Apply a power transform gridcellwise to make monthly residuals more Gaussian-like.
    The class inherits from Sklearn's Power transofrmer class. It is modified
    to allow for transformation parameters (lambda) which have a functional
    dependency on spatially resolved yearly mean temperature.
    Every month requires its own PowerTransform.

    Please refer to [1] for a description of the  Power transformer class.
    Please refer to [2] for an explanation of the modifications.

    Parameters
    ----------
    **kwargs :
        refer to the Power Transformer class to see a full list of possible options

    Attributes
    ----------
    coeffs_ : ndarray of shape (n_gridcell, n_coefficients)
        The coefficients to calculate lambda depending on the local yearly temperature.
        Defined via function lambda_function(coeff, local_yearly_T) as exponential dependency following [2]:
            def lambda_function(coeff, local_yearly_T):
                return(2/(1+coeff[0]*np.exp(local_yearly_T*coeff[1])))
            and n_coefficients = 2
    lambdas_ : ndarray of float of shape (n_gridcell, n_years)
        The parameters of the power transformation for each gridcell, calculated using lambda_function.

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    .. [2] Nath, S., Lejeune, Q., Beusch, L., Schleussner, C. F., & Seneviratne, S. I. (2021).
           MESMER-M: an Earth System Model emulator for spatially resolved monthly temperatures.
           Earth System Dynamics Discussions, 1-38..
    Examples
    --------
    TBD
    """

    # TODO: we dont actually save the lambdas anywhere, should we? then adjust documentation

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

    def fit(self, monthly_residuals, yearly_T, n_gridcells):
        """Estimate the optimal parameter lambda for each gridcell, given
        temperature residuals for one month of the year.
        The optimal lambda parameter for minimizing skewness is estimated on
        each gridcell independently using maximum likelihood.
        Parameters
        ----------
        monthly_residuals : ndarray of shape (n_years, n_gridcells)
            Monthly residuals after removing harmonic model fits, used to fit for the optimal transformation parameters (lambdas).
            Contains the value of one month for all years, e.g. all January values.
        yearly_T :  ndarray of shape (n_years, n_gridcells)
            yearly temperature values used as predictors for the lambdas.
        Returns
        -------
        self : object
        """
        # TODO: write what is in self
        # TODO: infer n_gridcells from data
        monthly_residuals = (
            monthly_residuals.copy()
        )  # force copy so that fit does not change X inplace

        self.coeffs_ = np.array(
            Parallel(n_jobs=-1, verbose=False)(
                delayed(self._yeo_johnson_optimize_lambda)(
                    monthly_residuals[:, gridcell], yearly_T[:, gridcell]
                )
                for gridcell in np.arange(n_gridcells)
            )
        )  # ,  desc = 'Optimizing Fmin:', position = 0, leave = True, file=sys.stdout)))
        # self.coeffs_ = np.array([self._yeo_johnson_optimize_fmin(X[:,i_grid], X_func[:,i_grid]) for i_grid in tqdm(np.arange(n_index),  desc = 'Optimizing Fmin:', position = 0, leave = True, file=sys.stdout)])
        # xarray: optim_function(col) for col in X.T
        # print(self.coeffs_.shape)
        self.mins_ = np.amin(monthly_residuals, axis=0)
        self.maxs_ = np.amax(monthly_residuals, axis=0)
        # print(self.coeffs_)

        if self.standardize:
            self._scaler = StandardScaler(copy=True)
            self._scaler.fit(monthly_residuals)

        return self

    def _yeo_johnson_optimize_lambda(self, local_monthly_residuals, local_yearly_T):
        """Find and return optimal lambda parameter of the Yeo-Johnson
        transform by MLE, for observed local monthly residual temperatures.
        Like for Box-Cox, MLE is done via the brent optimizer.
        Parameters
        ----------
        local_monthly_residuals : ndarray of shape (n_years,)
            Monthly residuals of one gridcell, used to fit for the optimal lambda.
        local_yearly_T :  ndarray of shape (n_years,)
            yearly temperature values of one gridcell used as predictor for lambda.
        Returns
        -------
        res: optimized coefficients for lambda function
        """

        def _neg_log_likelihood(coeffs):
            """Return the negative log likelihood of the observed local monthly residual temperatures
            as a function of lambda."""
            lambdas = lambda_function(coeffs, local_yearly_T)
            # version with sklearn yeo johnson transform
            # x_trans = np.zeros_like(x)
            # for i, lmbda in enumerate(lambdas):
            #     x_trans[i] = self._yeo_johnson_transform(x[i], lmbda)

            # version with own power transform
            transformed_local_monthly_resids = self._yeo_johnson_transform(
                local_monthly_residuals, lambdas
            )

            n_samples = local_monthly_residuals.shape[0]
            loglikelihood = (
                -n_samples / 2 * np.log(transformed_local_monthly_resids.var())
            )
            loglikelihood += (
                (lambdas - 1)
                * np.sign(local_monthly_residuals)
                * np.log1p(np.abs(local_monthly_residuals))
            ).sum()

            return -loglikelihood

        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        local_monthly_residuals = local_monthly_residuals[
            ~np.isnan(local_monthly_residuals)
        ]
        local_yearly_T = local_yearly_T[~np.isnan(local_yearly_T)]

        # choosing bracket -2, 2 like for boxcox
        bounds = np.c_[[0, -0.1], [1, 0.1]]
        # first guess is that data is already normal distributed
        first_guess = np.array([1, 0])

        return minimize(
            _neg_log_likelihood,
            first_guess,
            bounds=bounds,
            method="SLSQP",
            jac=rosen_der,
        ).x

    def _yeo_johnson_transform(self, local_monthly_residuals, lambdas):
        """Return transformed input local_monthly_residuals following Yeo-Johnson transform with
        parameter lambda.
        """

        eps = np.finfo(np.float64).eps

        transformed = np.zeros_like(local_monthly_residuals)
        # get positions of four cases:
        pos_a = (local_monthly_residuals >= 0) & (np.abs(lambdas) <= eps)
        pos_b = (local_monthly_residuals >= 0) & (np.abs(lambdas) > eps)
        pos_c = (local_monthly_residuals < 0) & (np.abs(lambdas - 2) > eps)
        pos_d = (local_monthly_residuals < 0) & (np.abs(lambdas - 2) <= eps)

        # assign values for the four cases
        transformed[pos_a] = np.log1p(local_monthly_residuals[pos_a])
        transformed[pos_b] = (
            np.power(local_monthly_residuals[pos_b] + 1, lambdas[pos_b]) - 1
        ) / lambdas[pos_b]
        transformed[pos_c] = -(
            np.power(-local_monthly_residuals[pos_c] + 1, 2 - lambdas[pos_c]) - 1
        ) / (2 - lambdas[pos_c])
        transformed[pos_d] = -np.log1p(-local_monthly_residuals[pos_d])

        return transformed

    def transform(self, monthly_residuals, yearly_T):
        """Apply the power transform to each feature using the fitted lambdas.
        Parameters
        ----------
        monthly_residuals : array-like, shape (n_years, n_gridcells)
            The monthly temperature data to be transformed using a power transformation with the fitted self.coeffs_.
            Contains the yearly values of one month for all gridcells, e.g. all January values.
        yearly_T: array-like, shape (n_years, n_gridcells)
            The yearly temperature values used as predictors for the lambdas using the fitted self.coeffs_.
        Returns
        -------
        transformed_monthly_resids : array-like, shape (n_years, n_gridcells)
            The transformed monthly residuals.
        """
        lambdas = self._get_yeo_johnson_lambdas(yearly_T)

        transformed_monthly_resids = np.zeros_like(monthly_residuals)

        # for gridcell, lmbda in enumerate(lambdas.T):
        #     for year, year_lmbda in enumerate(lmbda):
        #         with np.errstate(invalid='ignore'):  # hide NaN warnings
        #             transformed_monthly_resids[year, gridcell] = self._yeo_johnson_transform(monthly_residuals[year, gridcell], year_lmbda)
        for gridcell, lmbda in enumerate(lambdas.T):
            transformed_monthly_resids[:, gridcell] = self._yeo_johnson_transform(
                monthly_residuals[:, gridcell], lmbda
            )

        if self.standardize:
            transformed_monthly_resids = self._scaler.transform(
                transformed_monthly_resids
            )

        return transformed_monthly_resids

    def _get_yeo_johnson_lambdas(self, yearly_T):
        # TODO: check the dimensions in all of this

        lambdas = np.zeros_like(yearly_T)
        gridcell = 0
        # TODO: sure yearly_T.T gives local yearly T?
        for coeffs, local_yearly_T in zip(self.coeffs_, yearly_T.T):
            lambdas[:, gridcell] = lambda_function(coeffs, local_yearly_T)
            gridcell += 1

        # TODO: this should not be necessary?
        lambdas = np.where(lambdas < 0, 0, lambdas)
        lambdas = np.where(lambdas > 2, 2, lambdas)

        return lambdas

    def inverse_transform(self, transformed_monthly_T, yearly_T):
        """Apply the inverse power transformation using the fitted lambdas.
        The inverse of the Yeo-Johnson transformation is given by::
            if X >= 0 and lambda_ == 0:
                X = exp(X_trans) - 1
            elif X >= 0 and lambda_ != 0:
                X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
            elif X < 0 and lambda_ != 2:
                X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
            elif X < 0 and lambda_ == 2:
                X = 1 - exp(-X_trans)
        Parameters
        ----------
        transformed_monthly_T : array-like, shape (n_years, n_gridcells)
            The transformed data.
        yearly_T: array-like, shape (n_years, n_gridcells)
            The yearly temperature values used as predictors for the lambdas using the fitted self.coeffs_.
        Returns
        -------
        inverted_monthly_T : array-like, shape (n_years, n_gridcells)
            The inverted, i.e. original, monthly temperature values.
        """
        # TODO: if we save the lambdas, we would not need to give yearly temperaure here
        if self.standardize:
            transformed_monthly_T = self._scaler.inverse_transform(
                transformed_monthly_T
            )

        inverted_monthly_T = np.zeros_like(transformed_monthly_T)

        lambdas = self._get_yeo_johnson_lambdas(yearly_T)

        for gridcell, lmbda in enumerate(lambdas.T):
            for year, y_lmbda in enumerate(lmbda):
                with np.errstate(invalid="ignore"):  # hide NaN warnings
                    inverted_monthly_T[year, gridcell] = (
                        self._yeo_johnson_inverse_transform(
                            transformed_monthly_T[year, gridcell], y_lmbda
                        )
                    )

            # clip values to not exceed original range
            # apparently a relict from when lambda was not constrained to [0,2]
            inverted_monthly_T[:, gridcell] = np.where(
                inverted_monthly_T[:, gridcell] < self.mins_[gridcell],
                self.mins_[gridcell],
                inverted_monthly_T[:, gridcell],
            )
            inverted_monthly_T[:, gridcell] = np.where(
                inverted_monthly_T[:, gridcell] > self.maxs_[gridcell],
                self.maxs_[gridcell],
                inverted_monthly_T[:, gridcell],
            )

        return inverted_monthly_T
