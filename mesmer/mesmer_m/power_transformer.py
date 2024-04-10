import numpy as np

# from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize, rosen_der
from sklearn.preprocessing import PowerTransformer, StandardScaler


def lambda_function(coeff, local_yearly_T):
    return 2 / (1 + coeff[0] * np.exp(local_yearly_T * coeff[1]))


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

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

    def fit_fmin(self, monthly_residuals, yearly_T, n_gridcells):
        """Estimate the optimal parameter lambda for each gridcell, given 
        temperature residuals for one month of the year.
        The optimal lambda parameter for minimizing skewness is estimated on
        each gridcell independently using maximum likelihood.
        Parameters
        ----------
        monthly_residuals : ndarray of shape (n_years, n_gridcells)
            Monthly residuals after removing harmonic model fits, used to fit for the optimal transformation parameters.
            Contains the value of one month for all years, e.g. all January values.
        yearly_T :  ndarray of shape (n_years, n_gridcells)
            yearly temperature values used as predictors for lambda.
        Returns
        -------
        self : object
        """

        # TODO: infer n_gridcells from data
        monthly_residuals = (
            monthly_residuals.copy()
        )  # force copy so that fit does not change X inplace

        self.coeffs_ = np.array(
            Parallel(n_jobs=-1, verbose=False)(
                delayed(self._yeo_johnson_optimize_fmin)(
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

    def _yeo_johnson_optimize_fmin(self, local_monthly_residuals, local_yearly_T):
        """Find and return optimal lambda parameter of the Yeo-Johnson
        transform by MLE, for observed local monthly residual temperatures.
        Like for Box-Cox, MLE is done via the brent optimizer.
        """

        def _neg_log_likelihood(coeff):
            """Return the negative log likelihood of the observed local monthly residual temperatures
            as a function of lambda."""
            lambdas = lambda_function(coeff, x=x_func)
            # version with sklearn yeo johnson transform
            # x_trans = np.zeros_like(x)
            # for i, lmbda in enumerate(lambdas):
            #     x_trans[i] = self._yeo_johnson_transform(x[i], lmbda)

            # version with own power transform
            x_trans = self._yeo_johnson_transform_fmin(x, lambdas)

            n_samples = x.shape[0]
            loglike = -n_samples / 2 * np.log(x_trans.var())
            loglike += ((lambdas - 1) * np.sign(x) * np.log1p(np.abs(x))).sum()

            return -loglike

        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        x = x[~np.isnan(x)]
        x_func = x_func[~np.isnan(x_func)]

        # choosing bracket -2, 2 like for boxcox
        bounds = np.c_[[0, -0.1], [1, 0.1]]
        return minimize(
            _neg_log_likelihood,
            np.array([0.01, 0.01]),
            bounds=bounds,
            method="SLSQP",
            jac=rosen_der,
        ).x

    def _yeo_johnson_transform_fmin(self, x, lmbda):
        """Return transformed input x following Yeo-Johnson transform with
        parameter lambda.
        """

        out = np.zeros_like(x)
        # pos = x >= 0  # binary mask
        # when x >= 0
        # and lmbda = 0
        pos_a = (x >= 0) & (np.abs(lmbda) <= np.spacing(1.0))
        # and lmbda != 0
        pos_b = (x >= 0) & (np.abs(lmbda) > np.spacing(1.0))
        # when x<0
        # and lambda != 2
        pos_c = (x < 0) & (np.abs(lmbda - 2) > np.spacing(1.0))
        # and lambda == 2
        pos_d = (x < 0) & (np.abs(lmbda - 2) <= np.spacing(1.0))

        # assign values for the four cases
        out[pos_a] = np.log1p(x[pos_a])
        out[pos_b] = (np.power(x[pos_b] + 1, lmbda[pos_b]) - 1) / lmbda[pos_b]
        out[pos_c] = -(np.power(-x[pos_c] + 1, 2 - lmbda[pos_c]) - 1) / (
            2 - lmbda[pos_c]
        )
        out[pos_d] = -np.log1p(-x[pos_d])

        return out

    def transform_fmin(self, X, X_func):
        """Apply the power transform to each feature using the fitted lambdas.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to be transformed using a power transformation.
        Returns
        -------
        X_trans : array-like, shape (n_samples, n_features)
            The transformed data.
        """
        lambdas = self._get_yeo_johnson_lambdas(X_func)

        X_trans = np.zeros_like(X)

        # for i, lmbda in enumerate(lambdas.T):
        #     for j,j_lmbda in enumerate(lmbda):
        #         with np.errstate(invalid='ignore'):  # hide NaN warnings
        #             X_trans[j, i] = self._yeo_johnson_transform(X[j, i], j_lmbda)
        for i, lmbda in enumerate(lambdas.T):
            X_trans[:, i] = self._yeo_johnson_transform_fmin(X[:, i], lmbda)

        if self.standardize:
            X_trans = self._scaler.transform(X_trans)

        return X_trans

    def _get_yeo_johnson_lambdas(self, X_func):

        lambdas = np.zeros_like(X_func)
        i = 0
        for a, b in zip(self.coeffs_, X_func.T):
            lambdas[:, i] = lambda_function(a, b)
            i += 1

        lambdas = np.where(lambdas < 0, 0, lambdas)
        lambdas = np.where(lambdas > 2, 2, lambdas)

        return lambdas

    def inverse_transform_fmin(self, X, X_func):
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
        X : array-like, shape (n_samples, n_features)
            The transformed data.
        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            The original data
        """

        if self.standardize:
            X = self._scaler.inverse_transform(X)

        X_inv = np.zeros_like(X)

        lambdas = self._get_yeo_johnson_lambdas(X_func)

        for i, lmbda in enumerate(lambdas.T):
            for j, j_lmbda in enumerate(lmbda):
                with np.errstate(invalid="ignore"):  # hide NaN warnings
                    X_inv[j, i] = self._yeo_johnson_inverse_transform(X[j, i], j_lmbda)
            X_inv[:, i] = np.where(
                X_inv[:, i] < self.mins_[i], self.mins_[i], X_inv[:, i]
            )
            X_inv[:, i] = np.where(
                X_inv[:, i] > self.maxs_[i], self.maxs_[i], X_inv[:, i]
            )

        return X_inv
