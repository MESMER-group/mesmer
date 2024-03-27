import numpy as np

# from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize, rosen_der
from sklearn.preprocessing import PowerTransformer, StandardScaler


def lambda_function(coeff, x):
    return 2 / (1 + coeff[0] * np.exp(x * coeff[1]))


class PowerTransformerVariableLambda(PowerTransformer):
    """Apply a power transform featurewise to make data more Gaussian-like.
    The class inherits from Sklearn's Power transofrmer class. It is modified
    to allow for traansformation parameters (lambda) which have a functional
    dependency on spatially resolved yearly mean temperature.

    Please refer to [1] for a description of the  Power transformer class.
    Please refer to [2] for an explanation of the modifications.

    Parameters
    ----------
    **kwargs :
        refer to the Power Transformer class to see a full list of possible
        options

    Attributes
    ----------
    coeffs_ : ndarray of shape (n_featuers, n_functional_parameters)
        The parameters of the functional dependency lambda follows.
        Defined via function lambda_function(coeff, x)
            e.g. for exponential dependency following [2]:
                def lambda_function(coeff, x):
                    return(2/(1+coeff[0]*np.exp(x*coeff[1])))
                in this case n_functional_parameters = 2
            e.g. for polynomial dependency following:
                def lambda_function(coeff, x):
                    return(coeff[0] + coeff[1]*x + coeff[2]*x**2 )
                in this case n_functional_parameters = 3
    lambdas_ : ndarray of float of shape (n_features, n_years)
        The parameters of the power transformation for the selected features.
        Calculated for each feature using  lambda_function

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

    def fit_fmin(self, X, X_func, n_index):
        """Estimate the optimal parameter lambda for each feature.
        The optimal lambda parameter for minimizing skewness is estimated on
        each feature independently using maximum likelihood.
        Parameters
        ----------
        X : ndarray
            array-like, shape (n_samples, n_features)
            The data used to estimate the optimal transformation parameters.
        X_func : temp values to calculate lmbda as lmbda = a*y + b
        Returns
        -------
        self : object
        """
        X = X.copy()  # force copy so that fit does not change X inplace

        self.coeffs_ = np.array(
            Parallel(n_jobs=-1, verbose=False)(
                delayed(self._yeo_johnson_optimize_fmin)(
                    X[:, i_grid], X_func[:, i_grid]
                )
                for i_grid in np.arange(n_index)
            )
        )  # ,  desc = 'Optimizing Fmin:', position = 0, leave = True, file=sys.stdout)))
        # self.coeffs_ = np.array([self._yeo_johnson_optimize_fmin(X[:,i_grid], X_func[:,i_grid]) for i_grid in tqdm(np.arange(n_index),  desc = 'Optimizing Fmin:', position = 0, leave = True, file=sys.stdout)])
        # xarray: optim_function(col) for col in X.T
        # print(self.coeffs_.shape)
        self.mins_ = np.amin(X, axis=0)
        self.maxs_ = np.amax(X, axis=0)
        # print(self.coeffs_)

        if self.standardize:
            self._scaler = StandardScaler(copy=True)
            self._scaler.fit(X)

        return self

    def _yeo_johnson_optimize_fmin(self, x, x_func):
        """Find and return optimal lambda parameter of the Yeo-Johnson
        transform by MLE, for observed data x.
        Like for Box-Cox, MLE is done via the brent optimizer.
        """

        def _neg_log_likelihood(coeff):
            """Return the negative log likelihood of the observed data x as a
            function of lambda."""
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
