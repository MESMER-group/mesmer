import numpy as np
import xarray as xr

# from mesmer.core.utils import (
#     _check_dataarray_form,
#     _check_dataset_form
# )

# ToDo: Add inverse transform & StandardScaling prior to transforming 

from sklearn.preprocessing import StandardScaler

def fit_principal_components(
    X: xr.DataArray,
    n_components = None
):
    """
    Fit a principal component decomposition

    Parameters
    ----------
    X : xr.DataArray
        DataArray to decompose. Must be 2D. PCA is tranformed over the second dimension. 
    """
    
    if n_components == None: 
        n_components = X.values.shape[1]

    params = _fit_principal_component_decomposition_xr(
        X = X,
        n_components = n_components
    )

    return(params)

def _fit_principal_component_decomposition_xr(
    X: xr.DataArray,
    n_components: int, 
) -> xr.Dataset:
    """
    Fit a principal component decomposition

    Parameters
    ----------
    X : xr.DataArray
        DataArray to decompose. Must be 2D. 

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset of projection coefficients.
    """
    
    X_np = X.values
    
    std = StandardScaler().fit(X_np)
    X_np_std = std.transform(X_np)

    from sklearn.decomposition import PCA as PCA_np

    pca = PCA_np(n_components = n_components).fit(X_np_std)
    
    params = xr.Dataset({'coeffs': (('component', X.dims[1]), pca.components_), 
                        'mean': (X.dims[1], pca.mean_),
                        'explained_variance' : ('component', pca.explained_variance_),
                        'std_scale': (X.dims[1], std.scale_), 
                        'std_mean': (X.dims[1], std.mean_), 
                        'std_var': (X.dims[1], std.var_)},
                        coords = {X.dims[1]: X[X.dims[1]],
                                  'component': np.arange(n_components)})
    
    return(params)


def transform_principal_components(
    X: xr.DataArray, 
    params: xr.DataArray,
) -> xr.DataArray:
    """
    Project input data onto eigenspace.

    Parameters
    ----------
    X : xr.DataArray 
        DataArray to project onto eigenspace given the previously computed components.

    Returns
    -------
    T : xr.DataArray
        Vector of principal components.
    """

    T = _transform_principal_component_decomposition_xr(
        X = X, 
        params = params
    )
    
    return T

def _transform_principal_component_decomposition_xr(
    X: xr.DataArray,
    params: xr.DataArray
) -> xr.DataArray:

    from sklearn.decomposition import PCA as PCA_np
    
    std = StandardScaler()
    std.n_features_in_ = len(params['std_scale'])
    std.scale_ = params['std_scale'].values
    std.mean_ = params['std_mean'].values
    std.var_ = params['std_var'].values
    
    pca = PCA_np()
    pca.n_components_ = params['coeffs'].shape[0]
    pca.components_ = params['coeffs'].values
    pca.mean_ = params['mean'].values
    pca.explained_variance_ = params['explained_variance'].values
    
    X_trans_np = pca.transform(std.transform(X.values))
    
    X_trans = xr.DataArray(X_trans_np, 
                           dims = [X.dims[0], 'component'], 
                           coords = {X.dims[0]: X[X.dims[0]], 
                                     'component': np.arange(pca.n_components_)
                                     }
                           )
    return(X_trans)

def inverse_transform_principal_components(
    T: xr.DataArray, 
    params: xr.DataArray,
) -> xr.DataArray:
    """
    Project input data onto eigenspace.

    Parameters
    ----------
    X : xr.DataArray 
        DataArray to project onto eigenspace given the previously computed components.

    Returns
    -------
    T : xr.DataArray
        Vector of principal components.
    """

    T = _inverse_transform_principal_component_decomposition_xr(
        T = T, 
        params = params
    )
    
    return T

def _inverse_transform_principal_component_decomposition_xr(
    T: xr.DataArray,
    params: xr.DataArray
) -> xr.DataArray:

    from sklearn.decomposition import PCA as PCA_np
    
    pca = PCA_np()
    pca.n_components_ = params['coeffs'].shape[0]
    pca.components_ = params['coeffs'].values
    pca.mean_ = params['mean'].values
    pca.explained_variance_ = params['explained_variance'].values
    
    X_np_std = pca.inverse_transform(T.values)
    
    std = StandardScaler()
    std.n_features_in_ = len(params['std_scale'])
    std.scale_ = params['std_scale'].values
    std.mean_ = params['std_mean'].values
    std.var_ = params['std_var'].values
    
    X_np = std.inverse_transform(X_np_std)
    
    X = xr.DataArray(X_np, 
                    dims = [T.dims[0], 'gridcell'], 
                    coords = {T.dims[0]: T[T.dims[0]], 
                              params['coeffs'].dims[1]: params['coeffs'][params['coeffs'].dims[1]]
                                }
                    )
    return(X)