# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import joblib
import xarray as xr


def save_mesmer_bundle(
    bundle_file,
    params_lt,
    params_lv,
    params_gv,
    land_fractions,
    lat,
    lon,
):
    """
    Save all the information required to draw MESMER emulations to disk

    Parameters
    ----------
    bundle_file : str
        file in which to save the bundle
    params_lt : dict
        dictionary containing the calibrated parameters for the local trends emulations,
        keys relevant here:

        - ["targs"] (list of emulated variables, list of strs)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - [xx] (additional keys depending on employed method)
    params_lv : dict
        dictionary containing the calibrated parameters for the local variability
        emulations, keys relevant here

        - ["targs"] (list of variables which are emulated, list of strs)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - [xx] (additional keys depending on employed method)
    params_gv : dict
        dictionary containing the calibrated parameters for the global variability emulations, keys relevant here

        - ["targ"] (variable which is emulated, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (scenarios which are used for training, list of strs)
        - [xx] (additional keys depending on employed method)
    land_fractions : np.MaskedArray
        data containing land fractions (also used for helping generate output on lat-lon
        grids)
    lat : np.ndarray
        grid latitudes (used to check land_fractions shape)
    lon : np.ndarray
        grid longitudes (used to check land_fractions shape)
    """
    assert land_fractions.shape[0] == lat.shape[0]
    assert land_fractions.shape[1] == lon.shape[0]

    # hopefully right way around
    land_fractions = xr.DataArray(
        land_fractions, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
    )

    mesmer_bundle = {
        "params_lt": params_lt,
        "params_lv": params_lv,
        "params_gv": params_gv,
        "land_fractions": land_fractions,
    }
    joblib.dump(mesmer_bundle, bundle_file)
