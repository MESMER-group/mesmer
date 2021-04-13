import joblib
import xarray as xr


def save_mesmer_bundle(
    bundle_file,
    params_lt,
    params_lv,
    params_gv_T,
    seeds,
    land_fractions,
    lat,
    lon,
    time,
):
    """
    Save all the information required to draw MESMER emulations to disk

    TODO: parameters
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
        "params_gv_T": params_gv_T,
        "seeds": seeds,
        "land_fractions": land_fractions,
        "time": time,
    }
    joblib.dump(mesmer_bundle, bundle_file)
