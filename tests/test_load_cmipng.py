from mesmer.io._load_cmipng import _load_cmipng_var


def mock_cfg(type="all", start=1850, end=1950, gen=6):

    ref = dict(type=type, start=start, end=end)

    class cfg:
        def __init__(self):
            self.gen = gen
            self.ref = ref
            self.dir_cmipng = ""

    return cfg()


def test_load_cmipng_var_missing_data():

    dta, dta_global, lon, lat, time = _load_cmipng_var(
        esm="missing",
        scen="",
        cfg=mock_cfg(),
        varn="tas",
    )

    assert dta is None
    assert dta_global is None
    assert lon is None
    assert lat is None
    assert time is None
