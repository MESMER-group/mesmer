def test_make_realisations(test_mesmer_bundle):
    # TODO: split out load_mesmer_bundle function

    params_lt = test_mesmer_bundle["params_lt"]
    params_lv = test_mesmer_bundle["params_lv"]
    preds_lv = test_mesmer_bundle["preds_lv"]
    seeds = test_mesmer_bundle["seeds"]
    land_fractions = test_mesmer_bundle["land_fractions"]
    # TODO: make this something which isn't defined by the bundle
    n_realisations = test_mesmer_bundle["n_realisations"]

    import pdb
    pdb.set_trace()
    # format as needed

    res = make_realisations(
        preds_lt,
        params_lt,
        preds_lv,
        params_lv,
        n_realisations,
        seeds,
        land_fractions,
    )
