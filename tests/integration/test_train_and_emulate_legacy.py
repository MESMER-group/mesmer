import os
import sys

EXAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples"
)


sys.path.append(EXAMPLE_PATH)


def test_train_and_emulate_legacy():

    # load in configurations used in this script
    import config_across_scen_T_cmip6ng_test as cfg
    from train_create_emus_automated import main

    # TODO: update paths
    # cfg.EXAMPLE_OUTPUT_ROOT = os.path.join(cfg.MESMER_ROOT, "examples", "output")
    # cfg.dir_aux = os.path.join(cfg.EXAMPLE_OUTPUT_ROOT, "auxillary")
    # cfg.dir_mesmer_params = os.path.join(cfg.EXAMPLE_OUTPUT_ROOT, "calibrated_parameters")
    # cfg.dir_mesmer_emus = os.path.join(cfg.EXAMPLE_OUTPUT_ROOT, "emulations")

    main(cfg=cfg)
