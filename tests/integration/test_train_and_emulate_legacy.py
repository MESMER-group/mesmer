import importlib
import sys

EXAMPLE_PATH = importlib.resources.files("mesmer").parent / "examples"


sys.path.append(str(EXAMPLE_PATH))


def test_train_and_emulate_legacy():

    # load in configurations used in this script
    import config_tas_cmip6ng_example as cfg
    from train_create_emus_automated import main

    # TODO: update paths
    # cfg.EXAMPLE_OUTPUT_ROOT = os.path.join(cfg.MESMER_ROOT, "examples", "output")
    # cfg.dir_aux = os.path.join(cfg.EXAMPLE_OUTPUT_ROOT, "auxillary")
    # cfg.dir_mesmer_params = os.path.join(cfg.EXAMPLE_OUTPUT_ROOT, "calibrated_parameters")
    # cfg.dir_mesmer_emus = os.path.join(cfg.EXAMPLE_OUTPUT_ROOT, "emulations")

    main(cfg=cfg)
