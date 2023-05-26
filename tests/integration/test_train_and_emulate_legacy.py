import os
import sys

EXAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples"
)


sys.path.append(EXAMPLE_PATH)


def test_train_and_emulate_legacy():

    # load in configurations used in this script
    import config_across_scen_T_cmip6ng_test
    from train_create_emus_automated import main

    main(cfg=config_across_scen_T_cmip6ng_test)
