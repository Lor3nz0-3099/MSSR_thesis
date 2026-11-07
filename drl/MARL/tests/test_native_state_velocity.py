"""UNIT velocity source contract; no simulator or calibration data."""
from smores_ep.isaac.state_graph_publisher import SmoresStateGraphPublisher


def publisher(**kwargs):
    return SmoresStateGraphPublisher(stage=None, module_roots={"m0": "/m0"},
        states={"m0": object()}, docking=object(), **kwargs)


def test_native_measured_velocity_survives_repeated_same_stamp_and_reset_pose():
    calls = []
    measured = ((.3, -.2, .1), (0., .4, -.5))

    def velocity_provider(owner):
        calls.append(owner)
        return measured

    builder = publisher(velocity_provider=velocity_provider)
    # Repeated captures, including a reset teleport at the same physics stamp,
    # must report measured velocity rather than finite-difference artifacts.
    assert builder._velocity("m0", 1., (0., 0., 0.), 0.) == measured
    assert builder._velocity("m0", 1., (10., 0., 0.), 1.) == measured
    assert calls == ["m0", "m0"]
    assert builder._last_sample == {}


def test_legacy_default_keeps_existing_finite_difference_contract():
    builder = publisher()
    zero = ((0., 0., 0.), (0., 0., 0.))
    assert builder._velocity("m0", 1., (0., 0., 0.), 0.) == zero
    assert builder._velocity("m0", 2., (1., 2., 3.), .5) == ((1., 2., 3.), (0., 0., .5))
    assert builder._velocity("m0", 2., (1., 2., 3.), .5) == zero
