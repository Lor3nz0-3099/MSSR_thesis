from pathlib import Path


LAUNCH = (
    Path(__file__)
    .resolve()
    .parents[1]
    / "launch"
    / "smores_runtime.launch.py"
)


def _source():
    return LAUNCH.read_text(
        encoding="utf-8"
    )


def test_launch_declares_manipulation_course_arguments():
    source = _source()

    for name in (
        "phase1_manipulation_course",
        "phase1_manipulation_split",
        "phase1_manipulation_seed",
    ):
        assert (
            f'"{name}"'
            in source
        )


def test_launch_forwards_manipulation_course_to_isaac_cli():
    source = _source()

    assert (
        '"--phase1-manipulation-course"'
        in source
    )

    assert (
        '"--phase1-manipulation-split"'
        in source
    )

    assert (
        '"--phase1-manipulation-seed"'
        in source
    )

    compact = "".join(
        source.split()
    )

    assert (
        'LaunchConfiguration("phase1_manipulation_split")'
        in compact
    )

    assert (
        'LaunchConfiguration("phase1_manipulation_seed")'
        in compact
    )


def test_manipulation_course_enable_controls_forwarding():
    source = _source()

    assert (
        '_as_bool('
        'LaunchConfiguration('
        '"phase1_manipulation_course"'
        ')'
        '.perform(context)'
        ')'
        in "".join(
            source.split()
        )
    )
