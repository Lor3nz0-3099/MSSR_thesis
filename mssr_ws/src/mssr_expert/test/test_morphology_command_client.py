from __future__ import annotations

import pytest

from mssr_expert.nodes.smores_morphology_command_client import (
    build_argument_parser,
    build_command_payload,
    parse_parameters_json,
)


def test_parameters_are_parsed_as_one_json_object() -> None:
    assert parse_parameters_json('{"height_m": 0.12}') == {
        "height_m": 0.12
    }


@pytest.mark.parametrize("text", ("", "[]", "{bad json}"))
def test_invalid_parameter_payload_is_rejected_before_ros_publish(
    text: str,
) -> None:
    with pytest.raises(ValueError):
        parse_parameters_json(text)


def test_command_payload_preserves_nonempty_identifier() -> None:
    payload = build_command_payload(
        command_id="bridge-cross-suite-01",
        morphology="bridge8",
        behavior="cross_gap",
        parameters={"linear_m_s": 0.012},
    )

    assert payload["command_id"] == "bridge-cross-suite-01"
    assert payload["parameters"] == {"linear_m_s": 0.012}


def test_empty_command_identifier_is_rejected_before_ros_publish() -> None:
    with pytest.raises(ValueError, match="command_id"):
        build_command_payload(
            command_id="",
            morphology="bridge8",
            behavior="cross_gap",
            parameters={},
        )


def test_command_client_waits_indefinitely_by_default() -> None:
    parsed = build_argument_parser().parse_args(
        [
            "--morphology",
            "snake8",
            "--behavior",
            "crawl_stairs",
            "--command-id",
            "stair-test",
        ]
    )

    assert parsed.timeout_s == 0.0


def test_command_client_accepts_an_explicit_finite_timeout() -> None:
    parsed = build_argument_parser().parse_args(
        [
            "--morphology",
            "snake8",
            "--behavior",
            "crawl_stairs",
            "--command-id",
            "stair-test",
            "--timeout-s",
            "600",
        ]
    )

    assert parsed.timeout_s == 600.0
