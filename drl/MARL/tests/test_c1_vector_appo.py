from types import SimpleNamespace

import torch


def _transition(*, env_offset, terminated=False, truncated=False):
    owners = ("smores_01", "smores_02")

    modules = [
        {
            "module_id": owner,
            "position": [
                float(env_offset),
                float(index),
                0.05,
            ],
        }
        for index, owner in enumerate(owners)
    ]

    return SimpleNamespace(
        previous_state={
            "state": {
                "modules": modules,
            },
        },
        evaluation=SimpleNamespace(
            terminated=terminated,
            truncated=truncated,
        ),
    )


def _context():
    return SimpleNamespace(
        routing_snapshot={
            "module_refs": (
                SimpleNamespace(module_id="smores_01"),
                SimpleNamespace(module_id="smores_02"),
            ),
        },
    )


def test_record_learning_batch_preserves_owner_rewards_for_appo():
    from drl.MARL.c1_runner import record_learning_batch

    owners = (
        "smores_01",
        "smores_02",
    )

    captured = {}

    class Learner:
        possible_agents = owners

        def record_online_rollout_batch(self, **kwargs):
            captured.update(kwargs)

    learner = Learner()
    provider = SimpleNamespace(payload=None)

    step = SimpleNamespace(
        active_env_indices=(0, 1),
        transitions=(
            _transition(
                env_offset=0.0,
                terminated=False,
            ),
            _transition(
                env_offset=10.0,
                terminated=True,
            ),
        ),
        contexts=(
            _context(),
            _context(),
        ),
        decisions=(
            SimpleNamespace(name="decision_env_0"),
            SimpleNamespace(name="decision_env_1"),
        ),
        env_actions={
            owner: torch.tensor(
                [
                    [float(index + 1)],
                    [float(index + 11)],
                ],
                dtype=torch.float32,
            )
            for index, owner in enumerate(owners)
        },
        next_critic_graphs=(
            {"env": 0},
            {"env": 1},
        ),
    )

    scalar_rewards_by_env = {
        0: SimpleNamespace(
            team=10.0,
            owners={
                "smores_01": 1.0,
                "smores_02": 2.0,
            },
        ),
        1: SimpleNamespace(
            team=20.0,
            owners={
                "smores_01": 3.0,
                "smores_02": 4.0,
            },
        ),
    }

    args = SimpleNamespace(
        device="cpu",
    )

    record_learning_batch(
        args,
        learner,
        provider,
        step,
        scalar_rewards_by_env,
    )

    payload = provider.payload

    # APPO data: owner rewards remain distinct across both environments.
    assert torch.equal(
        payload.owner_rewards["smores_01"],
        torch.tensor(
            [
                [1.0],
                [3.0],
            ],
        ),
    )

    assert torch.equal(
        payload.owner_rewards["smores_02"],
        torch.tensor(
            [
                [2.0],
                [4.0],
            ],
        ),
    )

    # Mean-field/altruistic sidecar keeps one structural row per env.
    assert payload.module_ids_by_row == (
        (
            "smores_01",
            "smores_02",
        ),
        (
            "smores_01",
            "smores_02",
        ),
    )

    assert (
        payload.module_positions_by_id[0]["smores_01"][0]
        == 0.0
    )

    assert (
        payload.module_positions_by_id[1]["smores_01"][0]
        == 10.0
    )

    # MAPPO/skrl compatibility channel remains vectorized too,
    # but this is not a replacement for owner/mean-field APPO data.
    assert torch.equal(
        captured["team_rewards"],
        torch.tensor(
            [
                [10.0],
                [20.0],
            ],
        ),
    )

    assert captured["decisions"] == list(step.decisions)

    assert torch.equal(
        captured["terminated"],
        torch.tensor(
            [
                [False],
                [True],
            ],
            dtype=torch.bool,
        ),
    )

    assert torch.equal(
        captured["truncated"],
        torch.tensor(
            [
                [False],
                [False],
            ],
            dtype=torch.bool,
        ),
    )

    for owner in owners:
        assert captured["observations"][owner].shape == (2, 1)
        assert captured["states"][owner].shape == (2, 1)
        assert captured["next_observations"][owner].shape == (2, 1)
        assert captured["next_states"][owner].shape == (2, 1)
        assert captured["env_actions"][owner].shape[0] == 2



def test_c1_main_counts_and_records_two_active_env_transitions(
    tmp_path,
    monkeypatch,
):
    """C1 must consume the vector step as a batch, not silently use env 0."""
    import json

    import drl.MARL.c1_runner as runner
    import drl.MARL.phase1_vertical_slice as vertical
    import drl.MARL.reward_calibration as calibration
    import drl.MARL.reward_config as configuration

    owners = ("m0",)

    status = {
        "last_completed": {
            "outcome": "accepted",
            "command_id": "bc:vector:completed",
        },
        "last_applied": {
            "command_id": "bc:vector:applied",
        },
    }

    class Episode:
        def __init__(
            self,
            args,
            domain,
            seed,
            episode_id,
            bridge,
        ):
            self.domain = domain
            self.seed = seed
            self.episode_id = episode_id
            self.owners = owners
            self.env = object()
            self.context = object()
            self.synchronizer = object()
            self.availability = {}
            self.transport = SimpleNamespace(
                status=lambda: status,
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    from drl.MARL.direct_marl_env import DirectMARLTransition

    def make_transition(
        env_index,
    ):
        reward_marker = float(
            env_index + 1
        )

        # Distinct physical clocks make any shared episode_start_stamp
        # immediately visible as a vectorization bug.
        start_stamp = (
            100.0
            if env_index == 0
            else 200.0
        )

        return DirectMARLTransition(
            env_index=env_index,
            previous_state={
                "stamp": start_stamp,
            },
            actuation_result=SimpleNamespace(
                command_id=f"bc:vector:{env_index}",
                proposal_diagnostics={
                    "env_index": env_index,
                },
                decoded=(object(),),
            ),
            physics_result=SimpleNamespace(
                vector_step=1,
            ),
            next_state={
                "stamp": start_stamp + 0.25,
            },
            evaluation=SimpleNamespace(
                raw_reward=SimpleNamespace(
                    marker=reward_marker,
                    team=SimpleNamespace(
                        success=True,
                    ),
                ),
                terminated=True,
                truncated=False,
                success=True,
                failure_reason=None,
            ),
            episode_step=1,
        )

    class Slice:
        def __init__(self, **kwargs):
            self.calls = 0

        def step(self):
            self.calls += 1

            if self.calls > 1:
                return None

            return SimpleNamespace(
                active_env_indices=(0, 1),
                transitions=(
                    make_transition(0),
                    make_transition(1),
                ),
                contexts=(
                    SimpleNamespace(
                        actor_graph={
                            "x": torch.tensor([0.0]),
                        },
                        critic_graph={
                            "x": torch.tensor([0.0]),
                        },
                    ),
                    SimpleNamespace(
                        actor_graph={
                            "x": torch.tensor([1.0]),
                        },
                        critic_graph={
                            "x": torch.tensor([1.0]),
                        },
                    ),
                ),
                decisions=(
                    SimpleNamespace(
                        owner_log_probs={
                            "m0": torch.tensor(0.0),
                        },
                        global_value=torch.tensor(0.0),
                    ),
                    SimpleNamespace(
                        owner_log_probs={
                            "m0": torch.tensor(0.0),
                        },
                        global_value=torch.tensor(0.0),
                    ),
                ),
            )

    class Recorder:
        def __init__(self, *args, **kwargs):
            self.rows = []

        def observe(self, transition, **kwargs):
            self.rows.append(
                transition
            )

    memory = []
    learner = SimpleNamespace(
        memories={
            "m0": memory,
        },
        possible_agents=owners,
    )

    vector_calls = []
    legacy_calls = []
    updates = []

    def record_batch(
        args,
        learner_arg,
        provider,
        step,
        scalar_rewards_by_env,
    ):
        vector_calls.append(
            {
                "active_env_indices":
                    tuple(
                        step.active_env_indices
                    ),
                "rewards":
                    dict(
                        scalar_rewards_by_env
                    ),
            }
        )

        # Mimic one vector rollout write so final flush can update.
        memory.append(
            "vector batch"
        )

    def record_legacy(*args, **kwargs):
        legacy_calls.append(
            True
        )
        memory.append(
            "legacy row"
        )

    def fake_update(
        args,
        learner_arg,
        transitions,
        update_number,
    ):
        updates.append(
            (
                transitions,
                update_number,
            )
        )
        memory.clear()
        return {}

    config = SimpleNamespace(
        scalarize=lambda domain, raw_reward, availability:
            SimpleNamespace(
                team=10.0 * raw_reward.marker,
                owners={
                    "m0":
                        raw_reward.marker,
                },
            )
    )

    monkeypatch.setattr(
        runner,
        "build_observation_bridge",
        lambda root: object(),
    )

    monkeypatch.setattr(
        runner,
        "build_models",
        lambda *args: (
            object(),
            object(),
        ),
    )

    monkeypatch.setattr(
        runner,
        "selected_layouts",
        lambda args: [
            {
                "domain": "navigation",
                "seed": 7000,
                "split": "train",
                "difficulty": "easy",
            }
        ],
    )

    monkeypatch.setattr(
        runner,
        "PhysicalEpisode",
        Episode,
    )

    monkeypatch.setattr(
        vertical,
        "Phase1VerticalSliceRunner",
        Slice,
    )

    monkeypatch.setattr(
        calibration,
        "TransitionRecorder",
        Recorder,
    )

    monkeypatch.setattr(
        runner,
        "episode_manifest",
        lambda *args, **kwargs: {},
    )

    monkeypatch.setattr(
        runner,
        "create_learner",
        lambda *args, **kwargs:
            (
                learner,
                SimpleNamespace(
                    payload=None,
                ),
            ),
    )

    monkeypatch.setattr(
        runner,
        "record_learning_batch",
        record_batch,
    )

    monkeypatch.setattr(
        runner,
        "record_learning_step",
        record_legacy,
    )

    monkeypatch.setattr(
        runner,
        "update_learner",
        fake_update,
    )

    monkeypatch.setattr(
        configuration.RewardConfig,
        "load",
        lambda path: config,
    )

    monkeypatch.setattr(
        runner.signal,
        "signal",
        lambda *args: None,
    )

    rc = runner.main(
        [
            "train",
            "--output",
            str(tmp_path),
            "--reward-config",
            str(
                tmp_path
                / "unused_reward_config.json"
            ),
            "--steps-per-layout",
            "1",
            "--total-transitions",
            "1",
            "--rollout-transitions",
            "64",
        ]
    )

    assert rc == 0

    # The vector step contains two genuine physical transitions.
    result = json.loads(
        (
            tmp_path
            / "result.json"
        ).read_text()
    )

    assert (
        result[
            "physical_transitions"
        ]
        == 2
    )

    assert (
        result[
            "transition_counts"
        ][
            "navigation"
        ]
        == 2
    )

    # Both terminated envs are accounted for independently.
    assert result[
        "episodes"
    ] == 2

    outcomes = [
        json.loads(line)
        for line
        in (
            tmp_path
            / "episode_outcomes.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]

    assert {
        row["env_index"]:
            (
                row["start_stamp"],
                row["next_stamp"],
            )
        for row in outcomes
    } == {
        0: (
            100.0,
            100.25,
        ),
        1: (
            200.0,
            200.25,
        ),
    }

    # The legacy single-row APPO bridge must disappear from main().
    assert legacy_calls == []

    # Exactly one APPO vector recording call for one physical vector step.
    assert len(
        vector_calls
    ) == 1

    call = vector_calls[
        0
    ]

    assert call[
        "active_env_indices"
    ] == (
        0,
        1,
    )

    # Owner/team scalarization remains separate for each env.
    assert (
        call[
            "rewards"
        ][
            0
        ].team
        == 10.0
    )

    assert (
        call[
            "rewards"
        ][
            1
        ].team
        == 20.0
    )

    assert (
        call[
            "rewards"
        ][
            0
        ].owners[
            "m0"
        ]
        == 1.0
    )

    assert (
        call[
            "rewards"
        ][
            1
        ].owners[
            "m0"
        ]
        == 2.0
    )

    # Final partial rollout update is labelled by physical transitions,
    # not number of runner.step() calls.
    assert updates == [
        (
            2,
            1,
        )
    ]
