from types import SimpleNamespace


def _goal(goal_id):
    return SimpleNamespace(goal_id=goal_id)


class _Executor:
    def __init__(self):
        self.active_goals = ()
        self.prepared = ()
        self.seals = 0

    def prepared_relation_snapshot(self):
        return self.prepared

    def seal_active_align_wave(self):
        self.seals += 1
        return (("sealed",),)


class _Env:
    num_envs = 2

    def __init__(self):
        self.executors = [_Executor(), _Executor()]
        self.native_dispatcher = SimpleNamespace(
            executor_provider=lambda index: self.executors[index]
        )
        self.docking_managers = [
            SimpleNamespace(connections=[]),
            SimpleNamespace(connections=[]),
        ]
        self.generation = [0, 0]


def test_training_option_controller_sleeps_only_the_busy_env_after_wave_fill_seals():
    from drl.MARL.c1_training_options import NativeStructuralOptionController

    env = _Env()
    ctl = NativeStructuralOptionController(env, mode="event")

    assert ctl.decision_mask() == (True, True)

    env.executors[0].active_goals = (_goal("align-a"),)
    # First member joined: one more decision is allowed for wave fill.
    assert ctl.decision_mask() == (True, False)

    # No new member joins after that decision: env0 seals and sleeps.
    assert ctl.decision_mask() == (False, False)
    assert env.executors[0].seals == 1

    # A sibling env remains independent; resetting its generation reopens it.
    env.generation[1] += 1
    assert ctl.decision_mask() == (False, True)


def test_training_option_controller_reopens_on_prepared_handoff_and_keeps_resource_fixed_rate():
    from drl.MARL.c1_training_options import NativeStructuralOptionController

    env = _Env()
    ctl = NativeStructuralOptionController(env, mode="event")
    assert ctl.decision_mask() == (True, True)

    env.executors[0].active_goals = (_goal("align-a"),)
    assert ctl.decision_mask()[0]
    assert not ctl.decision_mask()[0]

    env.executors[0].active_goals = ()
    env.executors[0].prepared = (
        SimpleNamespace(
            source_module="smores_04",
            source_face="BOTTOM",
            target_module="smores_05",
            target_face="LEFT",
            docked=False,
        ),
    )
    assert ctl.decision_mask()[0]
    # Pending handoff stays hot until another option starts.
    assert ctl.decision_mask()[0]

    ctl.set_continuous_control(0, True)
    env.executors[0].active_goals = (_goal("resource-or-structural"),)
    assert ctl.decision_mask()[0]
    assert ctl.decision_mask()[0]
    ctl.set_continuous_control(0, False)


def test_training_option_controller_fixed_mode_preserves_every_frame_policy():
    from drl.MARL.c1_training_options import NativeStructuralOptionController

    env = _Env()
    ctl = NativeStructuralOptionController(env, mode="fixed")
    env.executors[0].active_goals = (_goal("align-a"),)
    for _ in range(4):
        assert ctl.decision_mask() == (True, True)


def test_native_runner_exposes_event_structural_scheduler_by_default():
    from drl.MARL.isaaclab_c1_runner import build_parser

    args = build_parser().parse_args(["--reward-config", "/unused/reward.json"])
    assert args.structural_scheduler == "event"


def test_native_provider_event_scheduler_marks_option_sleep_rows_inactive_and_neutral():
    import torch
    from drl.MARL.native_policy_bridge import build_native_policy_bridge
    from drl.MARL.tests.test_native_policy_bridge import (
        NativeEnv, ObservationBridge, active_handoff,
    )

    env = NativeEnv()
    env.generation = [0, 0]
    env.executors = [_Executor(), _Executor()]
    env.native_dispatcher = SimpleNamespace(
        executor_provider=lambda index: env.executors[index]
    )
    env.docking_managers = [SimpleNamespace(connections=[]), SimpleNamespace(connections=[])]
    observations = ObservationBridge()
    decoded = []

    class Binding:
        def decode_marl_action(self, **kwargs):
            decoded.append(kwargs)
            return (SimpleNamespace(command={
                "command": "wheel_velocity", "module_ids": ["m0"], "parameters": {}
            }),)

    bridge = build_native_policy_bridge(
        env,
        observation_bridge=observations,
        task_context_provider=lambda index: index,
        binding=Binding(),
        handoffs=[active_handoff(), active_handoff()],
        hard_mask_builder=lambda actor, routing: torch.tensor([False]),
        placeholder_action=lambda index, context: {"m0": torch.zeros(1)},
        post_step_observation_provider=lambda index: observations.prepare_snapshot(
            {"stamp": env.stamp, "env": index}, task_context=index
        ),
        structural_scheduler="event",
    )
    decision = SimpleNamespace(action=object(), action_set_prediction_template=object())

    assert [c.rollout_active for c in bridge.provider.current_action_contexts()] == [True, True]
    bridge.provider.encode_env_actions([decision, decision])

    env.stamp = 2.0
    env.executors[0].active_goals = (_goal("align-a"),)
    bridge.provider.current_critic_graphs()
    assert [c.rollout_active for c in bridge.provider.current_action_contexts()] == [True, False]
    bridge.provider.encode_env_actions([decision, None])

    env.stamp = 3.0
    bridge.provider.current_critic_graphs()
    contexts = bridge.provider.current_action_contexts()
    assert [c.rollout_active for c in contexts] == [False, False]
    bridge.provider.encode_env_actions([None, None])
    assert env.executors[0].seals == 1
    # 2 initial decisions + only env0 on the wave-fill frame; sleeping rows are neutral.
    assert len(decoded) == 3


def test_event_scheduler_wave_sealing_is_morphology_agnostic():
    from drl.MARL.c1_training_options import NativeStructuralOptionController

    for morphology in ("rc_car8", "snake8", "mobile_manipulator8"):
        env = _Env()
        env.tasks = [SimpleNamespace(morphology=morphology), SimpleNamespace(morphology=morphology)]
        ctl = NativeStructuralOptionController(env, mode="event")
        assert ctl.decision_mask() == (True, True)
        env.executors[0].active_goals = (_goal(f"{morphology}-align"),)
        assert ctl.decision_mask()[0] is True
        assert ctl.decision_mask()[0] is False
        assert env.executors[0].seals == 1
