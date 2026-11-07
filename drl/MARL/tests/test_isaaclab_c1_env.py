import pytest


def test_c1_training_env_uses_real_isaaclab_direct_marl_vector_contract():
    """C1 mass training must use Isaac Lab vectorization, not PhysicalEpisode."""

    try:
        from drl.MARL.isaaclab_c1_env import (
            MSSRC1IsaacLabEnv,
            MSSRC1IsaacLabEnvCfg,
            make_c1_isaaclab_env_cfg,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "drl.MARL.isaaclab_c1_env":
            pytest.fail(
                "missing canonical Isaac Lab C1 training environment"
            )
        raise

    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
    )

    owners = tuple(
        f"module_{index}"
        for index in range(8)
    )

    cfg = make_c1_isaaclab_env_cfg(
        possible_agents=owners,
        num_envs=2,
        device="cuda:0",
    )

    # The training environment itself must be a genuine
    # Isaac Lab DirectMARLEnv.
    assert issubclass(
        MSSRC1IsaacLabEnv,
        DirectMARLEnv,
    )

    assert isinstance(
        cfg,
        MSSRC1IsaacLabEnvCfg,
    )

    assert isinstance(
        cfg,
        DirectMARLEnvCfg,
    )

    # C1 physics contract.
    assert cfg.sim.dt == pytest.approx(
        1.0 / 240.0
    )

    # One MARL decision every 8 physics ticks:
    # 240 / 8 = 30 Hz.
    assert cfg.decimation == 8

    # Native Isaac Lab vector environments.
    assert cfg.scene.num_envs == 2
    assert cfg.scene.replicate_physics is True

    # Eight physical module owners.
    assert tuple(
        cfg.possible_agents
    ) == owners

    assert set(
        cfg.action_spaces
    ) == set(owners)

    assert set(
        cfg.observation_spaces
    ) == set(owners)

    # The tensor ABI is deliberately minimal:
    # the real actor/critic inputs remain the existing
    # structured graph sidecars.
    assert all(
        cfg.action_spaces[owner] == 1
        for owner in owners
    )

    assert all(
        cfg.observation_spaces[owner] == 1
        for owner in owners
    )

    assert cfg.state_space == 1


def test_c1_isaaclab_env_is_concrete_physical_vector_environment():
    """The canonical C1 environment must implement Isaac Lab's physical hooks."""
    import inspect

    from drl.MARL.isaaclab_c1_env import (
        MSSRC1IsaacLabEnv,
    )

    assert not inspect.isabstract(
        MSSRC1IsaacLabEnv
    ), (
        "MSSRC1IsaacLabEnv must implement the "
        "DirectMARLEnv physical hooks"
    )

    required = {
        "_setup_scene",
        "_pre_physics_step",
        "_apply_action",
        "_get_observations",
        "_get_states",
        "_get_rewards",
        "_get_dones",
    }

    assert required.issubset(
        MSSRC1IsaacLabEnv.__dict__
    )


def test_c1_env_exposes_direct_per_env_physical_state_bridge():
    """Training state must come directly from Isaac PhysX, not files/ROS."""

    from drl.MARL.isaaclab_c1_env import (
        MSSRC1IsaacLabEnv,
    )

    required = {
        "_ensure_state_readers",
        "_read_joint_states",
    }

    assert required.issubset(
        MSSRC1IsaacLabEnv.__dict__
    )


def test_c1_env_exposes_in_memory_canonical_state_graph():
    """Canonical C1 graphs must be available without JSON/ROS transport."""

    from drl.MARL.isaaclab_c1_env import (
        MSSRC1IsaacLabEnv,
    )

    from smores_ep.isaac.state_graph_publisher import (
        SmoresStateGraphPublisher,
    )

    assert "snapshot" in (
        SmoresStateGraphPublisher.__dict__
    )

    required = {
        "_ensure_state_graph_builders",
        "_state_graph_snapshots",
    }

    assert required.issubset(
        MSSRC1IsaacLabEnv.__dict__
    )


def test_c1_env_reuses_existing_expert_planner_and_topology_verifier():
    """Isaac Lab must consume the existing deterministic Expert core."""

    from drl.MARL.isaaclab_c1_env import (
        MSSRC1IsaacLabEnv,
    )

    required = {
        "_current_attributed_graphs",
        "_load_target_graph",
        "_ensure_expert_planners",
        "_ensure_topology_matchers",
        "_plan_expert_targets",
        "_verify_target_topology",
    }

    assert required.issubset(
        MSSRC1IsaacLabEnv.__dict__
    )


def test_c1_env_reuses_validated_expert_spawn_and_physical_backend():
    """The native env must use the existing physical Expert backend."""

    import math

    from drl.MARL.isaaclab_c1_env import (
        MSSRC1IsaacLabEnv,
        make_c1_expert_spawn_layout,
    )

    owners = tuple(
        f"M{i}"
        for i in range(8)
    )

    layout = make_c1_expert_spawn_layout(
        owners
    )

    assert set(layout) == set(owners)

    # Same validated 8-module self-assembly layout:
    # one root candidate at the centroid.
    x, y, z, yaw = layout["M0"]

    assert x == 0.0
    assert y == 0.0
    assert z == 0.0316
    assert yaw == 0.0

    # Seven remaining modules on the validated 0.34 m ring.
    for owner in owners[1:]:
        x, y, z, _yaw = layout[owner]

        assert math.hypot(
            x,
            y,
        ) == pytest.approx(
            0.34
        )

        assert z == pytest.approx(
            0.0316
        )

    required = {
        "_ensure_physical_expert_backends",
        "_start_expert_assembly",
    }

    assert required.issubset(
        MSSRC1IsaacLabEnv.__dict__
    )


def test_c1_env_owns_inprocess_expert_assembly_runtime():
    """Expert assembly must execute inside the vector Isaac Lab process."""

    from drl.MARL.isaaclab_c1_env import (
        MSSRC1IsaacLabEnv,
    )

    required = {
        "_begin_expert_assembly_runtime",
        "_expert_assembly_control_tick",
        "_expert_assembly_runtime_complete",
    }

    assert required.issubset(
        MSSRC1IsaacLabEnv.__dict__
    )


def test_native_snapshot_reports_the_actuator_limits_applied_to_physx():
    from types import SimpleNamespace
    from smores_ep.config.physics import SmoresActuatorConfig
    from drl.MARL.isaaclab_c1_env import MSSRC1IsaacLabEnv

    drive = SmoresActuatorConfig.payload_overdrive()
    row = {'smores_01': SimpleNamespace(_drive=drive)}
    env = SimpleNamespace(
        state_graph_builders=None, _ensure_state_readers=lambda: (row, row),
        sim=SimpleNamespace(stage=None), num_envs=2,
        module_roots_by_env=({'smores_01': '/env0/m0'}, {'smores_01': '/env1/m0'}),
        docking_managers=(None, None),
    )
    publishers = MSSRC1IsaacLabEnv._ensure_state_graph_builders(env)
    for publisher in publishers:
        assert publisher._actuators['smores_01'] is drive


def test_rebuilding_selected_actuation_backend_keeps_sibling_state(monkeypatch):
    from types import SimpleNamespace
    from drl.MARL.isaaclab_c1_env import MSSRC1IsaacLabEnv
    import smores_ep.isaac.dynamic_stage as drives
    import smores_ep.isaac.command_router as routers
    import smores_ep.isaac.primitive_executor as primitives
    monkeypatch.setattr(drives,'DynamicDriveController',lambda *a:object())
    monkeypatch.setattr(routers,'IsaacMultiModuleCommandRouter',lambda *a:SimpleNamespace(reset_free_modules=lambda ids:None))
    monkeypatch.setattr(primitives,'IsaacPrimitiveExecutor',lambda *a,**k:SimpleNamespace())
    previous=(object(),object())
    env=SimpleNamespace(num_envs=2,physical_expert_backends=previous,
        _ensure_state_readers=lambda:({'m0':object()},{'m0':object()}),
        sim=SimpleNamespace(stage=object()),module_roots_by_env=({'m0':'/e0/m'},{'m0':'/e1/m'}),
        docking_managers=(object(),object()),cfg=SimpleNamespace(possible_agents=('m0',)))
    result=MSSRC1IsaacLabEnv._ensure_native_actuation_backends(env,reset_env_ids=(0,))
    assert result[0] is not previous[0]
    assert result[1] is previous[1]
    assert result[0]['primitive_executor'].reset_free_modules_callback is result[0]['router'].reset_free_modules
