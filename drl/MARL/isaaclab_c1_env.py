"""Canonical Isaac Lab environment contract for C1 MARL training.

This module is intentionally the training boundary.

Isaac Lab owns:
    - native vector environments
    - simulation stepping
    - reset dispatch
    - SKRL environment ABI

Existing MSSR code owns:
    - structured graph observations
    - ActionSet semantics
    - SMORES actuation
    - C1 rewards and termination
    - Expert assembly
    - MSSRMAPPO / fixed-alpha APPO

The physical implementation is added incrementally behind this contract.
"""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from drl.MARL.isaaclab_c1_tasks import NativeC1TaskHooks


C1_PHYSICS_HZ = 240
C1_DECIMATION = 8
C1_MODULE_COUNT = 8
C1_ENV_SPACING_M = 1.5


@configclass
class MSSRC1IsaacLabEnvCfg(DirectMARLEnvCfg):
    """Isaac Lab configuration for vectorized C1 training."""

    domains: tuple[str, ...] = ("navigation",)
    seeds: tuple[int, ...] = ()
    reward_config_path: str = "logs/appo_smoke/calibration_validated_train/reward_config.initial.json"
    shaping_gamma: float = 0.99
    resample_train: bool = False
    train_sampler_seed: int = 0
    train_seed_pools: dict | None = None
    smoke_seed_cycle: bool = False
    course_split: str = 'train'
    # Default preserves the old explicitly Expert-driven validation path.
    initialization_mode: str = 'expert_validation'
    c1_probe_config_path: str | None = None
    c1_evidence_directory: str | None = None
    snapshot_bank_path: str | None = None
    snapshot_stage_weights: dict | None = None
    # Explicit viewer-only optimization; training keeps native live reads.
    c1_readonly_state_cache: bool = False

    # 240 Hz physics / 8 ticks = 30 Hz policy decisions.
    decimation = C1_DECIMATION

    # MARL time only: deterministic assembly does not consume this horizon.
    episode_length_s = 60.0

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / C1_PHYSICS_HZ,
        device="cuda:0",
        render_interval=C1_DECIMATION,
        physics=PhysxCfg(enable_ccd=True),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=C1_ENV_SPACING_M,
        replicate_physics=True,
    )

    # DirectMARLEnv multi-agent ABI.
    #
    # These fixed-shape tensors are deliberately only the SKRL/environment
    # carrier. MSSRMAPPO keeps the true structured graph/action information
    # in the already existing runtime provider and rollout sidecars.
    possible_agents: list[str] = []

    observation_spaces: dict[str, int] = {}
    action_spaces: dict[str, int] = {}

    state_space: int = 1


def make_c1_isaaclab_env_cfg(
    *,
    possible_agents: Sequence[str],
    num_envs: int,
    device: str = "cuda:0",
) -> MSSRC1IsaacLabEnvCfg:
    """Build the native vector environment configuration for C1."""

    owners = tuple(
        str(owner)
        for owner in possible_agents
    )

    if len(owners) != C1_MODULE_COUNT:
        raise ValueError(
            "C1 requires exactly eight physical module owners"
        )

    if (
        len(set(owners)) != len(owners)
        or any(not owner for owner in owners)
    ):
        raise ValueError(
            "C1 module owner IDs must be unique and non-empty"
        )

    if (
        isinstance(num_envs, bool)
        or not isinstance(num_envs, int)
        or num_envs <= 0
    ):
        raise ValueError(
            "num_envs must be a positive integer"
        )

    if not isinstance(device, str) or not device:
        raise ValueError(
            "device must be a non-empty string"
        )

    cfg = MSSRC1IsaacLabEnvCfg()

    cfg.possible_agents = list(
        owners
    )

    cfg.observation_spaces = {
        owner: 1
        for owner in owners
    }

    cfg.action_spaces = {
        owner: 1
        for owner in owners
    }

    cfg.state_space = 1

    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = (
        C1_ENV_SPACING_M
    )
    cfg.scene.replicate_physics = True

    cfg.sim.dt = (
        1.0
        / C1_PHYSICS_HZ
    )

    cfg.sim.device = device

    cfg.decimation = (
        C1_DECIMATION
    )

    return cfg


def make_c1_expert_spawn_layout(
    possible_agents,
):
    """Return the validated loose-module self-assembly spawn layout."""

    import math

    owners = tuple(
        str(owner)
        for owner in possible_agents
    )

    if len(owners) != C1_MODULE_COUNT:
        raise ValueError(
            "C1 Expert spawn requires exactly eight modules"
        )

    if (
        len(set(owners)) != len(owners)
        or any(not owner for owner in owners)
    ):
        raise ValueError(
            "C1 Expert module IDs must be unique and non-empty"
        )

    from types import SimpleNamespace
    from smores_ep.scenarios.parallel_self_assembly import radial_spawn_layout
    return radial_spawn_layout(SimpleNamespace(module_ids=owners,spawn_height_m=.0316,spawn_radius_m=.34))



class MSSRC1IsaacLabEnv(NativeC1TaskHooks, DirectMARLEnv):
    """Native Isaac Lab vector environment for MSSR C1.

    This task establishes only the physical substrate:

    - one Isaac Lab SimulationContext;
    - N native vector environments;
    - eight physical SMORES modules per environment;
    - one independent docking manager per environment.

    C1 Expert/reset, graph observations, semantic MARL actuation,
    rewards and termination are connected in later tasks.
    """

    def _setup_scene(self):
        from pathlib import Path

        from pxr import Gf, UsdGeom

        from smores_ep.config.geometry import (
            SmoresGeometry,
        )
        from smores_ep.isaac.docking import (
            IsaacDockingManager,
        )

        stage = self.sim.stage

        repository_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        module_usd = (
            repository_root
            / "assets"
            / "smores-ep"
            / "usd_physics"
            / "smores_ep_physics_v1.usd"
        )

        if not module_usd.is_file():
            raise FileNotFoundError(
                f"SMORES physics USD missing: {module_usd}"
            )

        geometry = SmoresGeometry()

        # Use the actual deterministic self-assembly spawn,
        # not the temporary pair layout used only by Task 2.
        expert_layout = (
            make_c1_expert_spawn_layout(
                self.cfg.possible_agents
            )
        )

        self._spawn_layout = expert_layout
        source_root = "/World/envs/env_0"

        for index, owner in enumerate(
            self.cfg.possible_agents
        ):
            prim_path = (
                f"{source_root}/SMORES_{index}"
            )

            prim = stage.DefinePrim(
                prim_path,
                "Xform",
            )

            prim.GetReferences().AddReference(
                str(module_usd)
            )

            (
                x_m,
                y_m,
                z_m,
                yaw_deg,
            ) = expert_layout[owner]

            from smores_ep.isaac.multi_module_stage import set_module_pose
            set_module_pose(stage,prim_path,(x_m,y_m,z_m),yaw_deg=yaw_deg)

        from drl.MARL.c1_physics_profile import configure_c1_module_stage
        configure_c1_module_stage(stage, {
            owner: f'{source_root}/SMORES_{index}'
            for index, owner in enumerate(self.cfg.possible_agents)
        })

        # Native Isaac Lab vectorization:
        # author env_0 once, clone all other environments.
        self.scene.clone_environments(
            copy_from_source=True
        )

        self.module_roots_by_env = tuple(
            {
                owner: (
                    f"/World/envs/env_{env_index}"
                    f"/SMORES_{module_index}"
                )
                for module_index, owner
                in enumerate(
                    self.cfg.possible_agents
                )
            }
            for env_index
            in range(
                self.cfg.scene.num_envs
            )
        )

        self.docking_managers = tuple(
            IsaacDockingManager(
                stage,
                module_roots,
                joints_root=(
                    f"/World/envs/env_{env_index}"
                    "/DockingJoints"
                ),
            )
            for env_index, module_roots
            in enumerate(
                self.module_roots_by_env
            )
        )

        if (
            len(self.docking_managers)
            != self.cfg.scene.num_envs
        ):
            raise RuntimeError(
                "one docking manager per environment is required"
            )

        self._setup_native_courses()

        # Constructed lazily after PhysX has initialized the articulations.
        self.state_readers_by_env = None
        self.state_graph_builders = None
        self.expert_planners = None
        self.topology_matchers = None
        self.physical_expert_backends = None
        self.expert_assembly_plans = None
        self.expert_assembly_executors = None

    def _pre_physics_step(self,actions):
        return NativeC1TaskHooks._pre_physics_step(self,actions)

    def _apply_action(self):
        return NativeC1TaskHooks._apply_action(self)

    def _begin_expert_assembly_runtime(self, morphology='rc_car8'):
        from drl.MARL.c1_initialization import require_expert_validation
        require_expert_validation(self)
        from drl.MARL.isaaclab_expert_runtime import InProcessAssemblyRuntime
        self._ensure_physical_expert_backends()
        self._start_expert_assembly(morphology, stamp_s=0.)
        self.assembly_runtimes = tuple(InProcessAssemblyRuntime(fsm,backend)
            for fsm,backend in zip(self.expert_assembly_executors,self.physical_expert_backends,strict=True))
        self.assembly_verified = [False]*self.num_envs
        self.assembly_morphology = morphology
        self._assembly_tick = 0
        return self.assembly_runtimes

    def _expert_assembly_control_tick(self):
        stamp = self._sim_step_counter * self.physics_dt
        if self._assembly_tick % 4 == 0:
            for runtime in self.assembly_runtimes:
                runtime.tick(stamp)
            if any(r.decision is not None and r.decision.done for r in self.assembly_runtimes):
                matches = self._verify_target_topology(self.assembly_morphology,stamp_s=stamp)
                for i,(runtime,match) in enumerate(zip(self.assembly_runtimes,matches,strict=True)):
                    if runtime.decision is not None and runtime.decision.done:
                        if match is None or len(self.docking_managers[i].connections)!=7:
                            raise RuntimeError(f'Expert FSM success without physical target topology in env{i}')
                        self.assembly_verified[i] = True
        for runtime in self.assembly_runtimes:
            runtime.apply()
        self._assembly_tick += 1

    def _expert_assembly_runtime_complete(self):
        return tuple(getattr(self,'assembly_verified',[False]*self.num_envs))

    def _ensure_state_readers(self):
        """Create direct PhysX readers once the simulation is initialized."""

        readers = getattr(
            self,
            "state_readers_by_env",
            None,
        )

        if readers is not None:
            return readers

        from drl.MARL.isaaclab_articulation import NativeArticulationAccess
        from smores_ep.config.physics import SmoresActuatorConfig
        from smores_ep.isaac.dynamic_stage import (
            ArticulationStateReader,
        )

        readers = tuple(
            {
                owner: ArticulationStateReader(
                    module_root,
                    SmoresActuatorConfig.payload_overdrive(),
                    stage=self.sim.stage,
                    articulation=NativeArticulationAccess(
                        self.sim.physics_manager.get_physics_sim_view(), module_root,
                        state_clock=(lambda i=env_index: (
                            self.sim.get_physics_step_count(), self.generation[i]
                        )) if self.cfg.c1_readonly_state_cache else None),
                )
                for owner, module_root
                in module_roots.items()
            }
            for env_index, module_roots
            in enumerate(self.module_roots_by_env)
        )

        if len(readers) != self.num_envs:
            raise RuntimeError(
                "state-reader batch does not match Isaac environments"
            )

        for env_index, row in enumerate(
            readers
        ):
            if set(row) != set(
                self.cfg.possible_agents
            ):
                raise RuntimeError(
                    "state-reader owner set mismatch "
                    f"for env {env_index}"
                )

        self.state_readers_by_env = (
            readers
        )

        from drl.MARL.isaaclab_articulation import NativeTransformSource
        self.native_transforms = NativeTransformSource(self.sim.stage,
            {reader.module_root: reader._articulation for row in readers for reader in row.values()},
            lambda: self._sim_step_counter)
        self.native_transforms.rigid_views = {path:self.sim.physics_manager.get_physics_sim_view().create_rigid_body_view(path)
            for path in getattr(self,'plunger_paths',{}).values()}
        for docking in self.docking_managers:
            docking._world_transform_provider = self.native_transforms

        return readers

    def _read_joint_states(self):
        """Read the current physical SMORES joint state for every env."""

        readers = (
            self._ensure_state_readers()
        )

        return tuple(
            {
                owner: reader.read()
                for owner, reader
                in row.items()
            }
            for row in readers
        )

    def _ensure_state_graph_builders(self):
        """Create one canonical state/graph builder per Isaac environment."""

        builders = getattr(
            self,
            "state_graph_builders",
            None,
        )

        if builders is not None:
            return builders

        from smores_ep.isaac.state_graph_publisher import (
            SmoresStateGraphPublisher,
        )

        readers = self._ensure_state_readers()

        def native_velocity(owner, env_index):
            values = readers[env_index][owner]._articulation.view.get_root_velocities().numpy()[0]
            return tuple(map(float, values[:3])), tuple(map(float, values[3:]))

        builders = tuple(
            SmoresStateGraphPublisher(
                stage=self.sim.stage,
                module_roots=self.module_roots_by_env[env_index],
                states=readers[env_index],
                docking=self.docking_managers[env_index],
                actuator_profiles={owner: reader._drive for owner, reader in readers[env_index].items()},
                include_contact_candidates=True,
                velocity_provider=lambda owner, index=env_index: native_velocity(owner, index),
            )
            for env_index in range(self.num_envs)
        )

        if len(builders) != self.num_envs:
            raise RuntimeError(
                "state/graph builder batch does not match Isaac environments"
            )

        self.state_graph_builders = builders

        return builders

    def _state_graph_snapshots(
        self,
        *,
        stamp_s: float,
    ):
        """Return one canonical in-memory C1 graph per vector environment."""

        if (
            isinstance(stamp_s, bool)
            or not isinstance(stamp_s, (int, float))
        ):
            raise TypeError(
                "stamp_s must be numeric"
            )

        return tuple(
            builder.snapshot(
                float(stamp_s)
            )
            for builder
            in self._ensure_state_graph_builders()
        )

    def _ensure_physical_expert_backends(self):
        """Legacy alias; these objects execute primitives, not assembly plans."""
        return self._ensure_native_actuation_backends()

    def _ensure_native_actuation_backends(self, *, reset_env_ids=()):
        """Create low-level policy/validation actuation, without an Expert FSM."""

        backends = getattr(
            self,
            "physical_expert_backends",
            None,
        )

        if backends is not None and not reset_env_ids:
            return backends

        from smores_ep.config.geometry import (
            SmoresGeometry,
        )
        from smores_ep.config.physics import (
            SMORES_EP_MAX_WHEEL_SPEED_RAD_S,
        )
        from smores_ep.isaac.command_router import (
            IsaacMultiModuleCommandRouter,
        )
        from smores_ep.isaac.dynamic_stage import (
            DynamicDriveController,
        )
        from smores_ep.isaac.primitive_executor import (
            IsaacPrimitiveExecutor,
        )
        from smores_ep.primitives.pose_control import (
            PoseControllerConfig,
        )

        geometry = SmoresGeometry()

        readers = (
            self._ensure_state_readers()
        )

        result = list(backends) if backends is not None else [None]*self.num_envs

        selected = tuple(reset_env_ids) if backends is not None else tuple(range(self.num_envs))
        for env_index in selected:
            states = readers[
                env_index
            ]

            drives = {
                owner: DynamicDriveController(
                    reader,
                    geometry,
                    SMORES_EP_MAX_WHEEL_SPEED_RAD_S,
                )
                for owner, reader
                in states.items()
            }

            router = (
                IsaacMultiModuleCommandRouter(
                    states,
                    drives,
                    self.docking_managers[
                        env_index
                    ],
                )
            )

            primitive_executor = (
                IsaacPrimitiveExecutor(
                    self.sim.stage,
                    self.module_roots_by_env[
                        env_index
                    ],
                    states,
                    self.docking_managers[
                        env_index
                    ],
                    motion_module_ids=tuple(
                        self.cfg.possible_agents
                    ),
                    geometry=geometry,
                    staging_collision_avoidance=True,
                    staging_center_clearance_m=0.110,
                    staging_waypoint_margin_m=0.015,
                    pose_controller=(
                        PoseControllerConfig(
                            max_linear_speed_m_s=(
                                geometry.wheel_radius_m
                                * SMORES_EP_MAX_WHEEL_SPEED_RAD_S
                            ),
                        )
                    ),
                )
            )

            primitive_executor.reset_free_modules_callback = (
                router.reset_free_modules
            )

            # File/ROS action sources do not exist in the native
            # training path, so there is nothing external to invalidate.
            primitive_executor.invalidate_module_command_sources_callback = (
                lambda _module_ids: None
            )

            result[env_index] = {
                    "states": states,
                    "drives": drives,
                    "router": router,
                    "primitive_executor": (
                        primitive_executor
                    ),
                }

        self.physical_expert_backends = tuple(
            result
        )

        return self.physical_expert_backends

    def _start_expert_assembly(
        self,
        morphology: str,
        *,
        stamp_s: float,
        env_ids=None,
    ):
        """Create the existing ParallelAssemblyExecutor independently per env."""

        from drl.MARL.c1_initialization import require_expert_validation
        require_expert_validation(self)

        from mssr_expert.execution.parallel_assembly_executor import (
            ParallelAssemblyExecutor,
            physical_fold_push_pairs,
            physical_posture_groups,
        )

        selected = tuple(range(self.num_envs)) if env_ids is None else tuple(env_ids)
        morphologies = [morphology]*self.num_envs if isinstance(morphology,str) else list(morphology)
        current = self._current_attributed_graphs(stamp_s=stamp_s)
        planners = self._ensure_expert_planners()
        planning_results = tuple(planners[i].plan(current_graph=current[i],
            target_graph=self._load_target_graph(morphologies[i])) for i in selected)

        executors = []

        for env_index, result in zip(selected, planning_results, strict=True):
            target = (
                result.target_graph
            )

            attributes = (
                target.global_attributes
            )

            target_to_module = dict(
                result.assignment
                .target_to_module
            )

            post_tilts = {}

            raw_tilts = attributes.get(
                "post_assembly_tilt_rad_by_vertex",
                {},
            )

            if isinstance(
                raw_tilts,
                dict,
            ):
                for vertex, angle in (
                    raw_tilts.items()
                ):
                    module_id = (
                        target_to_module[
                            str(vertex)
                        ]
                    )

                    post_tilts[
                        module_id
                    ] = float(angle)

            post_pans = {}

            raw_pans = attributes.get(
                "post_assembly_pan_rad_by_vertex",
                {},
            )

            if isinstance(
                raw_pans,
                dict,
            ):
                for vertex, angle in (
                    raw_pans.items()
                ):
                    module_id = (
                        target_to_module[
                            str(vertex)
                        ]
                    )

                    post_pans[
                        module_id
                    ] = float(angle)

            executor = (
                ParallelAssemblyExecutor(
                    plan=(
                        result.assembly_plan
                    ),
                    execution_id=(
                        f"c1-{morphologies[env_index]}"
                        f"-env-{env_index}-generation-{self.generation[env_index]+1}"
                    ),
                    enable_borrowed_helper=bool(
                        attributes.get(
                            "requires_helping_module",
                            False,
                        )
                    ),
                    layout_pose_by_module={},
                    post_assembly_tilt_by_module=(
                        post_tilts
                    ),
                    post_assembly_pan_by_module=(
                        post_pans
                    ),
                    posture_tilt_tolerance_rad=(
                        None
                        if attributes.get(
                            "post_assembly_tilt_tolerance_rad"
                        )
                        is None
                        else float(
                            attributes[
                                "post_assembly_tilt_tolerance_rad"
                            ]
                        )
                    ),
                    posture_tilt_max_servo_error_rad=(
                        None
                        if attributes.get(
                            "post_assembly_tilt_max_servo_error_rad"
                        )
                        is None
                        else float(
                            attributes[
                                "post_assembly_tilt_max_servo_error_rad"
                            ]
                        )
                    ),
                    coordinate_posture_tilts=bool(
                        attributes.get(
                            "coordinate_post_assembly_tilts",
                            False,
                        )
                    ),
                    posture_tilt_groups_by_module=(
                        physical_posture_groups(
                            attributes.get(
                                "post_assembly_tilt_groups_by_vertex"
                            ),
                            target_to_module,
                            set(post_tilts),
                        )
                        if post_tilts
                        else ()
                    ),
                    posture_push_by_lifter_module=(
                        physical_fold_push_pairs(
                            attributes.get(
                                "post_assembly_push_pairs_by_vertex"
                            ),
                            target_to_module,
                            set(post_tilts),
                        )
                        if post_tilts
                        else {}
                    ),
                    additional_known_module_ids=(
                        result.reserve_module_ids
                    ),
                )
            )

            executors.append(
                executor
            )

        self.expert_assembly_plans = (
            planning_results
        )

        self.expert_assembly_executors = tuple(
            executors
        )

        return (
            self.expert_assembly_executors
        )

    def _current_attributed_graphs(
        self,
        *,
        stamp_s: float,
    ):
        """Convert the canonical in-memory snapshots into Expert graphs."""

        from mssr_expert.graph.graph_builder import (
            GraphBuilder,
        )

        snapshots = (
            self._state_graph_snapshots(
                stamp_s=stamp_s
            )
        )

        builder = GraphBuilder()

        return tuple(
            builder.build(
                observation=snapshot["state"],
                graph_payload=snapshot["graph"],
            )
            for snapshot
            in snapshots
        )

    def _load_target_graph(
        self,
        morphology: str,
    ):
        """Load one of the three active C1 morphology targets."""

        from pathlib import Path

        from mssr_expert.graph.serialization import (
            load_attributed_graph,
        )

        active = {
            "rc_car8": "smores_rc_car8.json",
            "snake8": "smores_snake8.json",
            "mobile_manipulator8": (
                "smores_mobile_manipulator8.json"
            ),
        }

        try:
            filename = active[
                str(morphology)
            ]
        except KeyError as error:
            raise ValueError(
                "unsupported C1 morphology: "
                f"{morphology!r}"
            ) from error

        repository_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        target_path = (
            repository_root
            / "mssr_ws"
            / "src"
            / "mssr_expert"
            / "config"
            / filename
        )

        if not target_path.is_file():
            raise FileNotFoundError(
                f"C1 morphology target missing: {target_path}"
            )

        return load_attributed_graph(
            target_path
        )

    def _ensure_expert_planners(self):
        """Create one deterministic self-assembly planner per vector env."""

        from drl.MARL.c1_initialization import require_expert_validation
        require_expert_validation(self)

        planners = getattr(
            self,
            "expert_planners",
            None,
        )

        if planners is not None:
            return planners

        from mssr_expert.planning.smores_ep.parallel_self_assembly_planner import (
            ParallelSelfAssemblyPlanner,
        )

        planners = tuple(
            ParallelSelfAssemblyPlanner(
                require_disconnected_modules=True,
                layout_clearance_m=0.070,
                assignment_staging_distance_m=0.070,
                assignment_corridor_clearance_m=0.110,
            )
            for _ in range(
                self.num_envs
            )
        )

        self.expert_planners = planners

        return planners

    def _ensure_topology_matchers(self):
        """Create the existing topology verifier for every vector env."""

        matchers = getattr(
            self,
            "topology_matchers",
            None,
        )

        if matchers is not None:
            return matchers

        from mssr_expert.planning.smores_ep.self_reconfiguration_planner import (
            SmoresSelfReconfigurationPlanner,
        )

        matchers = tuple(
            SmoresSelfReconfigurationPlanner()
            for _ in range(
                self.num_envs
            )
        )

        self.topology_matchers = matchers

        return matchers

    def _plan_expert_targets(
        self,
        morphology: str,
        *,
        stamp_s: float,
    ):
        """Run the existing deterministic Expert planner independently per env."""

        morphologies = [morphology]*self.num_envs if isinstance(morphology,str) else list(morphology)
        targets = [self._load_target_graph(name) for name in morphologies]

        current_graphs = (
            self._current_attributed_graphs(
                stamp_s=stamp_s
            )
        )

        planners = (
            self._ensure_expert_planners()
        )

        return tuple(
            planner.plan(
                current_graph=current_graph,
                target_graph=target,
            )
            for planner, current_graph, target
            in zip(
                planners,
                current_graphs, targets,
            )
        )

    def _verify_target_topology(
        self,
        morphology: str,
        *,
        stamp_s: float,
    ):
        """Return the physical->target assignment when a morphology matches."""

        morphologies = [morphology]*self.num_envs if isinstance(morphology,str) else list(morphology)
        targets = [self._load_target_graph(name) for name in morphologies]

        current_graphs = (
            self._current_attributed_graphs(
                stamp_s=stamp_s
            )
        )

        matchers = (
            self._ensure_topology_matchers()
        )

        return tuple(
            matcher.configuration_assignment(
                current_graph,
                target,
            )
            for matcher, current_graph, target
            in zip(
                matchers,
                current_graphs, targets,
            )
        )

    def _get_observations(self):
        return NativeC1TaskHooks._get_observations(self)

    def _get_states(self):
        return NativeC1TaskHooks._get_states(self)

    def _get_rewards(self):
        return NativeC1TaskHooks._get_rewards(self)

    def _get_dones(self):
        return NativeC1TaskHooks._get_dones(self)
