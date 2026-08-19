"""Execution state machine for topology-preserving self-reconfiguration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from mssr_expert.execution.assembly_policy import (
    AssemblyExecutionPolicy,
)
from mssr_expert.execution.parallel_assembly_executor import (
    ParallelAssemblyExecutor,
)
from mssr_expert.execution.primitive_protocol import (
    PrimitiveGoalRequest,
    make_undock_goal,
    parse_primitive_statuses,
)
from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph
from mssr_expert.planning.smores_ep.self_reconfiguration_planner import (
    SelfReconfigurationPlan,
    SmoresSelfReconfigurationPlanner,
)


class SelfReconfigurationExecutionError(ValueError):
    """Raised when a reconfiguration executor is configured incorrectly."""


@dataclass(frozen=True)
class SelfReconfigurationDecision:
    """One observable decision from the reconfiguration state machine."""

    state: str
    phase: str
    primitive_goal: PrimitiveGoalRequest | None
    active_goal_ids: tuple[str, ...]
    completed_operation_count: int
    total_operation_count: int
    retained_connection_count: int
    done: bool
    success: bool
    message: str = ""

    @property
    def primitive_goal_payload(self) -> dict[str, Any] | None:
        if self.primitive_goal is None:
            return None
        return self.primitive_goal.to_dict()


class SelfReconfigurationExecutor:
    """Stow, release, redock, and verify one reconfiguration plan."""

    def __init__(
        self,
        plan: SelfReconfigurationPlan,
        execution_id: str = "self-reconfiguration",
        joint_timeout_s: float = 30.0,
        undock_timeout_s: float = 10.0,
        align_timeout_s: float = 60.0,
        dock_timeout_s: float = 10.0,
        retry_count: int = 2,
        dock_recovery_count: int = 2,
        assembly_policy: AssemblyExecutionPolicy | None = None,
    ) -> None:
        if not execution_id.strip():
            raise SelfReconfigurationExecutionError(
                "execution_id cannot be empty."
            )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (
                joint_timeout_s,
                undock_timeout_s,
                align_timeout_s,
                dock_timeout_s,
            )
        ):
            raise SelfReconfigurationExecutionError(
                "All primitive timeouts must be positive and finite."
            )
        if not isinstance(retry_count, int) or retry_count < 0:
            raise SelfReconfigurationExecutionError(
                "retry_count must be a non-negative integer."
            )

        self.plan = plan
        self.execution_id = execution_id
        self.joint_timeout_s = joint_timeout_s
        self.undock_timeout_s = undock_timeout_s
        self.retry_count = retry_count
        self._planner = SmoresSelfReconfigurationPlanner()
        self._assembly_policy = assembly_policy or AssemblyExecutionPolicy(
            align_timeout_s=align_timeout_s,
            dock_timeout_s=dock_timeout_s,
            align_retry_count=retry_count,
            dock_recovery_count=dock_recovery_count,
        )
        self._known_module_ids = tuple(
            sorted(
                set(plan.assignment.target_to_module.values())
                | set(plan.reserve_module_ids)
            )
        )

        self._prepare_items = tuple(
            sorted(plan.prepare_tilt_by_module.items())
        )
        self._prepare_tilt_groups = tuple(
            tuple(group) for group in plan.prepare_tilt_groups_by_module
        )
        raw_prepare_max_servo_error = (
            plan.source_graph.global_attributes.get(
                "pre_reconfiguration_tilt_max_servo_error_rad"
            )
            if plan.source_graph is not None
            else None
        )
        self._prepare_tilt_max_servo_error_rad = (
            None
            if raw_prepare_max_servo_error is None
            else float(raw_prepare_max_servo_error)
        )
        if self._prepare_tilt_max_servo_error_rad is not None and (
            not math.isfinite(self._prepare_tilt_max_servo_error_rad)
            or self._prepare_tilt_max_servo_error_rad <= 0.0
        ):
            raise SelfReconfigurationExecutionError(
                "pre_reconfiguration_tilt_max_servo_error_rad must be "
                "positive and finite."
            )
        self._prepare_goal_by_module: dict[str, str] = {}
        self._prepare_succeeded: set[str] = set()
        self._prepare_awaiting_goal_id: str | None = None
        self._stage_index = 0
        self._stage_detach_index = 0
        self._reserve_detach_index = 0
        self._active_goal: PrimitiveGoalRequest | None = None
        self._assembly_active_goal_ids: tuple[str, ...] = ()
        self._retry_by_operation: dict[tuple[str, int], int] = {}
        self._completed_prepare = 0
        self._completed_detach = 0
        self._assembly_completed = 0
        self._final_posture_completed = 0
        self._failure_message = ""
        self._assembly_executor = self._make_assembly_executor()
        self._phase = (
            "PREPARE"
            if self._prepare_items
            else self._first_reconfiguration_phase()
        )
        self._state = "READY"

    def _first_reconfiguration_phase(self) -> str:
        if self.plan.reserve_detach_actions:
            return "RESERVE_UNDOCK"
        return self._phase_after_reserve_detach()

    def _phase_after_reserve_detach(self) -> str:
        return "STAGE_UNDOCK" if self.plan.stages else "ASSEMBLY"

    def _make_assembly_executor(self) -> ParallelAssemblyExecutor:
        if self.plan.stages:
            stage_plan = self.plan.stages[self._stage_index].assembly_plan
            is_last_stage = self._stage_index + 1 == len(self.plan.stages)
            stage_suffix = f"stage-{self._stage_index}-attach"
        else:
            stage_plan = self.plan.assembly_plan
            is_last_stage = True
            stage_suffix = "attach"
        return ParallelAssemblyExecutor(
            plan=stage_plan,
            execution_id=f"{self.execution_id}-{stage_suffix}",
            **self._assembly_policy.executor_kwargs(),
            enable_borrowed_helper=stage_plan.requires_helper,
            helper_joint_timeout_s=self.joint_timeout_s,
            post_assembly_tilt_by_module=(
                self.plan.final_tilt_by_module if is_last_stage else {}
            ),
            post_assembly_pan_by_module=(
                self.plan.final_pan_by_module if is_last_stage else {}
            ),
            posture_tilt_tolerance_rad=(
                float(
                    self.plan.target_graph.global_attributes[
                        "post_assembly_tilt_tolerance_rad"
                    ]
                )
                if is_last_stage
                and self.plan.target_graph.global_attributes.get(
                    "post_assembly_tilt_tolerance_rad"
                ) is not None
                else None
            ),
            posture_tilt_max_servo_error_rad=(
                float(
                    self.plan.target_graph.global_attributes[
                        "post_assembly_tilt_max_servo_error_rad"
                    ]
                )
                if is_last_stage
                and self.plan.target_graph.global_attributes.get(
                    "post_assembly_tilt_max_servo_error_rad"
                ) is not None
                else None
            ),
            coordinate_posture_tilts=(
                self.plan.coordinate_final_tilts if is_last_stage else False
            ),
            posture_tilt_groups_by_module=(
                self.plan.final_tilt_groups_by_module
                if is_last_stage
                else ()
            ),
            posture_push_by_lifter_module=(
                self.plan.final_push_by_lifter_module
                if is_last_stage
                else {}
            ),
            additional_known_module_ids=self._known_module_ids,
        )

    @property
    def total_operation_count(self) -> int:
        return (
            len(self._prepare_items)
            + len(self.plan.detach_actions)
            + self.plan.assembly_plan.action_count
            + len(self.plan.final_tilt_by_module)
            + len(self.plan.final_pan_by_module)
        )

    def step(
        self,
        status_payload: Mapping[str, Any] | None = None,
        current_graph: AttributedRobotGraph | None = None,
    ) -> SelfReconfigurationDecision:
        """Consume current graph/status and emit at most one primitive goal."""

        if self._state == "FAILED":
            return self._decision(None, self._failure_message)
        if self._state == "SUCCEEDED":
            return self._decision(None, "Target morphology verified.")

        if self._phase == "PREPARE":
            return self._step_prepare(status_payload)

        if self._phase == "RESERVE_UNDOCK":
            terminal = self._consume_active_status(status_payload)
            if terminal is not None:
                return terminal
            if self._active_goal is not None:
                self._state = "WAITING_RESERVE_UNDOCK"
                return self._decision(
                    None,
                    f"Waiting for {self._active_goal.goal_id}.",
                )
            if self._reserve_detach_index >= len(
                self.plan.reserve_detach_actions
            ):
                self._phase = self._phase_after_reserve_detach()
                self._state = f"READY_{self._phase}"
                return self._decision(
                    None,
                    "Surplus leaf modules were released into the reserve "
                    "pool; reconfiguration of the assigned component starts.",
                )
            action = self.plan.reserve_detach_actions[
                self._reserve_detach_index
            ]
            retry = self._operation_retry()
            goal_id = (
                f"{self.execution_id}-reserve-undock-"
                f"{self._reserve_detach_index}"
            )
            if retry:
                goal_id += f"-r{retry}"
            goal = make_undock_goal(
                goal_id=goal_id,
                first_module_id=action.module_a_id,
                first_face=action.face_a,
                second_module_id=action.module_b_id,
                second_face=action.face_b,
                timeout_s=self.undock_timeout_s,
            )
            return self._dispatch(
                goal,
                f"Releasing surplus leaf {action.module_a_id}:"
                f"{action.face_a} from {action.module_b_id}:"
                f"{action.face_b}.",
            )

        if self._phase == "STAGE_UNDOCK":
            terminal = self._consume_active_status(status_payload)
            if terminal is not None:
                return terminal
            if self._active_goal is not None:
                self._state = f"WAITING_{self._phase}"
                return self._decision(
                    None,
                    f"Waiting for {self._active_goal.goal_id}.",
                )

        if self._phase == "STAGE_UNDOCK":
            stage = self.plan.stages[self._stage_index]
            if self._stage_detach_index >= len(stage.detach_actions):
                self._phase = "STAGE_ASSEMBLY"
                self._state = "READY_STAGE_ASSEMBLY"
                module_list = ", ".join(stage.mobile_module_ids)
                return self._decision(
                    None,
                    f"Wave [{module_list}] is free; its target docking starts "
                    "before another source connection is released.",
                )
            action = stage.detach_actions[self._stage_detach_index]
            retry = self._operation_retry()
            goal_id = (
                f"{self.execution_id}-stage-{self._stage_index}"
                f"-undock-{self._stage_detach_index}"
            )
            if retry:
                goal_id += f"-r{retry}"
            goal = make_undock_goal(
                goal_id=goal_id,
                first_module_id=action.module_a_id,
                first_face=action.face_a,
                second_module_id=action.module_b_id,
                second_face=action.face_b,
                timeout_s=self.undock_timeout_s,
            )
            return self._dispatch(
                goal,
                f"Releasing {action.module_a_id}:{action.face_a} from "
                f"{action.module_b_id}:{action.face_b}.",
            )

        if self._phase in {"STAGE_ASSEMBLY", "ASSEMBLY"}:
            child = self._assembly_executor.step(status_payload)
            self._assembly_active_goal_ids = child.active_goal_ids
            completed_before_stage = sum(
                stage.assembly_plan.action_count
                for stage in self.plan.stages[: self._stage_index]
            )
            self._assembly_completed = (
                completed_before_stage + child.completed_action_count
            )
            if child.done and not child.success:
                self._state = "FAILED"
                self._failure_message = child.message
                return self._decision(None, self._failure_message)
            if child.done and child.success:
                if (
                    self._phase == "STAGE_ASSEMBLY"
                    and self._stage_index + 1 < len(self.plan.stages)
                ):
                    completed_stage = self.plan.stages[
                        self._stage_index
                    ]
                    self._assembly_completed = (
                        completed_before_stage
                        + completed_stage.assembly_plan.action_count
                    )
                    self._stage_index += 1
                    self._stage_detach_index = 0
                    self._assembly_active_goal_ids = ()
                    self._assembly_executor = self._make_assembly_executor()
                    self._phase = "STAGE_UNDOCK"
                    self._state = "READY_NEXT_PROGRESSIVE_STAGE"
                    next_stage = self.plan.stages[self._stage_index]
                    completed_modules = ", ".join(
                        completed_stage.mobile_module_ids
                    )
                    next_modules = ", ".join(next_stage.mobile_module_ids)
                    return self._decision(
                        None,
                        f"Wave [{completed_modules}] is docked; the next "
                        f"source-leaf wave is [{next_modules}].",
                    )
                self._final_posture_completed = len(
                    self.plan.final_tilt_by_module
                ) + len(self.plan.final_pan_by_module)
                self._phase = "VERIFY"
                self._state = "VERIFYING_TARGET"
                return self._decision(
                    None,
                    "All planned docks completed; waiting for graph "
                    "verification.",
                )
            self._state = child.state
            return self._decision(child.primitive_goal, child.message)

        if self._phase == "VERIFY":
            if current_graph is not None and self._planner.target_reached(
                current_graph,
                self.plan,
            ):
                self._state = "SUCCEEDED"
                self._phase = "COMPLETE"
                return self._decision(
                    None,
                    f"{self.plan.source_morphology} was reconfigured into "
                    f"{self.plan.target_morphology}; the complete target "
                    "topology was verified.",
                )
            self._state = "VERIFYING_TARGET"
            return self._decision(
                None,
                "Waiting for the state graph to contain exactly the target "
                "face-attributed connections.",
            )

        raise SelfReconfigurationExecutionError(
            f"Unknown execution phase {self._phase!r}."
        )

    def _step_prepare(
        self,
        status_payload: Mapping[str, Any] | None,
    ) -> SelfReconfigurationDecision:
        """Admit all neutral-tilt goals before waiting for completion."""

        statuses = parse_primitive_statuses(status_payload)
        for index, (module_id, _) in enumerate(self._prepare_items):
            goal_id = self._prepare_goal_by_module.get(module_id)
            if goal_id is None:
                continue
            status = statuses.get(goal_id)
            if status is None:
                continue
            if (
                self._prepare_awaiting_goal_id == goal_id
                and status.state
                in {
                    "accepted",
                    "running",
                    "succeeded",
                    "failed",
                    "canceled",
                    "rejected",
                }
            ):
                self._prepare_awaiting_goal_id = None
            if status.failed:
                key = ("PREPARE", index)
                retry = self._retry_by_operation.get(key, 0)
                if retry < self.retry_count:
                    self._retry_by_operation[key] = retry + 1
                    self._prepare_goal_by_module.pop(module_id, None)
                    self._state = "RETRYING_PREPARE"
                    continue
                self._state = "FAILED"
                self._failure_message = (
                    f"Primitive {status.goal_id} failed: {status.code} "
                    f"{status.message}"
                ).strip()
                return self._decision(None, self._failure_message)
            if (
                status.succeeded
                and module_id not in self._prepare_succeeded
            ):
                self._prepare_succeeded.add(module_id)
                self._completed_prepare += 1

        expected = {module_id for module_id, _ in self._prepare_items}
        if self._prepare_succeeded == expected:
            self._phase = self._first_reconfiguration_phase()
            self._state = f"READY_{self._phase}"
            return self._decision(
                None,
                "Detached modules reached their coordinated neutral "
                "ground posture.",
            )

        if self._prepare_awaiting_goal_id is not None:
            self._state = "WAITING_PREPARE_ADMISSION"
            return self._decision(
                None,
                "Waiting for coordinated tilt-goal admission.",
            )

        prepare_group_index = 0
        prepare_group = expected
        if self._prepare_tilt_groups:
            for group_index, configured_group in enumerate(
                self._prepare_tilt_groups
            ):
                configured_modules = set(configured_group)
                if not configured_modules.issubset(
                    self._prepare_succeeded
                ):
                    prepare_group_index = group_index
                    prepare_group = configured_modules
                    break

        for index, (module_id, angle) in enumerate(self._prepare_items):
            if module_id not in prepare_group:
                continue
            if (
                module_id in self._prepare_succeeded
                or module_id in self._prepare_goal_by_module
            ):
                continue
            retry = self._retry_by_operation.get(("PREPARE", index), 0)
            goal_id = f"{self.execution_id}-prepare-{index}"
            if retry:
                goal_id += f"-r{retry}"
            parameters: dict[str, Any] = {"angle_rad": angle}
            if self._prepare_tilt_max_servo_error_rad is not None:
                parameters["max_servo_error_rad"] = (
                    self._prepare_tilt_max_servo_error_rad
                )
            if self.plan.prepare_stabilize_module_ids:
                parameters["stabilize_during_group_module_ids"] = list(
                    self.plan.prepare_stabilize_module_ids
                )
            if self._prepare_tilt_groups and len(prepare_group) > 1:
                parameters.update(
                    {
                        "coordination_group": (
                            f"{self.execution_id}-prepare"
                            + (
                                f"-group-{prepare_group_index}"
                                if self._prepare_tilt_groups
                                else ""
                            )
                        ),
                        "coordination_size": len(prepare_group),
                    }
                )
            goal = PrimitiveGoalRequest(
                goal_id=goal_id,
                primitive="set_tilt",
                module_ids=(module_id,),
                parameters=parameters,
                timeout_s=self.joint_timeout_s,
            )
            self._prepare_goal_by_module[module_id] = goal_id
            self._prepare_awaiting_goal_id = goal_id
            self._state = "DISPATCHING_PREPARE"
            return self._decision(
                goal,
                f"Admitting coordinated stow goal for {module_id}.",
            )

        self._state = "WAITING_PREPARE_RESULTS"
        return self._decision(
            None,
            "Waiting for coordinated neutral-posture group "
            f"{prepare_group_index + 1}.",
        )

    def _consume_active_status(
        self,
        status_payload: Mapping[str, Any] | None,
    ) -> SelfReconfigurationDecision | None:
        if self._active_goal is None:
            return None
        status = parse_primitive_statuses(status_payload).get(
            self._active_goal.goal_id
        )
        if status is None or not status.terminal:
            return None
        if status.failed:
            retry = self._operation_retry()
            if retry < self.retry_count:
                self._retry_by_operation[self._operation_key()] = retry + 1
                failed_goal = self._active_goal
                self._active_goal = None
                self._state = f"RETRYING_{self._phase}"
                return self._decision(
                    None,
                    f"Retrying {failed_goal.goal_id} after {status.code}: "
                    f"{status.message}",
                )
            self._state = "FAILED"
            self._failure_message = (
                f"Primitive {status.goal_id} failed: {status.code} "
                f"{status.message}"
            ).strip()
            self._active_goal = None
            return self._decision(None, self._failure_message)

        self._active_goal = None
        if self._phase == "RESERVE_UNDOCK":
            self._reserve_detach_index += 1
        else:
            self._stage_detach_index += 1
        self._completed_detach += 1
        self._state = f"{self._phase}_SUCCEEDED"
        return None

    def _dispatch(
        self,
        goal: PrimitiveGoalRequest,
        message: str,
    ) -> SelfReconfigurationDecision:
        self._active_goal = goal
        self._state = f"DISPATCHING_{self._phase}"
        return self._decision(goal, message)

    def _operation_key(self) -> tuple[str, int]:
        return self._phase, self._completed_detach

    def _operation_retry(self) -> int:
        return self._retry_by_operation.get(self._operation_key(), 0)

    def _decision(
        self,
        goal: PrimitiveGoalRequest | None,
        message: str,
    ) -> SelfReconfigurationDecision:
        active = ()
        if self._phase == "PREPARE":
            active = tuple(
                self._prepare_goal_by_module[module_id]
                for module_id, _ in self._prepare_items
                if module_id not in self._prepare_succeeded
                and module_id in self._prepare_goal_by_module
            )
        elif self._active_goal is not None:
            active = (self._active_goal.goal_id,)
        elif self._phase in {"STAGE_ASSEMBLY", "ASSEMBLY"}:
            active = self._assembly_active_goal_ids
        elif goal is not None:
            active = (goal.goal_id,)
        completed = (
            self._completed_prepare
            + self._completed_detach
            + self._assembly_completed
            + self._final_posture_completed
        )
        return SelfReconfigurationDecision(
            state=self._state,
            phase=self._phase,
            primitive_goal=goal,
            active_goal_ids=active,
            completed_operation_count=completed,
            total_operation_count=self.total_operation_count,
            retained_connection_count=self.plan.retained_connection_count,
            done=self._state in {"SUCCEEDED", "FAILED"},
            success=self._state == "SUCCEEDED",
            message=message,
        )
