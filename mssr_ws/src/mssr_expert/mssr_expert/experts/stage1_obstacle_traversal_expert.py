"""Stage 1 expert for SMORES-like parallel assembly over a low obstacle."""
from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.experts.expert_output import ExpertOutput
from mssr_expert.experts.swarm_expert_base import SwarmExpertBase
from mssr_expert.primitives.common import (
    distance_3d,
    edge_is_attached,
    extract_modules,
    limited_xy_velocity,
    module_position,
)
from mssr_expert.utils.deterministic import sorted_ids


class Stage1ObstacleTraversalExpert(SwarmExpertBase):
    """Assemble a support/climber topology before attempting obstacle traversal.

    The policy follows the SMORES-style workflow, adapted to freeform spherical
    modules: choose a target topology, assign modules to target roles, form
    parallel contact pairs, attach only contact-confirmed pairs, then move the
    assembled morphology.
    """

    def __init__(self, seed: int | None = None, max_speed: float = 0.2) -> None:
        super().__init__(seed=seed, max_speed=max_speed)
        self._support_ids: tuple[str, ...] = ()
        self._locked_pairs: tuple[tuple[str, str], ...] = ()
        self._phase = "POSITION_SUPPORTS"
        self._settle_started_step: int | None = None
        self._attach_started_step: int | None = None
        self._phase_started_step: int | None = None
        self._settle_steps_before_attach = 4
        self._settle_steps_after_attach = 10
        self._reattach_period_steps = 8

    def reset(self, scenario: Mapping[str, Any] | None = None) -> None:
        """Reset stage state."""
        super().reset(scenario)
        self._support_ids = ()
        self._locked_pairs = ()
        self._phase = "POSITION_SUPPORTS"
        self._settle_started_step = None
        self._attach_started_step = None
        self._phase_started_step = None

    def step(self, observation: Mapping[str, Any], graph: Mapping[str, Any]) -> ExpertOutput:
        self._step_count += 1
        modules = extract_modules(observation)
        if not modules:
            return ExpertOutput(fsm_state="WAIT_FOR_OBSERVATION")

        obstacle = observation.get("obstacle", {})
        if not isinstance(obstacle, Mapping):
            obstacle = {}
        obstacle_x = float(obstacle.get("x", 2.5))
        obstacle_y = float(obstacle.get("y", 0.0))
        obstacle_height = float(obstacle.get("height", 0.35))
        radius = self._nominal_radius(modules)
        nominal_z = self._nominal_z(modules)

        if set(self._support_ids).difference(modules) or len(self._support_ids) != min(3, len(modules)):
            self._support_ids = self._select_supports(modules, obstacle_x, obstacle_y, radius)
            self._locked_pairs = ()
            self._settle_started_step = None
            self._attach_started_step = None

        support_targets = self._support_targets(obstacle_x, obstacle_y, nominal_z, radius)
        if self._phase in ("FORM_CONTACT_PAIRS", "ATTACH_CONFIRMED_PAIRS"):
            self._locked_pairs = self._contact_confirmed_pairs(graph, modules, radius)
        contact_pair_count = len(self._locked_pairs)
        support_ready = self._supports_ready(modules, support_targets, x_tolerance=0.35, y_tolerance=0.45)
        pivots_attached = self._pairs_attached(graph, self._locked_pairs)

        roles = self._roles(modules)
        metrics = {
            "obstacle_height": obstacle_height,
            "support_ready": support_ready,
            "contact_pair_count": contact_pair_count,
            "target_pair_count": len(self._support_ids),
            "locked_pairs": [list(pair) for pair in self._locked_pairs],
            "pivots_attached": pivots_attached,
            "support_target_distances": self._target_distances(modules, support_targets),
            "assembly_complete": contact_pair_count == len(self._support_ids),
        }

        if self._phase == "POSITION_SUPPORTS" and not support_ready:
            self._fsm_state = "POSITION_SUPPORTS"
            staging_targets = self._staging_targets(modules, support_targets, radius)
            return ExpertOutput(
                locomotion={
                    **self._roll_commands(modules, support_targets, self.max_speed),
                    **self._roll_commands(modules, staging_targets, self.max_speed * 0.75),
                },
                fsm_state=self._fsm_state,
                active_primitive="position_support_row",
                primitive_params={
                    "support_targets": support_targets,
                    "staging_targets": staging_targets,
                    "target_topology": "three_parallel_support_climber_edges",
                },
                module_roles=roles,
                task_metrics={**metrics, "phase": "position_supports"},
            )

        if self._phase == "POSITION_SUPPORTS":
            self._phase = "FORM_CONTACT_PAIRS"
            self._locked_pairs = ()
            self._settle_started_step = None
            self._attach_started_step = None
            contact_pair_count = 0
            pivots_attached = False
            metrics = {**metrics, "contact_pair_count": 0, "locked_pairs": [], "assembly_complete": False}

        if self._phase == "FORM_CONTACT_PAIRS" and contact_pair_count < len(self._support_ids):
            self._settle_started_step = None
            self._attach_started_step = None
            self._fsm_state = "FORM_CONTACT_PAIRS"
            docking_targets = self._docking_targets(modules, radius)
            return ExpertOutput(
                locomotion={
                    **self._hold_commands(self._support_ids),
                    **self._roll_commands(modules, docking_targets, self.max_speed * 0.65),
                },
                fsm_state=self._fsm_state,
                active_primitive="parallel_contact_pairing",
                primitive_params={
                    "docking_targets": docking_targets,
                    "contact_required": True,
                    "target_surface_distance": 2.0 * radius,
                },
                module_roles=roles,
                task_metrics={
                    **metrics,
                    "phase": "form_contact_pairs",
                    "docking_target_distances": self._target_distances(modules, docking_targets),
                },
            )

        if self._phase == "FORM_CONTACT_PAIRS":
            self._phase = "ATTACH_CONFIRMED_PAIRS"

        if self._phase == "ATTACH_CONFIRMED_PAIRS" and not pivots_attached:
            if self._settle_started_step is None:
                self._settle_started_step = self._step_count
            settle_elapsed = self._step_count - self._settle_started_step
            if settle_elapsed < self._settle_steps_before_attach:
                self._fsm_state = "SETTLE_CONTACT_PAIRS"
                return ExpertOutput(
                    locomotion=self._hold_commands(tuple(modules)),
                    fsm_state=self._fsm_state,
                    active_primitive="settle_contact_pairs",
                    primitive_params={
                        "settle_steps_required": self._settle_steps_before_attach,
                        "settle_steps_elapsed": settle_elapsed,
                    },
                    module_roles=roles,
                    task_metrics={**metrics, "phase": "settle_before_attach"},
                )

            self._fsm_state = "ATTACH_CONFIRMED_PAIRS"
            magnetic = tuple(
                self._attach_command(climber_id, support_id, modules, radius)
                for support_id, climber_id in self._locked_pairs
                if not edge_is_attached(graph, climber_id, support_id, "surface_pivot")
            )
            return ExpertOutput(
                locomotion=self._hold_commands(tuple(modules)),
                magnetic=magnetic,
                fsm_state=self._fsm_state,
                active_primitive="attach_contact_confirmed_pivots",
                primitive_params={
                    "joint_type": "spherical",
                    "attachment_mode": "surface_pivot",
                    "pivot_axis": [0.0, 1.0, 0.0],
                    "attach_only_contact_confirmed_pairs": True,
                },
                module_roles=roles,
                attachment_modes={
                    f"{command['module_a_id']}:{command['module_b_id']}": "surface_pivot"
                    for command in magnetic
                },
                task_metrics={**metrics, "phase": "attach_confirmed_pairs"},
            )

        if self._phase == "ATTACH_CONFIRMED_PAIRS":
            self._phase = "VERIFY_CONNECTED_TOPOLOGY"
            self._attach_started_step = None

        if self._attach_started_step is None:
            self._attach_started_step = self._step_count
        attach_elapsed = self._step_count - self._attach_started_step
        if self._phase == "VERIFY_CONNECTED_TOPOLOGY" and attach_elapsed < self._settle_steps_after_attach:
            self._fsm_state = "VERIFY_CONNECTED_TOPOLOGY"
            return ExpertOutput(
                locomotion=self._hold_commands(tuple(modules)),
                fsm_state=self._fsm_state,
                active_primitive="verify_connected_topology",
                primitive_params={
                    "settle_steps_required": self._settle_steps_after_attach,
                    "settle_steps_elapsed": attach_elapsed,
                },
                module_roles=roles,
                task_metrics={**metrics, "phase": "verify_connected_topology"},
            )

        if self._phase == "VERIFY_CONNECTED_TOPOLOGY":
            self._phase = "ROLL_CLIMBERS_ON_SUPPORT_SURFACE"
            self._phase_started_step = self._step_count

        if self._phase == "ROLL_CLIMBERS_ON_SUPPORT_SURFACE":
            if self._climbers_reached_step_lip(modules, obstacle_x, obstacle_height, radius):
                self._phase = "INVERT_ROLES_ON_STEP"
                self._phase_started_step = self._step_count
            else:
                phase_elapsed = self._elapsed_in_phase()
                magnetic = (
                    self._refresh_surface_contacts(self._locked_pairs, modules, radius)
                    if phase_elapsed > 0 and phase_elapsed % self._reattach_period_steps == 0
                    else ()
                )
                self._fsm_state = "ROLL_CLIMBERS_ON_SUPPORT_SURFACE"
                return ExpertOutput(
                    locomotion=self._surface_roll_commands(modules),
                    magnetic=magnetic,
                    fsm_state=self._fsm_state,
                    active_primitive="roll_climbers_on_support_surface",
                    primitive_params={
                        "contact_update": "periodic_detach_attach",
                        "reattach_period_steps": self._reattach_period_steps,
                        "target": "climber_centers_reach_step_lip",
                    },
                    module_roles=roles,
                    task_metrics={
                        **metrics,
                        "phase": "surface_roll_to_step",
                        "phase_elapsed": phase_elapsed,
                        "contact_refresh_count": len(magnetic),
                    },
                )

        if self._phase == "INVERT_ROLES_ON_STEP":
            old_pairs = self._locked_pairs
            inverted_pairs = tuple((climber_id, support_id) for support_id, climber_id in old_pairs)
            self._support_ids = tuple(support_id for support_id, _ in inverted_pairs)
            self._locked_pairs = inverted_pairs
            self._phase = "PULL_OLD_SUPPORTS_ONTO_STEP"
            self._phase_started_step = self._step_count
            self._fsm_state = "INVERT_ROLES_ON_STEP"
            return ExpertOutput(
                locomotion=self._hold_commands(tuple(modules)),
                magnetic=(
                    *self._detach_commands(old_pairs),
                    *self._attach_commands(inverted_pairs, modules, radius, role="role_inversion_surface_pivot"),
                ),
                fsm_state=self._fsm_state,
                active_primitive="invert_support_climber_roles",
                primitive_params={
                    "old_pairs": [list(pair) for pair in old_pairs],
                    "new_pairs": [list(pair) for pair in inverted_pairs],
                    "reason": "climbers_reached_step_lip",
                },
                module_roles=self._roles(modules),
                task_metrics={
                    **metrics,
                    "phase": "role_inversion",
                    "inverted_pairs": [list(pair) for pair in inverted_pairs],
                },
            )

        self._fsm_state = "PULL_OLD_SUPPORTS_ONTO_STEP"
        traverse_commands = self._pull_old_supports_commands(modules)
        magnetic = (
            self._refresh_surface_contacts(self._locked_pairs, modules, radius)
            if self._elapsed_in_phase() > 0 and self._elapsed_in_phase() % self._reattach_period_steps == 0
            else ()
        )
        success = self._assembled_over_obstacle(modules, obstacle_x, radius)
        self._done = success
        return ExpertOutput(
            locomotion=traverse_commands,
            magnetic=magnetic,
            fsm_state="SUCCESS" if success else self._fsm_state,
            active_primitive="pull_old_supports_onto_step",
            primitive_params={
                "target_topology": "three_surface_pivot_edges",
                "motion": "role_inverted_forward_transfer",
                "contact_update": "periodic_detach_attach",
            },
            module_roles=roles,
            task_metrics={
                **metrics,
                "phase": "pull_old_supports_onto_step",
                "assembled_over": success,
                "contact_refresh_count": len(magnetic),
            },
            success=success,
            done=success,
        )

    def _select_supports(
        self,
        modules: Mapping[str, Mapping[str, Any]],
        obstacle_x: float,
        obstacle_y: float,
        radius: float,
    ) -> tuple[str, ...]:
        support_line_x = obstacle_x - 0.95 * radius
        lateral_spacing = 2.1 * radius
        slots = (
            (support_line_x, obstacle_y - lateral_spacing, 0.0),
            (support_line_x, obstacle_y, 0.0),
            (support_line_x, obstacle_y + lateral_spacing, 0.0),
        )
        available = list(sorted_ids(modules))
        supports: list[str] = []
        for slot in slots[: min(3, len(available))]:
            support_id = min(
                available,
                key=lambda module_id: (
                    abs(module_position(modules[module_id])[0] - slot[0])
                    + abs(module_position(modules[module_id])[1] - slot[1]),
                    module_id,
                ),
            )
            supports.append(support_id)
            available.remove(support_id)
        return tuple(supports)

    def _support_targets(
        self,
        obstacle_x: float,
        obstacle_y: float,
        z: float,
        radius: float,
    ) -> dict[str, list[float]]:
        support_line_x = obstacle_x - 0.95 * radius
        lateral_spacing = 2.1 * radius
        lateral_slots = (-lateral_spacing, 0.0, lateral_spacing)
        return {
            support_id: [support_line_x, obstacle_y + lateral_slots[index], z]
            for index, support_id in enumerate(self._support_ids)
        }

    def _staging_targets(
        self,
        modules: Mapping[str, Mapping[str, Any]],
        support_targets: Mapping[str, list[float]],
        radius: float,
    ) -> dict[str, list[float]]:
        support_items = tuple(support_targets.items())
        climbers = [module_id for module_id in sorted_ids(modules) if module_id not in self._support_ids]
        targets: dict[str, list[float]] = {}
        used: set[str] = set()
        for _, support_target in support_items:
            available = [module_id for module_id in climbers if module_id not in used]
            if not available:
                break
            staging = [support_target[0] - 2.25 * radius, support_target[1], support_target[2]]
            climber_id = min(
                available,
                key=lambda module_id: (distance_3d(module_position(modules[module_id]), tuple(staging)), module_id),
            )
            targets[climber_id] = staging
            used.add(climber_id)
        return targets

    def _docking_targets(
        self,
        modules: Mapping[str, Mapping[str, Any]],
        radius: float,
    ) -> dict[str, list[float]]:
        pairs_by_support = {support_id: climber_id for support_id, climber_id in self._locked_pairs}
        used_climbers = set(pairs_by_support.values())
        free_climbers = [
            module_id
            for module_id in sorted_ids(modules)
            if module_id not in self._support_ids and module_id not in used_climbers
        ]
        targets: dict[str, list[float]] = {}
        for support_id, climber_id in self._locked_pairs:
            if support_id in modules and climber_id in modules:
                targets[climber_id] = self._contact_target(modules, support_id, radius)
        for support_id in self._support_ids:
            if support_id in pairs_by_support or support_id not in modules or not free_climbers:
                continue
            target = self._contact_target(modules, support_id, radius)
            climber_id = min(
                free_climbers,
                key=lambda module_id: (distance_3d(module_position(modules[module_id]), tuple(target)), module_id),
            )
            targets[climber_id] = target
            free_climbers.remove(climber_id)
        return targets

    def _contact_target(
        self,
        modules: Mapping[str, Mapping[str, Any]],
        support_id: str,
        radius: float,
    ) -> list[float]:
        support_position = module_position(modules[support_id])
        return [support_position[0] - 1.96 * radius, support_position[1], support_position[2]]

    def _contact_confirmed_pairs(
        self,
        graph: Mapping[str, Any],
        modules: Mapping[str, Mapping[str, Any]],
        radius: float,
    ) -> tuple[tuple[str, str], ...]:
        support_set = set(self._support_ids)
        climber_set = set(modules) - support_set
        candidates: dict[str, list[str]] = {support_id: [] for support_id in self._support_ids}

        for edge in graph.get("edges", []):
            if not isinstance(edge, Mapping) or not self._is_contact_edge(edge):
                continue
            source, target = self._edge_modules(edge)
            if source is None or target is None:
                continue
            if source in support_set and target in climber_set:
                candidates[source].append(target)
            elif target in support_set and source in climber_set:
                candidates[target].append(source)

        pairs: list[tuple[str, str]] = []
        used_climbers: set[str] = set()
        for support_id in self._support_ids:
            available = sorted(climber_id for climber_id in candidates[support_id] if climber_id not in used_climbers)
            if not available:
                continue
            support_position = module_position(modules[support_id])
            climber_id = min(
                available,
                key=lambda module_id: (distance_3d(module_position(modules[module_id]), support_position), module_id),
            )
            pairs.append((support_id, climber_id))
            used_climbers.add(climber_id)
        return tuple(pairs)

    def _pairs_attached(
        self,
        graph: Mapping[str, Any],
        pairs: tuple[tuple[str, str], ...],
    ) -> bool:
        return bool(pairs) and all(
            edge_is_attached(graph, climber_id, support_id, "surface_pivot")
            for support_id, climber_id in pairs
        )

    def _attach_command(
        self,
        climber_id: str,
        support_id: str,
        modules: Mapping[str, Mapping[str, Any]],
        radius: float,
    ) -> Mapping[str, Any]:
        support_position = module_position(modules[support_id])
        climber_position = module_position(modules[climber_id])
        dx = climber_position[0] - support_position[0]
        dy = climber_position[1] - support_position[1]
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        return {
            "module_a_id": climber_id,
            "module_b_id": support_id,
            "command": "attach",
            "joint_type": "spherical",
            "attachment_mode": "surface_pivot",
            "contact_point_world": [
                support_position[0] + radius * dx / norm,
                support_position[1] + radius * dy / norm,
                support_position[2],
            ],
            "pivot_axis": [0.0, 1.0, 0.0],
            "allows_rotation": True,
            "is_load_bearing": True,
            "is_temporary": True,
            "role": "contact_confirmed_surface_pivot",
        }

    def _detach_commands(
        self,
        pairs: tuple[tuple[str, str], ...],
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            {
                "module_a_id": climber_id,
                "module_b_id": support_id,
                "command": "detach",
                "joint_type": "spherical",
                "attachment_mode": "surface_pivot",
                "role": "refresh_surface_contact",
            }
            for support_id, climber_id in pairs
        )

    def _attach_commands(
        self,
        pairs: tuple[tuple[str, str], ...],
        modules: Mapping[str, Mapping[str, Any]],
        radius: float,
        role: str,
    ) -> tuple[Mapping[str, Any], ...]:
        commands: list[Mapping[str, Any]] = []
        for support_id, climber_id in pairs:
            if support_id not in modules or climber_id not in modules:
                continue
            command = dict(self._attach_command(climber_id, support_id, modules, radius))
            command["role"] = role
            commands.append(command)
        return tuple(commands)

    def _refresh_surface_contacts(
        self,
        pairs: tuple[tuple[str, str], ...],
        modules: Mapping[str, Mapping[str, Any]],
        radius: float,
    ) -> tuple[Mapping[str, Any], ...]:
        return (
            *self._detach_commands(pairs),
            *self._attach_commands(pairs, modules, radius, role="rolling_surface_contact_refresh"),
        )

    def _edge_modules(self, edge: Mapping[str, Any]) -> tuple[str | None, str | None]:
        attrs = edge.get("attributes", {})
        if not isinstance(attrs, Mapping):
            attrs = {}
        source = edge.get("module_a_id") or edge.get("source") or attrs.get("module_a_id")
        target = edge.get("module_b_id") or edge.get("target") or attrs.get("module_b_id")
        return (
            source if isinstance(source, str) else None,
            target if isinstance(target, str) else None,
        )

    def _is_contact_edge(self, edge: Mapping[str, Any]) -> bool:
        attrs = edge.get("attributes", {})
        if not isinstance(attrs, Mapping):
            attrs = {}
        status = edge.get("status") or attrs.get("status")
        mode = edge.get("attachment_mode") or attrs.get("attachment_mode")
        return bool(
            status in ("in_contact", "connected")
            or mode in ("rolling_contact", "surface_pivot")
            or edge.get("is_magnet_enabled")
            or attrs.get("is_magnet_enabled")
        )

    def _supports_ready(
        self,
        modules: Mapping[str, Mapping[str, Any]],
        targets: Mapping[str, list[float]],
        x_tolerance: float,
        y_tolerance: float,
    ) -> bool:
        if not targets:
            return False
        for module_id, target in targets.items():
            if module_id not in modules:
                return False
            position = module_position(modules[module_id])
            if abs(position[0] - target[0]) > x_tolerance:
                return False
            if abs(position[1] - target[1]) > y_tolerance:
                return False
        return True

    def _roll_commands(
        self,
        modules: Mapping[str, Mapping[str, Any]],
        targets: Mapping[str, list[float]],
        max_speed: float,
    ) -> dict[str, Mapping[str, float]]:
        commands: dict[str, Mapping[str, float]] = {}
        for module_id, target in targets.items():
            if module_id not in modules:
                continue
            position = module_position(modules[module_id])
            commands[module_id] = limited_xy_velocity(
                target[0] - position[0],
                target[1] - position[1],
                max_speed,
            )
        return commands

    def _hold_commands(self, module_ids: tuple[str, ...]) -> dict[str, Mapping[str, float]]:
        return {
            module_id: {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
            for module_id in module_ids
        }

    def _target_distances(
        self,
        modules: Mapping[str, Mapping[str, Any]],
        targets: Mapping[str, list[float]],
    ) -> dict[str, float]:
        return {
            module_id: distance_3d(module_position(modules[module_id]), tuple(target))
            for module_id, target in targets.items()
            if module_id in modules
        }

    def _roles(self, modules: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
        roles = {module_id: "mobile" for module_id in sorted_ids(modules)}
        for support_id in self._support_ids:
            roles[support_id] = "anchor"
        for _, climber_id in self._locked_pairs:
            roles[climber_id] = "climber"
        return roles

    def _surface_roll_commands(
        self,
        modules: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, float]]:
        commands: dict[str, Mapping[str, float]] = {}
        for support_id, climber_id in self._locked_pairs:
            if support_id in modules:
                commands[support_id] = {
                    "vx": 0.0,
                    "vy": 0.0,
                    "yaw_rate": 0.0,
                }
            if climber_id in modules:
                commands[climber_id] = {
                    "vx": self.max_speed * 0.7,
                    "vy": 0.0,
                    "yaw_rate": 0.35,
                }
        return commands

    def _pull_old_supports_commands(
        self,
        modules: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, float]]:
        commands: dict[str, Mapping[str, float]] = {}
        for support_id, climber_id in self._locked_pairs:
            if support_id in modules:
                commands[support_id] = {
                    "vx": self.max_speed * 0.5,
                    "vy": 0.0,
                    "yaw_rate": 0.0,
                }
            if climber_id in modules:
                commands[climber_id] = {
                    "vx": self.max_speed * 0.8,
                    "vy": 0.0,
                    "yaw_rate": 0.25,
                }
        return commands

    def _climbers_reached_step_lip(
        self,
        modules: Mapping[str, Mapping[str, Any]],
        obstacle_x: float,
        obstacle_height: float,
        radius: float,
    ) -> bool:
        if not self._locked_pairs:
            return False
        for _, climber_id in self._locked_pairs:
            if climber_id not in modules:
                return False
            position = module_position(modules[climber_id])
            if position[0] < obstacle_x - 0.1 * radius and position[2] < radius + 0.5 * obstacle_height:
                return False
        return True

    def _elapsed_in_phase(self) -> int:
        if self._phase_started_step is None:
            return 0
        return max(0, self._step_count - self._phase_started_step)

    def _assembled_over_obstacle(
        self,
        modules: Mapping[str, Mapping[str, Any]],
        obstacle_x: float,
        radius: float,
    ) -> bool:
        if not self._locked_pairs:
            return False
        for support_id, climber_id in self._locked_pairs:
            if support_id not in modules or climber_id not in modules:
                return False
            support_position = module_position(modules[support_id])
            climber_position = module_position(modules[climber_id])
            if max(support_position[0], climber_position[0]) < obstacle_x + 0.75 * radius:
                return False
        return True

    def _nominal_z(self, modules: Mapping[str, Mapping[str, Any]]) -> float:
        return sum(module_position(module)[2] for module in modules.values()) / float(len(modules))

    def _nominal_radius(self, modules: Mapping[str, Mapping[str, Any]]) -> float:
        radii = [float(module.get("radius", 0.6)) for module in modules.values()]
        return sum(radii) / float(len(radii)) if radii else 0.6
