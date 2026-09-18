"""RC-Car8 held support-height control using the existing morphology model.

Height is the calibrated support/body clearance, not a world-Z pose goal.
The contact geometry is used only on its monotonic branch containing the
library's ready posture. PAN traction and TILT share motors on each module.
"""
from dataclasses import dataclass, field, replace
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import sys

from mssr_expert.behaviors.morphology_dof_model import SmoresMorphologyDofAnalyzer
from mssr_expert.behaviors.morphology_library import AssignedModule
from mssr_expert.behaviors.morphology_locomotion import validate_locomotion_dofs
from mssr_expert.planning.smores_ep.self_reconfiguration_planner import SmoresSelfReconfigurationPlanner
from mssr_expert.teleop.safety import SafetyDecision


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@lru_cache(maxsize=4)
def load_geometry(config_path=None):
    """Reuse the pure CAD model even in an external ROS-only environment.

    The installed expert package and the existing Isaac source package are
    separate. A symlink install or explicit source config identifies this
    checkout; no Isaac imports, physical constants or sys.path hacks are needed.
    """
    try:
        from smores_ep.config.geometry import SmoresGeometry
        return SmoresGeometry()
    except ModuleNotFoundError as error:
        if error.name not in {"smores_ep", "smores_ep.config", "smores_ep.config.geometry"}:
            raise
    anchors = [Path(__file__).resolve()]
    if config_path is not None:
        anchors.append(Path(config_path).resolve())
    for anchor in anchors:
        for parent in anchor.parents:
            path = parent / "scripts/smores_ep/src/smores_ep/config/geometry.py"
            if path.is_file():
                name = "_mssr_rc_existing_cad_geometry"
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                return module.SmoresGeometry()
    raise ValueError("Existing smores_ep CAD geometry unavailable; provide a source teleop_config_path")


@dataclass(frozen=True)
class RcCarObservation:
    assignments: tuple
    inventory: object

    @classmethod
    def from_graph(cls, graph, target_graph):
        """Recover physical IDs from face topology; target-slot graphs cannot arm."""
        try:
            if len(graph.nodes) != 8 or any(node.node_type != "physical_module" for node in graph.nodes):
                return None
            assignment = SmoresSelfReconfigurationPlanner().configuration_assignment(graph, target_graph)
            if assignment is None:
                return None
            roles = target_graph.node_by_id()
            assignments = tuple(sorted((AssignedModule(module_id, vertex, str(roles[vertex].attributes["target_role"]))
                                        for vertex, module_id in assignment.target_to_module.items()),
                                       key=lambda item: item.target_vertex_id))
            return cls(assignments, SmoresMorphologyDofAnalyzer().analyze(graph))
        except (KeyError, RuntimeError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class RcCarActions:
    module_actions: dict = field(default_factory=dict)
    joint_targets: tuple = ()
    intent: dict = field(default_factory=dict)
    module_roles: dict = field(default_factory=dict)
    allow_joint_updates: bool = False


class RcCarTeleopController:
    def __init__(self, library, *, geometry, height_rate_m_s=0.005, max_dt_s=0.1):
        if any(not finite(value) or value <= 0 for value in (height_rate_m_s, max_dt_s)):
            raise ValueError("RC height rate and maximum dt must be positive finite values")
        self.library = library
        self.geometry = geometry
        self.height_rate_m_s = height_rate_m_s
        self.max_dt_s = max_dt_s
        # Read the same validated profile consumed by drive_commands(), rather
        # than defining a second traction/steering conversion or set of limits.
        drive = library._profile("rc_car8")["drive"]
        self._linear_limit = float(drive["max_linear_m_s"])
        self._yaw_limit = float(drive["max_yaw_rate_rad_s"])
        pan_x, _, pan_z = geometry.pan_center_body_m
        self._branch_min = -math.atan2(pan_x + geometry.pan_visual_thickness_m / 2,
                                      geometry.pan_visual_radius_m - pan_z)
        self._assignments = ()
        self._targets = {}
        self.desired_chassis_height = None
        self._home = False
        self.recapture = False

    def home(self):
        self._home = True

    def _height(self, angle):
        return self.geometry.ground_contact_height_m(angle)

    def _angle(self, height, lower, upper):
        for _ in range(50):
            middle = (lower + upper) / 2
            if self._height(middle) > height:
                lower = middle
            else:
                upper = middle
        return (lower + upper) / 2

    def _safe(self, *, macro=False):
        return RcCarActions(
            module_actions={} if macro else self.library.drive_commands(
                "rc_car8", self._assignments, linear_m_s=0, yaw_rate_rad_s=0) if self._assignments else {},
            intent={} if self.desired_chassis_height is None else {"chassis_height_m": self.desired_chassis_height,
                                                                 "steering_yaw_rate_rad_s": 0.0},
            module_roles={item.module_id: item.target_role for item in self._assignments})

    def step(self, input, observation, dt, *, safety=SafetyDecision("NONE", False, True, False)):
        if safety.authority == "STRUCTURAL_MACRO":
            self.recapture = True
            self._home = False
            return self._safe(macro=True)
        if safety.authority == "ESTOP":
            self.recapture = True
            self._home = False
        if observation is None:
            self.recapture = True
            self._home = False
            return self._safe(macro=safety.authority == "STRUCTURAL_MACRO")
        templates = self.library.ready_joint_targets("rc_car8", observation.assignments)
        dofs = {(dof.module_id, dof.name): dof for dof in observation.inventory.dofs}
        bounds, positions = {}, {}
        for target in templates:
            tilt = dofs[(target.module_id, "tilt")]
            pan = dofs[(target.module_id, "pan")]
            if not finite(tilt.position_rad) or not pan.can_locomote or not tilt.can_shape:
                return self._safe()
            lower = max(self._branch_min, self.geometry.tilt_min_rad,
                        tilt.lower_limit_rad if finite(tilt.lower_limit_rad) else self.geometry.tilt_min_rad)
            upper = min(0.0, self.geometry.tilt_max_rad,
                        tilt.upper_limit_rad if finite(tilt.upper_limit_rad) else self.geometry.tilt_max_rad)
            if lower > upper or not lower - 1e-9 <= tilt.position_rad <= upper + 1e-9:
                return self._safe()
            bounds[target.module_id] = (lower, upper)
            positions[target.module_id] = tilt.position_rad
        min_height = max(self._height(upper) for lower, upper in bounds.values())
        max_height = min(self._height(lower) for lower, upper in bounds.values())
        if min_height > max_height:
            return self._safe()
        changed = observation.assignments != self._assignments
        self._assignments = observation.assignments
        if changed or self.recapture or safety.authority == "ESTOP":
            self._targets = dict(positions)
            self.desired_chassis_height = sum(map(self._height, positions.values())) / len(positions)
            self.recapture = False
        values = {name: getattr(input, name, None) for name in ("r2", "l2", "right_x", "right_y")}
        valid = finite(dt) and dt >= 0 and all(finite(value) and (0 <= value <= 1 if name in {"r2", "l2"}
                                                                else -1 <= value <= 1)
                                              for name, value in values.items())
        if not safety.motion_enabled or not valid:
            return self._safe()
        dt = min(dt, self.max_dt_s)
        increment = self.height_rate_m_s * dt
        if safety.allow_joint_updates:
            if values["right_y"] != 0:
                self._home = False
                wanted = self.desired_chassis_height + values["right_y"] * increment
            elif self._home:
                nominal = sum(self._height(item.angle_rad) for item in templates) / len(templates)
                wanted = self.desired_chassis_height + max(-increment, min(increment, nominal - self.desired_chassis_height))
                if abs(wanted - nominal) < 1e-12:
                    self._home = False
            else:
                wanted = self.desired_chassis_height
            self.desired_chassis_height = max(min_height, min(max_height, wanted))
        yaw = -values["right_x"] * self._yaw_limit
        propulsion = values["r2"] - values["l2"]
        actions = self.library.drive_commands("rc_car8", self._assignments,
                    linear_m_s=propulsion * self._linear_limit,
                    yaw_rate_rad_s=yaw if propulsion != 0 else 0.0)
        validate_locomotion_dofs(actions, observation.inventory)
        targets = []
        if safety.allow_joint_updates:
            for template in templates:
                module = template.module_id
                if abs(actions[module]["pan_rate_rad_s"]) > 1e-12:
                    self._targets[module] = positions[module]
                    continue
                current_height = self._height(self._targets[module])
                wanted = current_height + max(-increment, min(increment, self.desired_chassis_height - current_height))
                lower, upper = bounds[module]
                if abs(wanted - current_height) > 1e-12:
                    self._targets[module] = self._angle(max(min_height, min(max_height, wanted)), lower, upper)
                # Preserve the exact observed coordinate at neutral; never
                # invert a contact plateau or force the ready posture on entry.
                if abs(self._targets[module] - positions[module]) > 1e-6:
                    # Keep the profile's loaded-joint tolerance: an overly
                    # narrow override can trap the first coordinated snapshot.
                    targets.append(replace(template, angle_rad=self._targets[module]))
        return RcCarActions(actions, tuple(targets),
                            {"chassis_height_m": self.desired_chassis_height, "steering_yaw_rate_rad_s": yaw},
                            {item.module_id: item.target_role for item in self._assignments},
                            safety.allow_joint_updates)
