#!/usr/bin/env python3
"""RC-Car8 Nav2 navigation on an externally-known procedural track."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from smores_ep.isaac.obstacle_course import (
    BUTTON_PLATFORM_CENTER_XY_M,
    BUTTON_PLATFORM_SIZE_XY_M,
    rc_car_planar_obstacle_layout,
    sample_rc_car_planar_spec,
)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--goal-x",
        type=float,
        default=None,
        help="Explicit NavigateToPose target X in map/world frame.",
    )
    p.add_argument(
        "--goal-y",
        type=float,
        default=None,
        help="Explicit NavigateToPose target Y in map/world frame.",
    )
    p.add_argument(
        "--goal-yaw",
        type=float,
        default=None,
        help="Explicit NavigateToPose target yaw in radians.",
    )
    p.add_argument(
        "--route-json",
        type=Path,
        help=(
            "Composite-task JSON containing waypoints_xyyaw. Nav2 follows "
            "all poses with NavigateThroughPoses, including real curves."
        ),
    )
    p.add_argument("--action-timeout-s", type=float, default=900.0)
    p.add_argument(
        "--accept-position-m",
        type=float,
        default=None,
        help="Optional early physical position tolerance for explicit goals.",
    )
    p.add_argument(
        "--accept-yaw-rad",
        type=float,
        default=None,
        help="Optional early physical yaw tolerance for explicit goals.",
    )
    p.add_argument("--result-json", type=Path)
    p.add_argument("--dataset-path", type=Path)
    p.add_argument("--episode-id", default="")
    p.add_argument("--task-id", default="")
    p.add_argument(
        "--status-topic",
        default="/mssr/nav2/route_status",
    )
    return p


def point_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay

    length2 = dx * dx + dy * dy

    if length2 <= 1.0e-12:
        return math.hypot(px - ax, py - ay)

    t = (
        (px - ax) * dx
        + (py - ay) * dy
    ) / length2

    t = min(1.0, max(0.0, t))

    qx = ax + t * dx
    qy = ay + t * dy

    return math.hypot(px - qx, py - qy)


def load_composite_route(path: Path) -> tuple[tuple[float, float, float], ...]:
    """Load and validate a composite Nav2 waypoint sequence."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("waypoints_xyyaw") if isinstance(payload, dict) else None
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("--route-json requires at least two waypoints_xyyaw")
    route: list[tuple[float, float, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"Route waypoint {index} must be [x, y, yaw]")
        pose = tuple(float(value) for value in item)
        if not all(math.isfinite(value) for value in pose):
            raise ValueError(f"Route waypoint {index} contains a non-finite value")
        route.append(pose)
    return tuple(route)


def main():
    args = parser().parse_args()

    import rclpy

    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped, Twist
    from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from std_msgs.msg import String

    explicit_goal_values = (
        args.goal_x,
        args.goal_y,
        args.goal_yaw,
    )

    explicit_goal_mode = any(
        value is not None
        for value in explicit_goal_values
    )

    if explicit_goal_mode and not all(
        value is not None
        for value in explicit_goal_values
    ):
        raise ValueError(
            "--goal-x, --goal-y and --goal-yaw must be supplied together"
        )

    composite_route = (
        load_composite_route(args.route_json)
        if args.route_json is not None
        else ()
    )
    composite_payload = (
        json.loads(args.route_json.read_text(encoding="utf-8"))
        if args.route_json is not None
        else {}
    )
    # A composite route can describe the OccupancyGrid without forcing
    # Nav2 through every sampled waypoint.  When an explicit goal is also
    # supplied, keep the composite map but use NavigateToPose.
    composite_map_mode = bool(composite_route)
    explicit_route_mode = composite_map_mode and not explicit_goal_mode

    if composite_map_mode:
        margin_m = 0.75
        xs = [pose[0] for pose in composite_route]
        ys = [pose[1] for pose in composite_route]
        default_bounds = (
            min(xs) - margin_m,
            max(xs) + margin_m,
            min(ys) - margin_m,
            max(ys) + margin_m,
        )
        raw_bounds = composite_payload.get("platform_bounds_xy_m")
        platform_bounds = (
            tuple(float(value) for value in raw_bounds)
            if isinstance(raw_bounds, list) and len(raw_bounds) == 4
            else default_bounds
        )
        raw_start_pad = composite_payload.get(
            "start_pad_bounds_xy_m"
        )
        start_pad_bounds = (
            tuple(float(value) for value in raw_start_pad)
            if isinstance(raw_start_pad, list)
            and len(raw_start_pad) == 4
            else (
                composite_route[0][0] - 0.55,
                composite_route[0][0] + 0.30,
                composite_route[0][1] - 0.60,
                composite_route[0][1] + 0.60,
            )
        )

        baseline_layout = rc_car_planar_obstacle_layout(args.seed)
        raw_free_rectangles = composite_payload.get(
            "free_rectangles_xy_m",
            [],
        )
        free_rectangles = tuple(
            (
                float(rectangle[0]),
                float(rectangle[1]),
                float(rectangle[2]),
                float(rectangle[3]),
            )
            for rectangle in raw_free_rectangles
            if isinstance(rectangle, (list, tuple))
            and len(rectangle) == 4
        )

        raw_footprint = composite_payload.get("vehicle_footprint")
        vehicle_footprint = (
            {
                str(key): float(value)
                for key, value in raw_footprint.items()
            }
            if isinstance(raw_footprint, dict)
            else dict(baseline_layout["vehicle_footprint"])
        )

        raw_cones = composite_payload.get("cone_centers_xy_m", [])
        cone_centers = (
            tuple((float(point[0]), float(point[1])) for point in raw_cones)
            if isinstance(raw_cones, list)
            else ()
        )
        if explicit_goal_mode:
            gx = float(args.goal_x)
            gy = float(args.goal_y)
            gyaw = float(args.goal_yaw)
        else:
            gx, gy, gyaw = composite_route[-1]

        layout = {
            "track_profile": "composite_waypoints",
            "has_curve": any(
                abs(composite_route[index][2] - composite_route[index - 1][2])
                > 1.0e-3
                for index in range(1, len(composite_route))
            ),
            "platform_bounds_xy_m": platform_bounds,
            "start_pad_bounds_xy_m": start_pad_bounds,
            "free_rectangles_xy_m": free_rectangles,
            "centerline_xy_m": tuple((x, y) for x, y, _ in composite_route),
            "corridor_width_m": float(
                composite_payload.get("corridor_width_m", 1.10)
            ),
            "cone_centers_xy_m": cone_centers,
            "cone_radius_m": float(
                composite_payload.get("cone_radius_m", 0.05)
            ),
            "vehicle_footprint": vehicle_footprint,
            "goal_xyyaw": (gx, gy, gyaw),
            "finish_x_m": gx,
            "finish_y_m": gy,
            "finish_yaw_rad": gyaw,
        }
        route_id = f"rc-car-composite-{args.seed:06d}"
    elif explicit_goal_mode:
        gx = float(args.goal_x)
        gy = float(args.goal_y)
        gyaw = float(args.goal_yaw)

        # Explicit button navigation uses the exact same
        # large floor instantiated by ButtonTestCourse.
        platform_cx, platform_cy = (
            BUTTON_PLATFORM_CENTER_XY_M
        )

        platform_sx, platform_sy = (
            BUTTON_PLATFORM_SIZE_XY_M
        )

        platform_bounds = (
            platform_cx
            - 0.5 * platform_sx,

            platform_cx
            + 0.5 * platform_sx,

            platform_cy
            - 0.5 * platform_sy,

            platform_cy
            + 0.5 * platform_sy,
        )

        layout = {
            "track_profile": "explicit_pose",
            "has_curve": False,
            "platform_bounds_xy_m": platform_bounds,

            # make_map() already knows how to rasterize a start-pad.
            # Making the existing platform the start-pad gives us the
            # required free rectangular OccupancyGrid without creating
            # another map implementation.
            "start_pad_bounds_xy_m": platform_bounds,

            # Retained because make_map() expects these existing fields.
            "centerline_xy_m": (
                (
                    platform_bounds[0],
                    platform_cy,
                ),
                (
                    platform_bounds[1],
                    platform_cy,
                ),
            ),

            "corridor_width_m":
                platform_sy,
            "cone_centers_xy_m": (),
            "cone_radius_m": 0.01,

            "goal_xyyaw": (
                gx,
                gy,
                gyaw,
            ),

            # Compatibility fields for the existing result structure.
            # Physical finish detection is disabled in explicit mode.
            "finish_x_m": gx,
            "finish_y_m": gy,
            "finish_yaw_rad": gyaw,
        }

        route_id = (
            f"rc-car-explicit-{args.seed:06d}"
        )

    else:
        spec = sample_rc_car_planar_spec(
            args.seed
        )

        layout = rc_car_planar_obstacle_layout(
            args.seed,
            platform_center_x_m=1.10,
            platform_size_x_m=spec.platform_size_x_m,
            platform_size_y_m=spec.platform_size_y_m,
        )

        route_id = (
            f"rc-car-track-{args.seed:06d}"
        )

    class NodeImpl(Node):

        def __init__(self):
            super().__init__(
                "mssr_rc_car_track_navigation"
            )

            self.client = ActionClient(
                self,
                NavigateThroughPoses if explicit_route_mode else NavigateToPose,
                "navigate_through_poses" if explicit_route_mode else "navigate_to_pose",
            )

            self.cmd_pub = self.create_publisher(
                Twist,
                "/cmd_vel",
                10,
            )

            self._odom_pose = None
            self._robot_graph = None
            self._latest_cmd_vel = (0.0, 0.0)
            self._dataset_timestep = 0
            self._feedback_count = 0

            self.create_subscription(
                Odometry,
                "/odom",
                self._on_odom,
                20,
            )
            self.create_subscription(
                String,
                "/mssr/robot_graph",
                self._on_robot_graph,
                10,
            )
            self.create_subscription(
                Twist,
                "/cmd_vel",
                self._on_cmd_vel,
                20,
            )

            self.status_pub = self.create_publisher(
                String,
                args.status_topic,
                10,
            )

            qos = QoSProfile(depth=1)
            qos.reliability = (
                ReliabilityPolicy.RELIABLE
            )
            qos.durability = (
                DurabilityPolicy.TRANSIENT_LOCAL
            )

            self.map_pub = self.create_publisher(
                OccupancyGrid,
                "/map",
                qos,
            )

            # Live Isaac -> Nav2 obstacle bridge.
            self._dynamic_obstacle_path = (
                Path(args.result_json).parent
                / "rc_car_dynamic_obstacles.json"
            )
            self._last_dynamic_cones = None
            self._last_dynamic_sim_time = None
            self.create_timer(
                0.20,
                self._refresh_dynamic_obstacles,
            )



        def _refresh_dynamic_obstacles(self):
            path = self._dynamic_obstacle_path

            if not path.exists():
                return

            try:
                payload = json.loads(
                    path.read_text()
                )

                simulation_time_s = float(
                    payload["simulation_time_s"]
                )

                cones = tuple(
                    (
                        float(item["x_m"]),
                        float(item["y_m"]),
                    )
                    for item in payload["cones"]
                )

            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                return

            if not cones:
                return

            # Only process fresh Isaac samples.
            if (
                self._last_dynamic_sim_time is not None
                and simulation_time_s
                == self._last_dynamic_sim_time
            ):
                return

            previous = self._last_dynamic_cones

            changed = (
                previous is not None
                and len(previous) == len(cones)
                and any(
                    math.hypot(
                        new_x - old_x,
                        new_y - old_y,
                    )
                    > 0.003
                    for (
                        (new_x, new_y),
                        (old_x, old_y),
                    ) in zip(cones, previous)
                )
            )

            self._last_dynamic_sim_time = (
                simulation_time_s
            )

            self._last_dynamic_cones = cones

            # Publish every fresh map, exactly like the direct bridge
            # that was already validated in RViz.
            grid = self.make_map(cones)

            self.map_pub.publish(grid)

            if changed:
                self.get_logger().warn(
                    "LIVE OBSTACLE CHANGE -> "
                    "new /map published for Nav2 replanning"
                )



        def _on_odom(self, msg):
            q = msg.pose.pose.orientation

            yaw = 2.0 * math.atan2(
                float(q.z),
                float(q.w),
            )

            self._odom_pose = (
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                yaw,
            )

        def _on_robot_graph(self, msg):
            try:
                payload = json.loads(msg.data)
            except (TypeError, json.JSONDecodeError):
                return
            if isinstance(payload, dict):
                self._robot_graph = payload

        def _on_cmd_vel(self, msg):
            self._latest_cmd_vel = (
                float(msg.linear.x),
                float(msg.angular.z),
            )

        def record_dataset(self, progress, done, success, message):
            if args.dataset_path is None or self._robot_graph is None:
                return
            x_m, y_m, yaw_rad = self._odom_pose or (0.0, 0.0, 0.0)
            linear_m_s, yaw_rate_rad_s = self._latest_cmd_vel
            record = {
                "schema_version": "mssr.expert_transition.v3",
                "episode_id": args.episode_id or route_id,
                "timestep": self._dataset_timestep,
                "stage_id": self._dataset_timestep,
                "stage_name": "composite_nav2",
                "task_type": "flat_navigation",
                "fsm_state": "NAV2_ROUTE",
                "is_first": self._dataset_timestep == 0,
                "is_last": bool(done),
                "is_terminal": bool(done),
                "action_valid": not bool(done),
                "done": bool(done),
                "success": bool(success),
                "reward": 1.0 if done and success else 0.0,
                "discount": 0.0 if done else 1.0,
                "graph_t": self._robot_graph,
                "observation": {
                    "schema_version": "mssr.nav2_observation.v1",
                    "task_id": args.task_id,
                    "route_id": route_id,
                    "pose_xyyaw": [x_m, y_m, yaw_rad],
                    "goal_xyyaw": list(layout["goal_xyyaw"]),
                    "waypoints_xyyaw": [list(pose) for pose in composite_route],
                    "progress": float(progress),
                },
                "expert_action": {
                    "controller": "nav2",
                    "cmd_vel": {
                        "linear_x_m_s": linear_m_s,
                        "angular_z_rad_s": yaw_rate_rad_s,
                    },
                },
                "task_metrics": {
                    "task_id": args.task_id,
                    "progress": float(progress),
                    "message": str(message),
                },
            }
            args.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with args.dataset_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._dataset_timestep += 1

        def finish_metrics(self):
            """Physical RC-Car8 front position relative to finish line."""

            if self._odom_pose is None:
                return {
                    "available": False,
                    "reached": False,
                }

            x_m, y_m, yaw = self._odom_pose

            vehicle_length = float(
                layout["vehicle_footprint"]["length_m"]
            )

            front_offset = 0.5 * vehicle_length

            front_x = (
                x_m
                + front_offset * math.cos(yaw)
            )

            front_y = (
                y_m
                + front_offset * math.sin(yaw)
            )

            finish_x = float(layout["finish_x_m"])
            finish_y = float(layout["finish_y_m"])
            finish_yaw = float(
                layout["finish_yaw_rad"]
            )

            # Tangent of the track at the finish.
            tx = math.cos(finish_yaw)
            ty = math.sin(finish_yaw)

            # Normal along the finish stripe.
            nx = -ty
            ny = tx

            dx = front_x - finish_x
            dy = front_y - finish_y

            longitudinal = (
                dx * tx
                + dy * ty
            )

            lateral = (
                dx * nx
                + dy * ny
            )

            # Finish visual stripe is about 5.5 cm thick.
            # Front touching its near edge already counts.
            finish_tolerance_m = 0.030

            half_finish_width = (
                0.5
                * float(layout["corridor_width_m"])
                + 0.025
            )

            reached = (
                longitudinal >= -finish_tolerance_m
                and abs(lateral) <= half_finish_width
            )

            return {
                "available": True,
                "reached": bool(reached),

                "base_x_m": x_m,
                "base_y_m": y_m,
                "base_yaw_rad": yaw,

                "front_x_m": front_x,
                "front_y_m": front_y,

                "front_progress_past_finish_m":
                    longitudinal,

                "front_lateral_from_finish_m":
                    lateral,

                "front_offset_m":
                    front_offset,

                "finish_tolerance_m":
                    finish_tolerance_m,
            }

        def front_reached_finish(self):
            return bool(
                self.finish_metrics().get(
                    "reached",
                    False,
                )
            )

        def status(
            self,
            done,
            success,
            progress,
            message,
        ):
            msg = String()

            msg.data = json.dumps({
                "schema_version":
                    "mssr.nav2_route_status.v1",
                "route_id": route_id,
                "seed": args.seed,
                "route_kind":
                    layout["track_profile"],
                "done": bool(done),
                "success": bool(success),
                "progress": max(
                    0.0,
                    min(1.0, float(progress)),
                ),
                "message": message,
            })

            self.status_pub.publish(msg)


        def make_map(self, cones=None):
            resolution = 0.025

            x0, x1, y0, y1 = [
                float(v)
                for v in layout["platform_bounds_xy_m"]
            ]

            width = math.ceil(
                (x1 - x0) / resolution
            )

            height = math.ceil(
                (y1 - y0) / resolution
            )

            centerline = [
                (float(x), float(y))
                for x, y in layout["centerline_xy_m"]
            ]

            road_half = (
                0.5
                * float(layout["corridor_width_m"])
            )

            sx0, sx1, sy0, sy1 = [
                float(v)
                for v in layout["start_pad_bounds_xy_m"]
            ]

            free_rectangles = tuple(
                (
                    float(rectangle[0]),
                    float(rectangle[1]),
                    float(rectangle[2]),
                    float(rectangle[3]),
                )
                for rectangle in layout.get(
                    "free_rectangles_xy_m",
                    (),
                )
                if isinstance(rectangle, (list, tuple))
                and len(rectangle) == 4
            )

            if cones is None:
                cones = tuple(
                    (float(x), float(y))
                    for x, y in layout["cone_centers_xy_m"]
                )
            else:
                cones = tuple(
                    (float(x), float(y))
                    for x, y in cones
                )

            cone_radius = float(
                layout["cone_radius_m"]
            )

            def point_segment_distance(
                px, py,
                ax, ay,
                bx, by,
            ):
                dx = bx - ax
                dy = by - ay
                l2 = dx * dx + dy * dy

                if l2 <= 1.0e-12:
                    return math.hypot(
                        px - ax,
                        py - ay,
                    )

                t = (
                    (px - ax) * dx
                    + (py - ay) * dy
                ) / l2

                t = max(
                    0.0,
                    min(1.0, t),
                )

                qx = ax + t * dx
                qy = ay + t * dy

                return math.hypot(
                    px - qx,
                    py - qy,
                )

            data = [100] * (
                width * height
            )

            for iy in range(height):
                y = (
                    y0
                    + (iy + 0.5) * resolution
                )

                for ix in range(width):
                    x = (
                        x0
                        + (ix + 0.5) * resolution
                    )

                    on_start_pad = (
                        sx0 <= x <= sx1
                        and sy0 <= y <= sy1
                    )

                    road_distance = min(
                        point_segment_distance(
                            x,
                            y,
                            a[0],
                            a[1],
                            b[0],
                            b[1],
                        )
                        for a, b in zip(
                            centerline[:-1],
                            centerline[1:],
                        )
                    )

                    on_road = (
                        road_distance
                        <= road_half
                    )

                    on_extra_free = any(
                        rx0 <= x <= rx1
                        and ry0 <= y <= ry1
                        for rx0, rx1, ry0, ry1
                        in free_rectangles
                    )

                    if not (
                        on_start_pad
                        or on_road
                        or on_extra_free
                    ):
                        continue

                    blocked = any(
                        math.hypot(
                            x - cx,
                            y - cy,
                        )
                        <= cone_radius
                        for cx, cy in cones
                    )

                    if not blocked:
                        data[
                            iy * width + ix
                        ] = 0

            grid = OccupancyGrid()

            grid.header.frame_id = "map"
            grid.header.stamp = (
                self.get_clock()
                .now()
                .to_msg()
            )

            grid.info.map_load_time = (
                grid.header.stamp
            )

            grid.info.resolution = resolution
            grid.info.width = width
            grid.info.height = height

            grid.info.origin.position.x = x0
            grid.info.origin.position.y = y0
            grid.info.origin.orientation.w = 1.0

            grid.data = data

            return grid


    rclpy.init()
    node = NodeImpl()

    result = {
        "schema_version":
            "mssr.rc_car_nav2_track_result.v1",
        "seed": args.seed,
        "track_profile":
            layout["track_profile"],
        "known_environment": layout,
        "goal_xyyaw": [
            float(v)
            for v in layout["goal_xyyaw"]
        ],
        "explicit_goal_mode": bool(
            explicit_goal_mode or explicit_route_mode
        ),
        "waypoints_xyyaw": [list(pose) for pose in composite_route],
        "success": False,
    }

    try:

        print(
            f"track={layout['track_profile']} "
            f"curve={layout['has_curve']} "
            f"cones={len(layout['cone_centers_xy_m'])}"
        )

        grid = node.make_map()

        for _ in range(20):
            grid.header.stamp = (
                node.get_clock()
                .now()
                .to_msg()
            )

            node.map_pub.publish(grid)

            rclpy.spin_once(
                node,
                timeout_sec=0.10,
            )

        if not node.client.wait_for_server(
            timeout_sec=60.0
        ):
            raise RuntimeError(
                (
                    "NavigateThroughPoses unavailable"
                    if explicit_route_mode
                    else "NavigateToPose unavailable"
                )
            )

        gx, gy, gyaw = [
            float(v)
            for v in layout["goal_xyyaw"]
        ]

        def stamped_pose(x_m, y_m, yaw_rad):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = node.get_clock().now().to_msg()
            pose.pose.position.x = x_m
            pose.pose.position.y = y_m
            pose.pose.orientation.z = math.sin(0.5 * yaw_rad)
            pose.pose.orientation.w = math.cos(0.5 * yaw_rad)
            return pose

        if explicit_route_mode:
            goal = NavigateThroughPoses.Goal()
            goal.poses = [stamped_pose(*pose) for pose in composite_route]
        else:
            goal = NavigateToPose.Goal()
            goal.pose = stamped_pose(gx, gy, gyaw)

            rc_car_bt = (
                SCRIPT_DIR.parent.parent
                / "mssr_ws"
                / "src"
                / "mssr_expert"
                / "config"
                / "smores_rc_car_nav_to_pose.xml"
            )

            if not rc_car_bt.is_file():
                raise RuntimeError(
                    f"RC-Car Nav2 behavior tree not found: {rc_car_bt}"
                )

            goal.behavior_tree = str(rc_car_bt)

        node.status(
            False,
            False,
            0.0,
            (
                "Explicit Nav2 pose started."
                if explicit_goal_mode
                else "Composite Nav2 waypoint route started."
                if explicit_route_mode
                else "Procedural RC track started."
            ),
        )

        def feedback_cb(msg):
            remaining = float(
                getattr(
                    msg.feedback,
                    "distance_remaining",
                    1.0,
                )
            )

            # Feedback is informative; exact terminal
            # success still comes from Nav2 itself.
            progress = max(
                0.0,
                min(
                    0.99,
                    1.0
                    - remaining
                    / max(3.0, remaining),
                ),
            )

            node.status(
                False,
                False,
                progress,
                (
                    "Following explicit Nav2 pose."
                    if explicit_goal_mode
                    else "Following composite Nav2 waypoints."
                    if explicit_route_mode
                    else "Following Nav2 track path."
                ),
            )
            node._feedback_count += 1
            if node._feedback_count % 5 == 0:
                node.record_dataset(progress, False, False, "Nav2 route active")

        send = node.client.send_goal_async(
            goal,
            feedback_callback=feedback_cb,
        )

        rclpy.spin_until_future_complete(
            node,
            send,
            timeout_sec=60.0,
        )

        handle = send.result()

        if handle is None or not handle.accepted:
            raise RuntimeError(
                "Nav2 rejected track goal"
            )

        future = handle.get_result_async()

        deadline = (
            time.monotonic()
            + args.action_timeout_s
        )

        physical_finish_success = False
        coarse_goal_success = False
        coarse_goal_metrics = None

        def coarse_explicit_goal_metrics():
            if (
                not explicit_goal_mode
                or args.accept_position_m is None
                or args.accept_yaw_rad is None
                or node._odom_pose is None
            ):
                return None

            x_m, y_m, yaw_rad = node._odom_pose

            position_error_m = math.hypot(
                x_m - gx,
                y_m - gy,
            )

            yaw_error_rad = math.atan2(
                math.sin(gyaw - yaw_rad),
                math.cos(gyaw - yaw_rad),
            )

            return {
                "position_error_m": position_error_m,
                "yaw_error_rad": yaw_error_rad,
                "accepted": (
                    position_error_m
                    <= float(args.accept_position_m)
                    and abs(yaw_error_rad)
                    <= float(args.accept_yaw_rad)
                ),
            }

        while (
            rclpy.ok()
            and not future.done()
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.10,
            )

            coarse_goal_metrics = (
                coarse_explicit_goal_metrics()
            )

            if (
                coarse_goal_metrics is not None
                and coarse_goal_metrics["accepted"]
            ):
                coarse_goal_success = True

                node.get_logger().warn(
                    "COARSE NAV2 GOAL ACCEPTED: "
                    f"position_error="
                    f"{coarse_goal_metrics['position_error_m']:.3f}m, "
                    f"yaw_error="
                    f"{math.degrees(coarse_goal_metrics['yaw_error_rad']):+.1f}deg"
                )

                cancel_future = handle.cancel_goal_async()

                rclpy.spin_until_future_complete(
                    node,
                    cancel_future,
                    timeout_sec=2.0,
                )

                rclpy.spin_until_future_complete(
                    node,
                    future,
                    timeout_sec=2.0,
                )

                break

            if (
                not (explicit_goal_mode or explicit_route_mode)
                and node.front_reached_finish()
            ):
                physical_finish_success = True

                metrics = node.finish_metrics()

                node.get_logger().warn(
                    "PHYSICAL FINISH SUCCESS: "
                    "RC-Car8 front reached finish line; "
                    f"progress="
                    f"{metrics['front_progress_past_finish_m']:+.3f}m"
                )

                cancel_future = (
                    handle.cancel_goal_async()
                )

                rclpy.spin_until_future_complete(
                    node,
                    cancel_future,
                    timeout_sec=2.0,
                )

                # Give the action server a moment to report cancellation.
                rclpy.spin_until_future_complete(
                    node,
                    future,
                    timeout_sec=2.0,
                )

                break

        # The action could finish between loop iterations, so perform
        # one last physical tolerance check before interpreting status.
        if not coarse_goal_success:
            coarse_goal_metrics = (
                coarse_explicit_goal_metrics()
            )
            coarse_goal_success = bool(
                coarse_goal_metrics is not None
                and coarse_goal_metrics["accepted"]
            )

        if coarse_goal_success:
            if future.done() and future.result() is not None:
                status = int(
                    future.result().status
                )
            else:
                status = int(
                    GoalStatus.STATUS_CANCELING
                )

            success = True
            completion_source = "coarse_explicit_goal"

        elif physical_finish_success:
            if future.done() and future.result() is not None:
                status = int(
                    future.result().status
                )
            else:
                status = int(
                    GoalStatus.STATUS_CANCELING
                )

            success = True
            completion_source = (
                "physical_front_finish_line"
            )

        else:
            # One final physical check before declaring timeout.
            if (
                not (explicit_goal_mode or explicit_route_mode)
                and node.front_reached_finish()
            ):
                success = True
                physical_finish_success = True
                status = int(
                    GoalStatus.STATUS_CANCELING
                )
                completion_source = (
                    "physical_front_finish_line"
                )

            elif not future.done():
                handle.cancel_goal_async()

                raise TimeoutError(
                    "RC track navigation timeout"
                )

            else:
                wrapped = future.result()

                status = int(wrapped.status)

                # Nav2 success remains valid too. With the current
                # goal at the finish line this necessarily places the
                # physical front beyond the stripe.
                success = (
                    status
                    == GoalStatus.STATUS_SUCCEEDED
                )

                completion_source = (
                    "nav2_goal"
                    if success
                    else "nav2_failure"
                )

        stop = Twist()

        for _ in range(10):
            node.cmd_pub.publish(stop)
            rclpy.spin_once(
                node,
                timeout_sec=0.06,
            )

        message = (
            "RC-Car8 front reached physical finish line."
            if success
            and completion_source == "physical_front_finish_line"
            else (
                (
                    "Explicit Nav2 target reached."
                    if explicit_goal_mode
                    else "Composite Nav2 waypoint route completed."
                    if explicit_route_mode
                    else "RC track completed."
                )
                if success
                else f"Nav2 status={status}"
            )
        )

        if completion_source == "coarse_explicit_goal":
            message = (
                "RC-Car8 entered coarse "
                "pre-reconfiguration goal region."
            )

        for _ in range(5):
            node.status(
                True,
                success,
                1.0 if success else 0.0,
                message,
            )

            rclpy.spin_once(
                node,
                timeout_sec=0.06,
            )

        node.record_dataset(
            1.0 if success else 0.0,
            True,
            success,
            message,
        )

        result["success"] = success
        result["status"] = status
        result["message"] = message
        result["completion_source"] = completion_source
        result["finish_validation"] = node.finish_metrics()
        result["coarse_goal_validation"] = coarse_goal_metrics

        rc = 0 if success else 1

    except Exception as exc:

        result["error"] = str(exc)

        try:
            for _ in range(5):
                node.status(
                    True,
                    False,
                    0.0,
                    str(exc),
                )

                rclpy.spin_once(
                    node,
                    timeout_sec=0.06,
                )
        except Exception:
            pass

        rc = 1

    finally:

        if args.result_json:
            args.result_json.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            args.result_json.write_text(
                json.dumps(
                    result,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

        node.destroy_node()
        rclpy.shutdown()

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
