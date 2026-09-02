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
    rc_car_planar_obstacle_layout,
    sample_rc_car_planar_spec,
)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--action-timeout-s", type=float, default=900.0)
    p.add_argument("--result-json", type=Path)
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


def main():
    args = parser().parse_args()

    import rclpy

    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped, Twist
    from nav2_msgs.action import NavigateToPose
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from std_msgs.msg import String

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
                NavigateToPose,
                "navigate_to_pose",
            )

            self.cmd_pub = self.create_publisher(
                Twist,
                "/cmd_vel",
                10,
            )

            self._odom_pose = None

            self.create_subscription(
                Odometry,
                "/odom",
                self._on_odom,
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

                    if not (
                        on_start_pad
                        or on_road
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
                "NavigateToPose unavailable"
            )

        gx, gy, gyaw = [
            float(v)
            for v in layout["goal_xyyaw"]
        ]

        goal = NavigateToPose.Goal()

        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = (
            node.get_clock()
            .now()
            .to_msg()
        )

        goal.pose.pose.position.x = gx
        goal.pose.pose.position.y = gy

        goal.pose.pose.orientation.z = (
            math.sin(0.5 * gyaw)
        )

        goal.pose.pose.orientation.w = (
            math.cos(0.5 * gyaw)
        )

        node.status(
            False,
            False,
            0.0,
            "Procedural RC track started.",
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
                "Following Nav2 track path.",
            )

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

        while (
            rclpy.ok()
            and not future.done()
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.10,
            )

            if node.front_reached_finish():
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

        if physical_finish_success:
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
            if node.front_reached_finish():
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
                "RC track completed."
                if success
                else f"Nav2 status={status}"
            )
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

        result["success"] = success
        result["status"] = status
        result["message"] = message
        result["completion_source"] = completion_source
        result["finish_validation"] = node.finish_metrics()

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
