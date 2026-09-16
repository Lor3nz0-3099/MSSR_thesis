#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def angle_error(a: float, b: float) -> float:
    return math.atan2(
        math.sin(a - b),
        math.cos(a - b),
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Post-Nav2 RC-Car8 reverse-arc alignment before "
            "gap reconfiguration."
        )
    )
    p.add_argument(
        "--target-yaw-rad",
        type=float,
        default=0.0,
    )
    return p


class Monitor(Node):

    def __init__(self):
        super().__init__("gap_rc_alignment")

        self.odom = None

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.create_subscription(
            Odometry,
            "/odom",
            self.on_odom,
            20,
        )

    def on_odom(self, msg):
        self.odom = msg

    def spin_until(self, predicate, timeout, message):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            rclpy.spin_once(
                self,
                timeout_sec=0.10,
            )

            if predicate():
                return

        raise TimeoutError(message)

    def pose(self):
        if self.odom is None:
            raise RuntimeError("/odom unavailable")

        p = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation

        yaw = math.atan2(
            2.0 * (q.w*q.z + q.x*q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z),
        )

        return float(p.x), float(p.y), yaw


def align_rc_for_gap(
    monitor: Monitor,
    target_yaw_rad: float,
) -> None:
    """
    Same post-Nav2 reverse-arc controller used by the button expert.

    The only semantic difference is the desired-heading source:
    button -> bearing relative to the live button;
    gap    -> fixed world-frame reconfiguration heading.
    """

    monitor.spin_until(
        lambda: monitor.odom is not None,
        10,
        "/odom unavailable for RC final alignment.",
    )

    x0, y0, yaw0 = monitor.pose()

    def desired_yaw(x, y):
        # Gap reconfiguration requires a fixed world-frame heading.
        return target_yaw_rad

    desired0 = desired_yaw(x0, y0)
    err0 = angle_error(
        desired0,
        yaw0,
    )

    print()
    print("============================================================")
    print(" RC-CAR FINAL REVERSE-ARC ALIGNMENT")
    print("============================================================")
    print(
        f"yaw before  = {math.degrees(yaw0):+.2f} deg"
    )
    print(
        f"desired yaw = {math.degrees(desired0):+.2f} deg"
    )
    print(
        f"yaw error   = {math.degrees(err0):+.2f} deg"
    )

    # IDENTICAL TO BUTTON.
    tol = math.radians(0.5)
    start = time.monotonic()
    stable = 0
    last_print = 0.0

    try:
        # IDENTICAL TO BUTTON.
        while time.monotonic() - start < 75.0:
            rclpy.spin_once(
                monitor,
                timeout_sec=0.04,
            )

            x, y, yaw = monitor.pose()

            desired = desired_yaw(
                x,
                y,
            )

            err = angle_error(
                desired,
                yaw,
            )

            # IDENTICAL TO BUTTON:
            # stop immediately once yaw enters tolerance.
            if abs(err) <= tol:
                break

            # IDENTICAL TO BUTTON.
            wz = max(
                -0.30,
                min(
                    0.30,
                    1.2 * err,
                ),
            )

            # IDENTICAL TO BUTTON.
            if (
                abs(wz) < 0.07
                and abs(err) > tol
            ):
                wz = math.copysign(
                    0.07,
                    err,
                )

            # IDENTICAL TO BUTTON:
            # RC-Car8 cannot reliably pivot in place.
            # Use the same small reverse arc.
            travel = math.hypot(
                x - x0,
                y - y0,
            )

            cmd = Twist()
            cmd.linear.x = -0.025
            cmd.angular.z = wz

            monitor.cmd_pub.publish(cmd)

            now = time.monotonic()

            if now - last_print >= 1.0:
                print(
                    f"yaw={math.degrees(yaw):+7.2f} deg  "
                    f"desired={math.degrees(desired):+7.2f} deg  "
                    f"error={math.degrees(err):+6.2f} deg  "
                    f"wz={wz:+.3f}"
                )
                last_print = now

        else:
            raise RuntimeError(
                "RC final yaw alignment timeout."
            )

    finally:
        # IDENTICAL TO BUTTON.
        stop = Twist()

        for _ in range(15):
            monitor.cmd_pub.publish(stop)
            rclpy.spin_once(
                monitor,
                timeout_sec=0.04,
            )

    xf, yf, yawf = monitor.pose()
    errf = angle_error(
        target_yaw_rad,
        yawf,
    )

    print(
        f"yaw after   = {math.degrees(yawf):+.2f} deg"
    )
    print(
        f"yaw error   = {math.degrees(errf):+.2f} deg"
    )
    print(
        f"travel      = {math.hypot(xf-x0, yf-y0):.3f} m"
    )
    print("ALIGNMENT SUCCESS")


def main() -> int:
    args = parser().parse_args()

    rclpy.init()
    monitor = Monitor()

    try:
        align_rc_for_gap(
            monitor,
            float(args.target_yaw_rad),
        )
        return 0

    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
