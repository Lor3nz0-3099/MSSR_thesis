"""Bounded orbit intent; only the Isaac runtime adapter applies a view."""
from dataclasses import dataclass
import math


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class CameraOrbit:
    azimuth_rad: float
    elevation_rad: float
    radius_m: float

    def view(self, center):
        if len(center) != 3:
            raise ValueError("camera center must have three coordinates")
        target = tuple(_finite(value, "camera center") for value in center)
        horizontal = self.radius_m * math.cos(self.elevation_rad)
        eye = (target[0] + horizontal * math.cos(self.azimuth_rad),
               target[1] + horizontal * math.sin(self.azimuth_rad),
               target[2] + self.radius_m * math.sin(self.elevation_rad))
        return eye, target


class CameraController:
    def __init__(self, *, radius_m=2.0, yaw_rate_deg_s=90.0, pitch_rate_deg_s=60.0,
                 elevation_min_deg=5.0, elevation_max_deg=85.0,
                 initial_azimuth_deg=-45.0, initial_elevation_deg=35.0,
                 max_dt_s=0.1) -> None:
        self.radius = _finite(radius_m, "radius")
        self.yaw_rate = math.radians(_finite(yaw_rate_deg_s, "yaw rate"))
        self.pitch_rate = math.radians(_finite(pitch_rate_deg_s, "pitch rate"))
        self.minimum = math.radians(_finite(elevation_min_deg, "minimum elevation"))
        self.maximum = math.radians(_finite(elevation_max_deg, "maximum elevation"))
        self.max_dt = _finite(max_dt_s, "max delta")
        if self.radius <= 0 or self.max_dt <= 0 or self.yaw_rate < 0 or self.pitch_rate < 0:
            raise ValueError("camera radius/delta must be positive and rates nonnegative")
        if not 0 < self.minimum < self.maximum < math.pi / 2:
            raise ValueError("camera elevation bounds must be within (0,90) degrees")
        self.azimuth = math.radians(_finite(initial_azimuth_deg, "initial azimuth"))
        self.elevation = math.radians(_finite(initial_elevation_deg, "initial elevation"))
        self._bound()

    def _bound(self):
        self.azimuth = (self.azimuth + math.pi) % (2 * math.pi) - math.pi
        self.elevation = min(self.maximum, max(self.minimum, self.elevation))

    def step(self, sample, dt: float) -> CameraOrbit:
        delta = _finite(dt, "wall delta")
        if delta < 0:
            raise ValueError("wall delta must be nonnegative")
        if sample.connected:
            x, y = _finite(sample.left_x, "left X"), _finite(sample.left_y, "left Y")
            if not -1 <= x <= 1 or not -1 <= y <= 1:
                raise ValueError("camera stick values must be normalized")
            delta = min(delta, self.max_dt)
            self.azimuth += x * self.yaw_rate * delta
            self.elevation += y * self.pitch_rate * delta
            self._bound()
        return CameraOrbit(self.azimuth, self.elevation, self.radius)
