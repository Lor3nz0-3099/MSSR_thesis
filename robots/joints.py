"""Shared joint type definitions for module attachments."""

from __future__ import annotations

from enum import Enum


class JointType(str, Enum):
    """Physical joint type requested for a magnetic attachment."""

    RIGID = "rigid"
    SPHERICAL = "spherical"
    HINGE = "hinge"
