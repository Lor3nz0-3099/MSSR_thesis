"""Certified source-frame contracts, outside the learned policy inputs.

Providers own source certification; E1 owns all transformation mathematics.
This module reads no source files and defines no backend-specific resolver.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Protocol

from .bc_v1_contract import PolicyInputs
from .features import PhysicalFrameConfig, SourceFrameTransform


_LOCAL_FRAMES = frozenset({
    "module_body", "body_local", "endpoint_a_body", "connection_intrinsic",
    "resource_local", "intrinsic",
})


def _text(value: str, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _records(values: tuple, cls: type, name: str) -> None:
    if type(values) is not tuple or any(type(value) is not cls for value in values):
        raise TypeError(f"{name} must be an immutable tuple of {cls.__name__}")


def _identity(values: tuple[tuple[str, str], ...], name: str) -> None:
    if type(values) is not tuple or any(type(pair) is not tuple or len(pair) != 2 for pair in values):
        raise TypeError(f"{name} must be an immutable tuple of string pairs")
    for key, value in values:
        _text(key, name + " key")
        _text(value, name + " value")
    if len({key for key, _ in values}) != len(values):
        raise ValueError(f"Duplicate {name} key")


@dataclass(frozen=True)
class BridgeEvidence:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.path, "evidence path")
        if (type(self.sha256) is not str or len(self.sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.sha256)):
            raise ValueError("evidence sha256 must be a lowercase hexadecimal SHA256")


@dataclass(frozen=True)
class SourceContractCertificate:
    """Evidence-backed source certification; world identity needs no certificate."""

    contract_id: str
    evidence: tuple[BridgeEvidence, ...]

    def __post_init__(self) -> None:
        _text(self.contract_id, "contract_id")
        _records(self.evidence, BridgeEvidence, "evidence")
        if not self.evidence:
            raise ValueError("source contract certificate requires nonempty evidence")


@dataclass(frozen=True)
class FrameSourceContext:
    backend_id: str
    source_identity: tuple[tuple[str, str], ...]
    certificates: tuple[SourceContractCertificate, ...]

    def __post_init__(self) -> None:
        _text(self.backend_id, "backend_id")
        _identity(self.source_identity, "source_identity")
        _records(self.certificates, SourceContractCertificate, "certificates")


@dataclass(frozen=True)
class BridgeResolution:
    source_frame: str
    target_frame: str
    scope_id: str
    transform: SourceFrameTransform | None
    certified: bool
    reason: str
    contract_id: str | None
    evidence: tuple[BridgeEvidence, ...]

    def __post_init__(self) -> None:
        for name in ("source_frame", "target_frame", "scope_id", "reason"):
            _text(getattr(self, name), name)
        if self.source_frame in _LOCAL_FRAMES or self.target_frame in _LOCAL_FRAMES:
            raise ValueError("local/intrinsic geometry is not a global bridge candidate")
        if type(self.certified) is not bool:
            raise TypeError("certified must be boolean")
        if self.contract_id is not None:
            _text(self.contract_id, "contract_id")
        _records(self.evidence, BridgeEvidence, "evidence")
        if not self.certified:
            if self.transform is not None:
                raise ValueError("uncertified resolution must not carry an E1 transform")
            return
        if self.target_frame != "world":
            raise ValueError("certified bridge target_frame must be world for current E1")
        if self.source_frame == "world":
            if self.transform is not None:
                raise ValueError("world identity must be implicit, without a redundant transform")
            return
        if self.contract_id is None:
            raise ValueError("certified named frame requires a nonempty contract_id")
        if not self.evidence:
            raise ValueError("certified named frame requires nonempty evidence")
        if type(self.transform) is not SourceFrameTransform:
            raise ValueError("certified named frame requires an explicit E1 transform")
        if self.transform.frame_id != self.source_frame:
            raise ValueError("transform frame_id differs from resolution source_frame")
        # Reuse E1's finite/proper-rotation checks; do not reproduce its math.
        SourceFrameTransform(self.transform.frame_id, self.transform.translation, self.transform.rotation)


@dataclass(frozen=True)
class SourceFrameBridgeBatch:
    resolutions: tuple[BridgeResolution, ...]
    provider_identity: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _records(self.resolutions, BridgeResolution, "resolutions")
        _identity(self.provider_identity, "provider_identity")
        if self.resolutions and not self.provider_identity:
            raise ValueError("nonempty bridge batch requires provider_identity")


class SourceFrameProvider(Protocol):
    """Only approved current physical inputs and source certificates enter.

    Backend implementations supply resolution and serializable policy identity.
    No label, future state, supervision mask or behavioral stage argument exists.
    """

    def resolve(self, inputs: PolicyInputs, source: FrameSourceContext) -> SourceFrameBridgeBatch:
        ...

    def identity(self) -> Mapping:
        ...


@dataclass(frozen=True)
class PipelineDiagnostic:
    """Audit record for the later validator; never a learned input."""

    stage: str
    source_scope: str
    source_name: str
    classification: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("stage", "source_scope", "source_name", "classification", "reason"):
            _text(getattr(self, name), name)


def frame_config_for(batch: SourceFrameBridgeBatch) -> PhysicalFrameConfig:
    """Compile coherent certified bridges into E1's existing world config.

    Unknown/uncertified named frames supply no transform. Contradictory results
    raise ValueError; E1 has one transform per frame, so disagreement even across
    scopes cannot be represented safely. Repeated coherent proofs compile once.
    An uncertified world is rejected because E1 treats world as implicit identity.
    """
    if type(batch) is not SourceFrameBridgeBatch:
        raise TypeError("Expected SourceFrameBridgeBatch")
    by_scope, by_frame, transforms = {}, {}, {}
    for result in batch.resolutions:
        signature = (result.target_frame, result.certified, result.contract_id, result.transform)
        key = (result.source_frame, result.scope_id)
        if key in by_scope and by_scope[key] != signature:
            raise ValueError(f"Conflicting bridge certificates for frame/scope {key!r}")
        by_scope[key] = signature
        # Scope is retained in audit, but cannot select distinct transforms in E1.
        physical = (result.target_frame, result.certified, result.transform)
        if result.source_frame in by_frame and by_frame[result.source_frame] != physical:
            raise ValueError(f"Conflicting bridge resolutions for frame {result.source_frame!r}")
        by_frame[result.source_frame] = physical
        if not result.certified:
            if result.source_frame == "world":
                raise ValueError("uncertified world cannot use E1 implicit world identity")
            continue
        if result.transform is not None:
            transforms[result.source_frame] = result.transform
    return PhysicalFrameConfig(transforms=tuple(transforms[name] for name in sorted(transforms)))


def _json_value(value):
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("Bridge policy keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError("Bridge policy must contain only serializable configuration")


def bridge_policy_fingerprint(policy: Mapping) -> str:
    """Canonical JSON SHA256, following the existing cache hashing convention.

    Mapping insertion order is irrelevant; sequence/component order is retained.
    The caller supplies provider/revision/config/source/evidence policies. This
    helper neither enumerates derived sample transforms nor imports the cache.
    """
    if not isinstance(policy, Mapping):
        raise TypeError("Bridge policy must be a Mapping")
    payload = json.dumps(_json_value(policy), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
