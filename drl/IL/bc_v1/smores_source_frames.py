"""SMORES CURRENT planar task-frame bridges; source readers remain separate.

The stage_local contract attests metres, radians, world-Z yaw, and applicability
of the supplied evidence to the current source identity and D2 task-frame data.
FrameSourceContext must come from a verifying source reader/caller: merely
constructing a certificate does not verify a file digest. The source reader
verifies small evidence files; resolvers consume receipts without I/O. No task
geometry is transformed here and no evidence is invented.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from .bc_v1_contract import NumericFeature, PolicyInputs
from .features import SourceFrameTransform
from .source_frame_bridges import (
    BridgeEvidence, BridgeResolution, FrameSourceContext, SourceContractCertificate, SourceFrameBridgeBatch,
    bridge_policy_fingerprint,
)


STAGE_LOCAL_CONTRACT_ID = "mssr.smores.current_task_frame.planar.v1"
LEGACY_RC_MAP_CONTRACT_ID = "mssr.smores.isaac_nav2.legacy_rc_map.v1"
_ORIGIN = "world.task_frame.origin_xyz_m"
_YAW = "world.task_frame.yaw_rad"


def _scope(source: FrameSourceContext) -> str:
    return "current-source:" + bridge_policy_fingerprint({
        "backend_id": source.backend_id, "source_identity": dict(source.source_identity),
    })


def _current_components(inputs: PolicyInputs, name: str, size: int) -> tuple[float, ...]:
    features = [feature for feature in inputs.context.static_geometry if feature.name == name]
    if not features:
        raise ValueError(f"missing current {name}")
    if len(features) != 1:
        raise ValueError(f"conflicting/duplicate current features: {name}")
    feature = features[0]
    if type(feature) is not NumericFeature or feature.valid is not True:
        raise ValueError(f"invalid current feature: {name}")
    if type(feature.values) is not tuple or len(feature.values) != size:
        raise ValueError(f"malformed current dimensions: {name}")
    if any(type(value) not in (float, int) for value in feature.values):
        raise ValueError(f"non-numeric current components: {name}")
    try:
        values = tuple(float(value) for value in feature.values)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"nonfinite current components: {name}") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"nonfinite current components: {name}")
    return values


def resolve_stage_local(inputs: PolicyInputs, source: FrameSourceContext) -> BridgeResolution:
    """Resolve only current WORLD origin/yaw under the explicit planar contract.

    Missing, malformed, ambiguous or uncertified data returns unresolved with a
    reason; no origin/yaw default exists. The certificate's evidence is passed
    through deterministically. Digest disagreement for one path fails closed.
    """
    if type(inputs) is not PolicyInputs or not isinstance(source, FrameSourceContext):
        raise TypeError("Expected PolicyInputs and FrameSourceContext")
    certificates = tuple(cert for cert in source.certificates if cert.contract_id == STAGE_LOCAL_CONTRACT_ID)
    evidence = tuple(sorted({e for cert in certificates for e in cert.evidence}, key=lambda e: (e.path, e.sha256)))
    contract_id = STAGE_LOCAL_CONTRACT_ID if certificates else None
    scope_id = _scope(source)

    def unresolved(reason: str) -> BridgeResolution:
        return BridgeResolution("stage_local", "world", scope_id, None, False, reason, contract_id, evidence)

    if source.backend_id != "smores_ep":
        return unresolved("stage_local planar source contract requires SMORES backend smores_ep")
    if not source.source_identity:
        return unresolved("missing current source_identity for stage_local certification")
    if not certificates or not evidence:
        return unresolved("missing evidence-backed current stage_local planar source contract")
    digests = {}
    for item in evidence:
        if item.path in digests and digests[item.path] != item.sha256:
            return unresolved(f"conflicting current evidence digest: {item.path}")
        digests[item.path] = item.sha256
    try:
        origin = _current_components(inputs, _ORIGIN, 3)
        yaw = _current_components(inputs, _YAW, 1)[0]
        cosine, sine = math.cos(yaw), math.sin(yaw)
        transform = SourceFrameTransform("stage_local", origin, (
            cosine, -sine, 0., sine, cosine, 0., 0., 0., 1.,
        ))
    except ValueError as exc:
        return unresolved(str(exc))
    return BridgeResolution(
        "stage_local", "world", scope_id, transform, True,
        "current WORLD task-frame origin/yaw under certified SMORES planar source contract",
        contract_id, evidence,
    )


class SmoresSourceFrameProvider:
    """Current planar frame and independently certified historical RC map.

    repo_root is retained for future source-reader composition, without I/O.
    policy is serializable configuration: its fingerprint is captured at creation
    and does not change if the caller later mutates the original mapping.
    Neither current-source receipts nor absolute storage paths enter identity.
    """

    def __init__(self, repo_root: Path, policy: Mapping):
        self.repo_root = Path(repo_root)
        self._policy_fingerprint = bridge_policy_fingerprint(policy)

    def identity(self) -> Mapping:
        return {
            "provider_id": "mssr.bc_v1.smores_source_frames",
            "revision": "1.1.1",
            "stage_local_contract_id": STAGE_LOCAL_CONTRACT_ID,
            "legacy_rc_map_contract_id": LEGACY_RC_MAP_CONTRACT_ID,
            "policy_fingerprint": self._policy_fingerprint,
        }

    def resolve(self, inputs: PolicyInputs, source: FrameSourceContext) -> SourceFrameBridgeBatch:
        if type(inputs) is not PolicyInputs or not isinstance(source, FrameSourceContext):
            raise TypeError("Expected PolicyInputs and FrameSourceContext")
        scope_id = _scope(source)
        results = [BridgeResolution(
            "world", "world", scope_id, None, True,
            "typed current physical graph declares WORLD; implicit identity", None, (),
        )]
        # Only task/environment named global channels are bridge candidates.
        # Module-body connectors, B-in-A edges and resources are never visited.
        features = (*inputs.context.static_geometry, *inputs.observation.features)
        frames = {feature.name.partition(".")[0] for feature in features}
        if "stage_local" in frames:
            results.append(resolve_stage_local(inputs, source))
        if "map" in frames:
            result = resolve_legacy_rc_map(source)
            if result.certified and dict(source.source_identity).get("source_policy_fingerprint") != self._policy_fingerprint:
                result = replace(result, certified=False, transform=None, reason="legacy map source/provider policy identity mismatch")
            results.append(result)
        return SourceFrameBridgeBatch(tuple(results), tuple(sorted(self.identity().items())))


def resolve_legacy_rc_map(source: FrameSourceContext) -> BridgeResolution:
    """Consume a reader-verified receipt without reopening any source file."""
    if not isinstance(source, FrameSourceContext):
        raise TypeError("Expected FrameSourceContext")
    certificates = tuple(c for c in source.certificates if c.contract_id == LEGACY_RC_MAP_CONTRACT_ID)
    evidence = tuple(sorted({e for c in certificates for e in c.evidence}, key=lambda e: (e.path, e.sha256)))
    identity = dict(source.source_identity)
    fingerprint = bridge_policy_fingerprint({"evidence": [(e.path, e.sha256) for e in evidence]})
    certified = (source.backend_id == "smores_ep" and bool(evidence)
                 and identity.get("legacy_rc_contract_revision") == "1.0.0"
                 and identity.get("source_kind") == "expert"
                 and identity.get("legacy_rc_evidence_fingerprint") == fingerprint)
    return BridgeResolution(
        "map", "world", _scope(source), SourceFrameTransform("map", (0., 0., 0.)) if certified else None,
        certified, "verified historical SMORES/Isaac/Nav2 RC map convention" if certified else
        identity.get("certification_reason", "missing applicable verified legacy RC source contract"),
        LEGACY_RC_MAP_CONTRACT_ID if certificates else None, evidence,
    )


def _relative_path(path, root: Path) -> str:
    if type(path) is not str or not path:
        raise ValueError("missing source evidence path")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return str(candidate.resolve().relative_to(root))
    except ValueError as exc:
        raise ValueError("source evidence path outside repository") from exc


def _pinned_document(root: Path, path: str, digest: str, evidence: dict, *, document=True):
    """Read only bounded small proof files; no JSONL/trajectory hashing."""
    relative = _relative_path(path, root)
    expected = BridgeEvidence(relative, digest)
    target = root / relative
    if target.suffix == ".jsonl" or not target.is_file() or target.stat().st_size > 2 * 1024 * 1024:
        raise ValueError(f"missing/unsupported small source evidence: {relative}")
    signature = target.stat()
    content = target.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected.sha256:
        raise ValueError(f"source evidence digest mismatch: {relative}")
    after = target.stat()
    if (signature.st_ino, signature.st_size, signature.st_mtime_ns, signature.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"source evidence changed while reading: {relative}")
    evidence[relative] = expected
    if not document:
        return None
    try:
        result = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"malformed source evidence: {relative}") from exc
    if not isinstance(result, Mapping):
        raise ValueError(f"source evidence must be a document: {relative}")
    return result


def _physical_vector(value, size: int) -> tuple:
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError("malformed physical source evidence")
    return tuple(value)


def _verify_platform(environment: Mapping, course: Mapping) -> None:
    layout = environment.get("layout", {})
    if environment.get("layout_frame_id") != "map" or layout.get("frame_id") != "map" or course.get("frame_id") != "world":
        raise ValueError("contradictory MAP/WORLD source evidence declarations")
    bounds = _physical_vector(layout.get("platform_bounds_xy_m"), 4)
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("malformed MAP bounds")
    boxes = [box for box in course.get("collision_boxes", ()) if box.get("semantic") == "rc_car_planar_support_platform"]
    if len(boxes) != 1:
        raise ValueError("missing/ambiguous independent WORLD platform evidence")
    center, size = (_physical_vector(boxes[0].get(key), 3) for key in ("center_xyz_m", "size_xyz_m"))
    xy = ((bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2)
    lengths = (bounds[1] - bounds[0], bounds[3] - bounds[2])
    if (any(v <= 0 for v in size) or any(not math.isclose(a, b, rel_tol=0., abs_tol=1e-9)
            for a, b in zip((*xy, *lengths), (*center[:2], *size[:2])))):
        raise ValueError("contradictory independent MAP/WORLD platform evidence")
    for key, expected in (("center_xyz_m", center), ("size_xyz_m", size)):
        actual = _physical_vector(environment.get("platform", {}).get(key), 3)
        if any(not math.isclose(a, b, rel_tol=0., abs_tol=1e-9) for a, b in zip(actual, expected)):
            raise ValueError("contradictory environment/current WORLD platform evidence")


def _read_legacy_source_context(provenance: Mapping, metadata: Mapping, *, repo_root: Path, policy: Mapping) -> FrameSourceContext:
    """Verify the pinned legacy source chain using small current-source proofs.

    Caller composition must attach provenance['dataset_identity'] =
    {'manifest': active_collection_manifest_path, 'manifest_sha256': its_digest}.
    Raw Canonical provenance alone cannot identify a copied active collection;
    absent binding remains uncertified. No Canonical row or source is rewritten.

    Exact source paths/one-based rows join the frozen compact membership and
    phase descriptors; large streams are never reopened. The frozen source is
    trusted through that receipt, not rehashed per sample. Integrity caching can
    be added at source/episode scope by later runtime composition.
    """
    if not all(isinstance(value, Mapping) for value in (provenance, metadata, policy)):
        raise TypeError("Expected provenance, metadata and policy mappings")
    root = Path(repo_root).resolve()
    backend = metadata.get("backend_id", "unknown")

    def unverified(reason):
        return FrameSourceContext(backend, (("certification_reason", reason),), ())

    contract = policy.get("legacy_rc_map")
    refs = provenance.get("source_records", ())
    active = provenance.get("dataset_identity")
    if not contract or not refs or not active:
        return unverified("missing legacy source contract, exact references or active dataset identity")
    if backend not in {"unknown", "smores_ep"}:
        return unverified("backend is outside the historical SMORES legacy contract")
    if (policy.get("schema_version") != "mssr.bc_v1.source_frames.v1"
            or contract.get("contract_id") != LEGACY_RC_MAP_CONTRACT_ID or contract.get("revision") != "1.0.0"
            or contract.get("backend_id") != "smores_ep" or contract.get("source_kind") != "expert"
            or contract.get("certification_policy") != "pinned_source_membership_and_geometry_and_historical_launch"):
        raise ValueError("unsupported historical source certification policy")
    canonical_pin, compact_pin = contract["canonical"]["manifest"], contract["compact"]["manifest"]
    if (_relative_path(active.get("manifest"), root) != _relative_path(canonical_pin["path"], root)
            or active.get("manifest_sha256") != canonical_pin["sha256"]):
        return unverified("active dataset identity is not the approved immutable collection")
    proofs = {}
    canonical = _pinned_document(root, canonical_pin["path"], canonical_pin["sha256"], proofs)
    episodes = [entry for entry in canonical.get("episodes", ()) if entry.get("episode_id") == provenance.get("logical_episode_id")]
    if len(episodes) != 1 or episodes[0].get("kind") != "expert" or episodes[0].get("task") != "rc_car":
        return unverified("source episode is not a unique frozen RC expert member")
    link = canonical.get("source_dataset", {})
    if (canonical.get("dataset_name") != contract["canonical"]["dataset_name"]
            or _relative_path(link.get("manifest"), root) != _relative_path(compact_pin["path"], root)
            or link.get("manifest_sha256") != compact_pin["sha256"]):
        raise ValueError("contradictory Canonical-to-compact source evidence link")
    compact = _pinned_document(root, compact_pin["path"], compact_pin["sha256"], proofs)
    if compact.get("schema_version") != contract["compact"]["schema_version"]:
        raise ValueError("wrong compact source evidence schema")
    env_pin = episodes[0].get("environment_geometry")
    if not env_pin:
        return unverified("frozen source has no certified legacy MAP environment evidence")
    env_relative = _relative_path(env_pin["path"], root)
    directory = Path(env_relative).parent
    compact_root = Path(_relative_path(compact_pin["path"], root)).parent
    members = [entry for entry in compact.get("episodes", ())
               if compact_root / entry.get("path", "") == directory and entry.get("task") == "rc_car"]
    if len(members) != 1:
        raise ValueError("missing/ambiguous compact source membership evidence")
    catalog = contract["evidence_catalog"]

    def read_catalog(path):
        relative = str(path)
        if relative not in catalog:
            raise ValueError(f"missing pinned source evidence: {relative}")
        return _pinned_document(root, relative, catalog[relative], proofs)

    episode = read_catalog(directory / "manifest.json")
    if episode.get("schema_version") != "mssr.expert_compact_episode.v1" or episode.get("task") != "rc_car":
        raise ValueError("wrong compact episode source evidence")
    declared_env = episode.get("environment_geometry", {})
    if (directory / declared_env.get("path", "") != Path(env_relative)
            or declared_env.get("sha256") != env_pin["sha256"] or catalog.get(env_relative) != env_pin["sha256"]):
        raise ValueError("contradictory pinned environment source evidence")
    environment = read_catalog(env_relative)
    assembly = read_catalog(directory / "assembly_manifest.json")
    if (assembly.get("phase") != "assembly" or assembly.get("observation_constants", {}).get("robot_family") != "smores_ep"
            or episode.get("phases", {}).get("assembly", {}).get("observation_constants", {}).get("robot_family") != "smores_ep"):
        raise ValueError("missing independent SMORES backend source evidence")
    stream_receipts = set()
    for ref in refs:
        stream = Path(_relative_path(ref.get("_loader_source_path"), root))
        phase = stream.stem
        if stream.parent != directory or stream.suffix != ".jsonl" or phase not in members[0].get("phases", ()) or ref.get("_loader_source_kind") != "expert":
            raise ValueError("source stream not in approved compact membership")
        current = read_catalog(directory / (phase + "_manifest.json"))
        descriptor = current.get("compact", {})
        declared = episode.get("phases", {}).get(phase, {})
        if (current.get("schema_version") != "mssr.expert_compact_phase.v1" or current.get("phase") != phase
                or descriptor != declared.get("compact") or compact_root / descriptor.get("path", "") != stream):
            raise ValueError("contradictory exact phase/stream source evidence")
        index = ref.get("_loader_source_row")
        if type(index) is not int or not 1 <= index <= descriptor.get("records", 0):
            raise ValueError("exact source row outside frozen stream membership")
        _verify_platform(environment, current.get("observation_constants", {}).get("course", {}))
        stream_receipts.add((str(stream), descriptor["sha256"], index))
    launch = contract["launch"]
    if launch.get("odom_pose_frame") != "isaac_world" or launch.get("map_to_odom") != "identity":
        raise ValueError("historical launch convention evidence is not Isaac-world/map identity")
    _pinned_document(root, launch["path"], launch["sha256"], proofs, document=False)
    evidence = tuple(proofs[key] for key in sorted(proofs))
    identity = {
        "canonical_manifest_sha256": canonical_pin["sha256"], "compact_manifest_sha256": compact_pin["sha256"],
        "source_kind": "expert", "source_episode": episodes[0]["episode_id"],
        "stream_receipts": json.dumps(sorted(stream_receipts), separators=(",", ":")),
        "legacy_rc_contract_revision": "1.0.0", "source_policy_fingerprint": bridge_policy_fingerprint(policy),
        "legacy_rc_evidence_fingerprint": bridge_policy_fingerprint({"evidence": [(e.path, e.sha256) for e in evidence]}),
    }
    return FrameSourceContext("smores_ep", tuple(sorted(identity.items())), (
        SourceContractCertificate(LEGACY_RC_MAP_CONTRACT_ID, evidence),
    ))


def _read_current_stage_context(provenance, metadata, root, policy, *, record_reader=None):
    contract = policy.get("current_stage_local")
    geometry = metadata.get("current_task_geometry")
    active = provenance.get("dataset_identity")
    refs = provenance.get("source_records", ())
    if not contract or not geometry or not active or not refs:
        return None
    if metadata.get("backend_id", "smores_ep") != "smores_ep":
        return None
    if (contract.get("contract_id") != STAGE_LOCAL_CONTRACT_ID
            or contract.get("geometry_convention") != "world_z_planar_m_rad"):
        raise ValueError("unsupported current stage_local source contract")
    pin = contract["canonical_manifest"]
    if (_relative_path(active.get("manifest"), root) != pin["path"]
            or active.get("manifest_sha256") != pin["sha256"]):
        return None
    features = {f["name"]: f for f in geometry}
    if len(features) != len(geometry) or _ORIGIN not in features or _YAW not in features:
        return None
    if any(f.get("valid") is not True for f in geometry):
        return None
    origin = _physical_vector(features[_ORIGIN].get("values"), 3)
    yaw = _physical_vector(features[_YAW].get("values"), 1)[0]
    local = {name: _physical_vector(f.get("values"), len(f.get("values", ())))
             for name, f in features.items() if name.startswith("stage_local.")}
    if not local:
        return None
    proofs = {}
    canonical = _pinned_document(root, pin["path"], pin["sha256"], proofs)
    eid = provenance.get("logical_episode_id")
    if len([e for e in canonical.get("episodes", ()) if e.get("episode_id") == eid]) != 1:
        return None
    source_root = Path(contract["source_root"])
    directory = source_root / "episodes" / eid

    def small(path):
        relative = _relative_path(str(path), root)
        target = root / relative
        if not target.is_file() or target.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("missing/unsupported small current source evidence")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return _pinned_document(root, relative, digest, proofs)

    episode = small(directory / "manifest.json")
    recorded = small(directory / "source_manifest.json")
    if (episode.get("schema_version") != "mssr.curated_teleop_episode.v1"
            or recorded.get("schema_version") != "mssr.teleop_recording_manifest.v1"
            or episode.get("episode_id") != eid or recorded.get("episode_id") != eid):
        return None
    course_path = str(directory / "course.json")
    audited = [a for a in metadata.get("audit", ()) if a.get("sha256")
               and _relative_path(a.get("path"), root) == course_path]
    if len(audited) != 1:
        return None
    course = _pinned_document(root, course_path, audited[0]["sha256"], proofs)
    if course.get("frame_id") != "world" or course.get("geometry_sha256") != episode.get("geometry_sha256"):
        return None

    def matches_current(course):
        mission = course.get("mission", {})
        if course.get("frame_id") != "world" or mission.get("schema_version") != contract["mission_schema"]:
            return False
        matches = []
        for task in mission.get("tasks", ()):
            parameters = task.get("parameters", {})
            frame = parameters.get("stage_frame", {})
            # Physical equality selects the already-scoped D1/D2 frame; no task
            # ID/type, behavioral stage, Canonical index or instruction is read.
            if frame.get("origin_world_xyz_m") != list(origin) or frame.get("yaw_rad") != yaw:
                continue
            if parameters.get("geometry_frame") != "world":
                continue
            agrees = True
            for name, values in local.items():
                _, group, field = name.split(".", 2)
                source_geometry = parameters.get(group, {})
                raw = source_geometry.get(field)
                raw_values = tuple(raw) if isinstance(raw, (list, tuple)) else (raw,)
                agrees = agrees and source_geometry.get("coordinate_frame") == "stage_local" and raw_values == values
            if agrees:
                matches.append(frame)
        return len(matches) == 1

    if not matches_current(course):
        return None
    stamp = provenance.get("physical_stamp")
    if type(stamp) not in (int, float) or not math.isfinite(stamp):
        return None
    grouped = {}
    for ref in refs:
        path = _relative_path(ref.get("_loader_source_path"), root)
        declarations = [d for d in episode.get("datasets", ()) if str(source_root / d.get("path", "")) == path]
        if len(declarations) != 1:
            return None
        declaration = declarations[0]
        if _relative_path(declaration.get("source_path"), root) not in {
                _relative_path(d.get("path"), root) for d in recorded.get("structural_streams", ())}:
            return None
        index = ref.get("_loader_source_row")
        if type(index) is not int or not 1 <= index <= declaration.get("samples", 0):
            return None
        grouped.setdefault(path, set()).add(index)
    for path, indices in grouped.items():
        found = set()
        def sequential():
            with (root / path).open("rb") as stream:
                for index, line in enumerate(stream, 1):
                    if index in indices:
                        yield index, json.loads(line)
                    if index >= max(indices):
                        break
        records = sequential() if record_reader is None else record_reader(root / path, tuple(sorted(indices)))
        for index, row in records:
            if index not in indices or index in found:
                return None
            observation = row.get("observation", {})
            if (row.get("schema_version") != contract["source_schema"]
                    or observation.get("schema_version") != contract["observation_schema"]
                    or observation.get("robot_family") != "smores_ep"
                    or row.get("graph_t", {}).get("stamp") != stamp
                    or observation.get("stamp") != stamp
                    or not matches_current(observation.get("course", {}))
                    or observation["course"].get("geometry_sha256") != course.get("geometry_sha256")):
                return None
            # Digest the verified current physical fragment, not actions,
            # behavioral fields or G_t+1 contained in the source row.
            receipt_path = path + "#L" + str(index) + ":current-frame"
            physical_digest = bridge_policy_fingerprint({
                "source_schema": row["schema_version"],
                "observation_schema": observation["schema_version"],
                "robot_family": observation["robot_family"], "physical_stamp": stamp,
                "origin_world_xyz_m": origin, "yaw_rad": yaw, "stage_local_geometry": local,
                "geometry_sha256": observation["course"]["geometry_sha256"],
            })
            proofs[receipt_path] = BridgeEvidence(receipt_path, physical_digest)
            found.add(index)
        if found != indices:
            return None
    identity = {"canonical_manifest_sha256": pin["sha256"], "current_stage_contract_revision": contract["revision"],
                "source_policy_fingerprint": bridge_policy_fingerprint(policy), "source_episode": eid,
                "current_frame_fingerprint": bridge_policy_fingerprint({"origin": origin, "yaw": yaw, "local_geometry": local})}
    return FrameSourceContext("smores_ep", tuple(sorted(identity.items())), (
        SourceContractCertificate(STAGE_LOCAL_CONTRACT_ID, tuple(proofs[p] for p in sorted(proofs))),
    ))


def read_frame_source_context(provenance: Mapping, metadata: Mapping, *, repo_root: Path, policy: Mapping,
                              record_reader=None) -> FrameSourceContext:
    """Independently verify legacy MAP and current planar source receipts.

    For current frames, caller composition supplies current_task_geometry as the
    origin/yaw and stage_local NumericFeature dictionaries already selected by
    D1/D2. This audit-only descriptor contains no task ID or behavioral metadata.
    Exact current source rows and small sidecars must independently agree with it.
    Production composition injects the shared indexed exact-record reader;
    standalone callers retain the sequential fallback with identical validation.
    """
    legacy = _read_legacy_source_context(provenance, metadata, repo_root=repo_root, policy=policy)
    current = _read_current_stage_context(provenance, metadata, Path(repo_root).resolve(), policy,
        record_reader=record_reader)
    if current is None:
        return legacy
    identity = dict(current.source_identity)
    if legacy.certificates:
        identity.update(dict(legacy.source_identity))
    return FrameSourceContext("smores_ep", tuple(sorted(identity.items())), current.certificates + legacy.certificates)
