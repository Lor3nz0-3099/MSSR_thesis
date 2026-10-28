"""Certified current-state observation bridge for the BC-v4 shared actor."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1 import live_observation as live
from drl.IL.bc_v1 import normalization as f2
from drl.IL.bc_v1.actor_prepare import tensorize_actor_sample
from drl.IL.bc_v1.actor_visibility import load_visibility_policy
from drl.IL.bc_v1.batching import collate_policy_samples
from drl.IL.bc_v1.candidates import enumerate_candidates
from drl.IL.bc_v1.dataset import CachedInputs, CachedSample
from drl.IL.bc_v1.feature_schema import load_feature_schema
from drl.IL.bc_v1.smores_adapter import SMORESAdapter
from drl.IL.bc_v1.tensorization import load_tensor_layout, tensorize_policy_sample


@dataclass(frozen=True)
class BCV4RuntimeObservation:
    """One complete BC-v4 actor input derived from current physical state."""

    physical_state: Any

    # Collated tensor input consumed directly by H1/H2.
    actor_graph: Mapping[str, Any]

    # Single-sample routing used to bind H2 candidates to physical modules.
    routing_snapshot: dict[str, Any]

    # Full collated batch required later by the ActionSet decoder.
    decoder_batch: Mapping[str, Any]

    # Centralized critic input from the same current physical sample, before
    # actor visibility projection. Never passed to H1/H2 by this bridge.
    privileged_graph: Mapping[str, Any] | None = None


class BCV4RuntimeObservationBridge:
    """Atomic SMORES current snapshot -> certified BC-v4 actor input."""

    def __init__(
        self,
        *,
        spec,
        feature_schema,
        normalization_policy,
        normalizers,
        tensor_layout,
        visibility,
        adapter,
        graph_builder,
        bridges,
    ) -> None:
        self.spec = spec
        self.feature_schema = feature_schema
        self.normalization_policy = normalization_policy
        self.normalizers = normalizers
        self.tensor_layout = tensor_layout
        self.visibility = visibility
        self.adapter = adapter
        self.graph_builder = graph_builder
        self.bridges = bridges

    @classmethod
    def from_repository(
        cls,
        *,
        repo_root: str | Path,
        graph_builder,
        bridges,
    ) -> "BCV4RuntimeObservationBridge":
        """Build the exact frozen BC-v4 actor-visible preprocessing stack."""

        root = Path(repo_root).resolve()
        cfg = root / "configs/bc_v1"

        spec = c.load_robot_spec(
            cfg / "smores_ep.robot.json"
        )

        feature_schema = load_feature_schema(
            cfg / "features.json"
        )

        normalization_policy = f2.load_normalization_policy(
            cfg / "normalization.json",
            feature_schema,
        )

        tensor_layout = load_tensor_layout(
            cfg / "tensor_layout.v1.json"
        )

        visibility = load_visibility_policy(
            cfg / "actor_visibility_r1m.json"
        )

        normalizer_path = (
            root
            / "artifacts/bc_v1/normalization/"
            "f2_actor_visible_r1m_v1/"
            "normalizers.json"
        )

        actual_sha = hashlib.sha256(
            normalizer_path.read_bytes()
        ).hexdigest()

        fit_context_path = (
            normalizer_path.parent
            / "fit_context.json"
        )

        fit_context = json.loads(
            fit_context_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            fit_context.get("normalizer_sha256")
            != actual_sha
        ):
            raise ValueError(
                "BC-v4 actor-visible normalizer does not "
                "match its frozen fit_context"
            )

        if (
            fit_context.get(
                "actor_visibility_fingerprint"
            )
            != visibility.fingerprint
        ):
            raise ValueError(
                "BC-v4 actor-visible normalizer / visibility "
                "identity mismatch"
            )

        if (
            fit_context.get("TRAIN_only") is not True
            or fit_context.get("TEST_payload_access") != 0
        ):
            raise ValueError(
                "BC-v4 actor-visible normalizer provenance "
                "is not TRAIN-only"
            )

        normalizer_document = json.loads(
            normalizer_path.read_text(
                encoding="utf-8"
            )
        )

        normalizers = f2.read_normalizers(
            normalizer_path,
            expected_metadata=normalizer_document["metadata"],
        )

        # tensorize_actor_sample performs the final explicit
        # normalizer <-> visibility fingerprint compatibility check.
        return cls(
            spec=spec,
            feature_schema=feature_schema,
            normalization_policy=normalization_policy,
            normalizers=normalizers,
            tensor_layout=tensor_layout,
            visibility=visibility,
            adapter=SMORESAdapter(spec),
            graph_builder=graph_builder,
            bridges=bridges,
        )

    def prepare_snapshot(
        self,
        raw_snapshot,
        *,
        task_context,
        observation=None,
        assigned_goal=None,
    ) -> BCV4RuntimeObservation:
        """Derive one BC-v4 sample from one atomic current-state snapshot."""

        if observation is None:
            observation = c.ObservationBundle()

        # Same physical snapshot decoder already validated by L1b.
        physical_state = live.encode_smores_snapshot(
            raw_snapshot,
            adapter=self.adapter,
            graph_builder=self.graph_builder,
        )

        # Fail closed: no legacy zero-fill for missing current measurements.
        live._require_current_measurements(
            physical_state.graph,
            self.spec,
        )

        if not isinstance(task_context, c.TaskContext):
            raise TypeError(
                "BC-v4 runtime observation requires TaskContext"
            )

        if not isinstance(observation, c.ObservationBundle):
            raise TypeError(
                "BC-v4 runtime observation requires ObservationBundle"
            )

        # Strict contract first: unknown/non-certified feature names fail.
        strict = c.PolicyInputs(
            physical_state.graph,
            observation,
            task_context,
        )

        inputs = live.select_policy_inputs(
            strict.graph,
            strict.observation,
            strict.context,
        )

        geometry = live.transform_inputs(
            inputs,
            live.frame_config_for(
                self.bridges
            ),
        )

        sources = (
            *inputs.context.static_geometry,
            *inputs.observation.features,
        )

        for source, derived in zip(
            sources,
            geometry.quantities,
        ):
            if source.valid and not derived.valid:
                raise ValueError(
                    "Uncertified or missing physical geometry frame"
                )

        features = live.extract_features(
            inputs,
            geometry,
            self.spec,
            self.feature_schema,
        )

        live.validate_physical_pipeline(
            inputs,
            geometry,
            features,
            self.bridges,
            self.feature_schema,
        )

        candidates = enumerate_candidates(
            inputs.graph,
            self.spec,
            self.feature_schema,
        )

        current = CachedInputs(
            inputs,
            features,
            candidates,
        )

        # Runtime inference has no teacher supervision.
        sample = CachedSample(
            current,
            c.SupervisionBundle(),
            c.SampleAudit(
                episode_id="marl-live-current",
                transition_index=0,
                physical_stamp=physical_state.stamp,
                provenance="marl-runtime",
            ),
        )

        # Keep the centralized critic's true current environment measurements.
        # Ordinary G2 tensorization reuses the frozen F2 catalog and units. Do
        # not silently take its default scaler path for absent coverage.
        scalers = dict(self.normalizers.scalers)
        for _, row, index, key, _ in f2._feature_components(
            features, self.spec, self.feature_schema, self.normalization_policy
        ):
            if key is not None and row.validity[index] and key not in scalers:
                raise ValueError(f'Missing privileged normalizer coverage: {key}')
        privileged_sample = tensorize_policy_sample(
            sample,
            spec=self.spec,
            feature_schema=self.feature_schema,
            normalization_policy=self.normalization_policy,
            normalizers=self.normalizers,
            tensor_layout=self.tensor_layout,
        )
        privileged_batch = collate_policy_samples([privileged_sample])

        # IMPORTANT:
        # actor visibility projection happens HERE, before F2 tensorization.
        tensor_sample = tensorize_actor_sample(
            sample,
            spec=self.spec,
            feature_schema=self.feature_schema,
            normalization_policy=self.normalization_policy,
            normalizers=self.normalizers,
            tensor_layout=self.tensor_layout,
            visibility=self.visibility,
            assigned_goal=assigned_goal,
        )

        # H1/H2 consume the collated tensor graph.
        decoder_batch = collate_policy_samples(
            [tensor_sample]
        )

        return BCV4RuntimeObservation(
            physical_state=physical_state,
            actor_graph=decoder_batch["inputs"],
            routing_snapshot=tensor_sample["routing"],
            decoder_batch=decoder_batch,
            privileged_graph=privileged_batch['inputs'],
        )
