"""Opt-in BC-v5 ActionSet live path, without offline candidate exceptions.

Current physical snapshot -> actor-visible H1 features -> Stage-11 structural
catalog -> H2 -> ActionSet decoder -> recheck *fresh physical* joint snapshot.
This is a guarded action-set interface, not an installed robot transport node.
Only L1b's currently certified navigation TaskContext is supported.
"""
from __future__ import annotations

from .action_set_decoder import decode_action_sets, assert_executable
from .actor_visibility import project_actor_features, VisibilityPolicy
from .bc_v5_joint_safe_candidates import live_align_is_physically_safe
from .bc_v5_topology_candidates import enumerate_candidates_v5
from .live_observation import LiveObservation, LivePhysicalState, prepare_live_observation


def prepare_v5_live(state: LivePhysicalState, *, visibility: VisibilityPolicy,
                    assigned_goal=None, **kwargs) -> LiveObservation:
    """Actor-visible projection, identical E2 v5 topology admission for live.

    Explicitly require the reviewed visibility policy, not a hidden default.
    Any goal certificate must be based on current information and is validated
    by actor_visibility.project_actor_features; no future task phase is used.
    """
    if not isinstance(visibility, VisibilityPolicy):
        raise TypeError('Explicit frozen actor visibility policy required')
    if 'candidate_provider' in kwargs or 'feature_projector' in kwargs:
        raise ValueError('Candidate generator and actor projection are frozen in BC-v5')
    schema = kwargs['feature_schema']
    normalizers = kwargs['normalizers']
    if normalizers.metadata.get('actor_visibility_fingerprint') != visibility.fingerprint:
        raise ValueError('BC-v5 live normalizer/visibility mismatch')
    return prepare_live_observation(state,
        candidate_provider=enumerate_candidates_v5,
        feature_projector=lambda features, inputs: project_actor_features(
            features, inputs.context.task_id, visibility, schema,
            assigned_goal=assigned_goal),
        **kwargs)


def gate_v5_bundle(bundle, *, decision_state: LivePhysicalState,
                   fresh_state: LivePhysicalState, physical_snapshot_confirmed: bool,
                   spec, schema, claim_resolver, availability=None):
    """Fail CLOSED for the whole decoded bundle; never substitute alternatives.

    The external physical snapshot provider must certify REAL current FixedJoint
    state (not desired executor faces). A fresh or same-stamp atomic snapshot is
    required. This guard NEVER accepts Stage-10 exception metadata.
    """
    if type(physical_snapshot_confirmed) is not bool or not physical_snapshot_confirmed:
        raise ValueError('Fresh physical-joint snapshot is mandatory before dispatch')
    if not isinstance(fresh_state, LivePhysicalState) or not isinstance(decision_state, LivePhysicalState):
        raise TypeError('Expected typed physical snapshots')
    if fresh_state.stamp < decision_state.stamp:
        raise ValueError('Physical snapshot regressed behind H2 decision')
    if not callable(claim_resolver):
        raise TypeError('Physical claim resolver required')
    current = enumerate_candidates_v5(fresh_state.graph, spec, schema)
    structural = {(a.action_id, b.initiator, b.connectors)
                  for c in current.structural_candidates
                  for a,b in ((c.action, binding) for binding in c.bindings)}
    resources = {(c.action.action_id, c.resource_ref) for c in current.resource_candidates}
    for action in bundle.actions:
        if action.resource is not None:
            if (action.action_id, action.resource) not in resources:
                raise ValueError('Current physical state rejects resource candidate')
        else:
            key = action.action_id, action.initiator, action.connectors
            if key not in structural:
                raise ValueError('Current physical topology rejects structural candidate')
            if action.action_id == 'align_faces' and not live_align_is_physically_safe(
                    action.initiator, fresh_state.graph, physical_snapshot_confirmed=True):
                raise ValueError('Align initiator has physical FixedJoint or unknown connection')
    return assert_executable(bundle, claim_resolver, availability)


def decode_guarded_v5(prediction, observation: LiveObservation, normalizers,
                      *, fresh_state: LivePhysicalState, physical_snapshot_confirmed: bool,
                      spec, schema, claim_resolver, availability=None, threshold=0.):
    """Only guarded v5 decoding API. No historical labels, no Phase 10 override."""
    if not isinstance(observation, LiveObservation):
        raise TypeError('Expected prepared BC-v5 LiveObservation')
    if observation.inputs.candidates != enumerate_candidates_v5(observation.state.graph, spec, schema):
        raise ValueError('Observation does not use the BC-v5 candidate catalog')
    bundles = decode_action_sets(prediction, observation.batch, normalizers, threshold=threshold)
    if len(bundles) != 1:
        raise ValueError('Live ActionSet inference requires one graph')
    return gate_v5_bundle(bundles[0], decision_state=observation.state,
        fresh_state=fresh_state, physical_snapshot_confirmed=physical_snapshot_confirmed,
        spec=spec, schema=schema, claim_resolver=claim_resolver, availability=availability)


def infer_guarded_v5(model, state: LivePhysicalState, *,
                     fresh_physical_state_provider, visibility: VisibilityPolicy,
                     normalizers, spec, schema, claim_resolver,
                     context, observation, bridges, normalization_policy, tensor_layout,
                     assigned_goal=None, availability=None, threshold=0.):
    """One explicit BC-v5 ActionSet inference cycle, end-to-end up to transport.

    Caller supplies a trusted *new physical-joint snapshot provider* read after
    the model forward. That provider returns ``(LivePhysicalState, bool)`` where
    bool attests the snapshot is based on real current FixedJoint state.
    Output is a guarded ActionBundle; the robot transport/claim owner still
    performs its own final availability and execution checks.
    """
    if not callable(fresh_physical_state_provider):
        raise TypeError('Fresh physical-joint provider required')
    prepared = prepare_v5_live(state, visibility=visibility, assigned_goal=assigned_goal,
        context=context, observation=observation, bridges=bridges,
        spec=spec, feature_schema=schema, normalization_policy=normalization_policy,
        normalizers=normalizers, tensor_layout=tensor_layout)
    import torch
    with torch.no_grad():
        prediction = model(prepared.batch['inputs'])
    fresh, confirmed = fresh_physical_state_provider()
    return decode_guarded_v5(prediction, prepared, normalizers,
        fresh_state=fresh, physical_snapshot_confirmed=confirmed,
        spec=spec, schema=schema, claim_resolver=claim_resolver,
        availability=availability, threshold=threshold)
