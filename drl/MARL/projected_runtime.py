"""Sample-preserving native MARL projection, separate from legacy decoding."""
from drl.MARL.action_projection import ProjectionCandidate, project_executable_action_set


def project_marl_action(*, binding, action, prediction_template, routing, batch,
                        normalizers, configured_arguments=None, availability=None,
                        feasibility=None, bulk_host_reads=True):
    """Decode each sampled member, retaining raw selection/arguments for PPO.

    Individual decoder/adapter failures become drops; contract/infrastructure
    failures still raise. No actor rerun, replacement action or resampling.
    """
    import torch
    from drl.MARL.policy import bind_marl_candidates
    from drl.MARL.smores_runtime_binding import materialize_marl_action_set_prediction,DecodedSMORESAction
    from drl.IL.bc_v1.live_policy import decode_runtime_action
    from drl.IL.bc_v1.action_decoder import _finite_tree
    # Model/schema errors are infrastructure failures, never policy penalties.
    _finite_tree(prediction_template)
    records = bind_marl_candidates(routing=routing,presence_logits=prediction_template['presence_logits'])
    prediction = materialize_marl_action_set_prediction(action=action,
        prediction_template=prediction_template,routing=routing,batch=batch)
    binding._action_set_decoder({**prediction,'presence_logits':torch.full_like(
        prediction['presence_logits'],-1.)},batch,normalizers,
        configured_arguments=configured_arguments,threshold=0.)
    candidates = []
    infrastructure_errors = ('Frozen F2 mismatch','Missing F2 physical binding','Not an ActionSet prediction',
        'ActionSet candidate routing mismatch','ActionSet routing length mismatch','Nonfinite H2 prediction')
    selected_presence = action.presence.detach().reshape(-1).cpu().tolist() if bulk_host_reads else None
    for record in records:
        index = record.candidate_index
        if not (selected_presence[index] if bulk_host_reads else bool(action.presence[index])):
            continue
        choice,claims,decoded,error,command = None,(),None,None,None
        selected = torch.full_like(prediction['presence_logits'],-1.)
        selected[index] = 1.
        try:
            member, = binding._action_set_decoder({**prediction,'presence_logits':selected},batch,
                normalizers,configured_arguments=configured_arguments,threshold=0.)
            choice, = member.actions
            claims = binding._claim_resolver(choice)
            # ActionSet scope is unordered; connector order already records
            # endpoint roles. Reuse the existing BC runtime serialization
            # adapter, retaining the original sampled ActionChoice unchanged.
            command = decode_runtime_action(binding._adapter,choice)
        except ValueError as exc:
            if any(token in str(exc) for token in infrastructure_errors):
                raise
            error = str(exc)
        if error is None:
            # A malformed adapter output is a broken integration contract.
            decoded = DecodedSMORESAction(choice,binding._validate_decoded_command(command))
        candidates.append(ProjectionCandidate(record.candidate_id,index,float(record.presence_logit),
            (record.owner_agent_id,),choice,claims,decoded,error))
    return project_executable_action_set(candidates,availability=availability,feasibility=feasibility,
        raw_sampled_action=action)


class ProjectedNativeProposalBinding:
    def __init__(self,binding,env,*,bulk_host_reads=True):
        self.binding,self.env = binding,env
        self.bulk_host_reads = bulk_host_reads
        self.last_projection_by_env = {}

    def decode_marl_action(self,**kwargs):
        dispatcher = getattr(self.env,'native_dispatcher',None)
        checker = getattr(dispatcher,'check_projected_candidate',None)
        projection = project_marl_action(binding=self.binding,**kwargs,bulk_host_reads=self.bulk_host_reads,
            feasibility=(lambda c:checker(self._env_index,c)) if checker is not None else None)
        self.last_projection_by_env[self._env_index] = projection
        return tuple(c.decoded for c in projection.accepted_candidates)


def projection_outcome(projection, *, executed=None, physical_drops=(), statuses=()):
    from drl.MARL.smores_runtime_binding import SMORESMARLProposalOutcome
    if executed is None:
        executed = projection.accepted_candidate_ids
    all_drops = (*projection.dropped,*physical_drops)
    conflict = tuple(sorted({o for d in all_drops if d.code=='RESOURCE_BUSY' for o in d.owner_ids}))
    invalid = tuple(sorted({o for d in all_drops if d.code=='INVALID_GOAL' for o in d.owner_ids}))
    owners = tuple(sorted({o for c in projection.sampled_candidates if c.candidate_id in executed for o in c.owner_ids}))
    return SMORESMARLProposalOutcome(True,'ACCEPTED' if executed else 'EMPTY','projected policy proposal',
        owners,tuple(statuses),conflict_owner_ids=conflict,invalid_owner_ids=invalid,
        projection=projection,physical_drops=tuple(physical_drops),executed_candidate_ids=tuple(executed))
