"""Observational C1 assembly/probe evaluator and bounded episode reward ledger.

Targets are evaluator-only. No action generation or mask mutation.
"""
from dataclasses import asdict,dataclass,replace
from collections import defaultdict
import math

from drl.MARL.c1_config import C1Settings,physical_bounds,inherited_event_weights
from drl.MARL.c1_structure import assess_structure,canonical_signature,norm
from drl.MARL.c1_probe import ProbeGeometry,probe_metrics
from drl.MARL.phase1_reward import ActionOutcomeSignals,_validate_signals


@dataclass(frozen=True)
class C1Evaluation:
    phase: str
    assembly_valid_now: bool
    success: bool
    terminated: bool
    truncated: bool
    reason: str | None
    assessment: object
    probe: object | None
    reward_breakdown: dict
    owner_rewards: dict
    team_reward: float
    target_graph_complete: bool | None = None
    assembly_stable: bool = False
    fold_complete: bool | None = None
    functional_probe_success: bool = False
    episode_success: bool = False
    target_assignment: dict | None = None
    stagnation: dict | None = None
    redundancy: dict | None = None


class AssemblyProbeEvaluator:
    def __init__(self,task,course,owner_ids,*,settings=None,max_steps=1800,event_config_path=None,target_graph=None,discount_factor=.99):
        self.task,self.course,self.owner_ids=task,course,tuple(owner_ids)
        if len(self.owner_ids)!=8 or len(set(self.owner_ids))!=8:raise ValueError('C1 requires eight unique physical owners')
        self.settings=settings or C1Settings.load()
        self.episode_mode=self.settings.values.get('episode_mode','legacy')
        self.discount_factor=float(discount_factor)
        if not math.isfinite(self.discount_factor) or not 0<=self.discount_factor<=1:
            raise ValueError('Invalid physical-step discount')
        if self.episode_mode=='discovery':target_graph=None
        if self.episode_mode=='guided' and target_graph is None:
            raise ValueError('Guided C1 requires an external evaluator target')
        self.guided_gate=None
        if self.episode_mode=='guided':
            from drl.MARL.c1_guided_success import GuidedAssemblyGate
            self.guided_gate=GuidedAssemblyGate(target_graph,self.owner_ids)
        from drl.MARL.c1_stagnation import EpisodeStagnationMonitor
        from drl.MARL.c1_redundancy import RedundancyMonitor
        self.stagnation_monitor=EpisodeStagnationMonitor(self.settings)
        self.redundancy_monitor=RedundancyMonitor(self.owner_ids,self.settings)
        self._last_monitor_score=0.
        self.bounds=physical_bounds(course,self.settings,self.owner_ids)
        self.geometry=ProbeGeometry(task,course,self.settings)
        self.weights={**self.settings.weights,**inherited_event_weights(task,event_config_path)}
        self.max_steps=max_steps
        self.phase='ASSEMBLY_NOT_VALID';self.last_frame=None;self.last_assessment=None
        self.stable_since=None;self.probe_anchor=None;self.last_probe=None
        self.assembly_valid_timestamp=None;self.probe_start_timestamp=None;self.probe_success_timestamp=None
        self.probe_attempts=[];self.best_lcc=0.;self.best_merges=0.;self.best_probe=0.;self.stability_paid=False
        self.seen_terminal_goals=set();self.seen_joint_ids=set();self.removed_joint_ids=set()
        self.counts=defaultdict(int);self.breakdown=defaultdict(float);self.return_=0.;self.steps=0
        self.terminal=False;self.last_result=None
        from drl.MARL.c1_guided_feedback import (
            AlignOutcomeFeedback, TargetGraphProgress,
        )
        self.align_feedback=AlignOutcomeFeedback()
        self.target_progress=(
            TargetGraphProgress(target_graph,self.owner_ids)
            if target_graph is not None else None
        )
        self.last_target_fraction=0.0

    def prime(self,frame):
        """Baseline a measured initial snapshot before the first physical step.

        Disconnected components are allowed; missing evidence, invalid joints,
        collisions and uncontrolled dynamics are not. This records no reward,
        action opportunity, elapsed stability, probe attempt or success.
        """
        if self.steps or self.last_frame is not None or self.terminal:
            raise RuntimeError('C1 evaluator must be primed before its first step')
        assessment=assess_structure(frame,self.bounds)
        if not assessment.safe or assessment.reason not in ('valid','disconnected_assembly'):
            raise ValueError(f'Invalid C1 initial snapshot: {assessment.reason}')
        gate=self.guided_gate.assess(frame) if self.guided_gate is not None else None
        if self.target_progress is not None:
            self.target_progress.prime(frame,
                incompatible_weight=self.settings.values.get('target_incompatibility_weight',1.))
            self.last_target_fraction=self.target_progress._last_match/len(self.target_progress.target_edges)
        self.best_lcc=(max(assessment.component_sizes,default=1)-1)/7
        self.best_merges=(8-len(assessment.component_sizes))/7
        monitor_score=self.best_lcc+self.best_merges
        if self.target_progress is not None:
            monitor_score=self.target_progress.current_potential
            if gate is not None and gate.fold_complete:monitor_score+=1.
        self._last_monitor_score=monitor_score
        self.stagnation_monitor.best_structural=monitor_score
        self.stagnation_monitor.best_functional=0.
        self.seen_joint_ids.update(c.joint_id for c in frame.connections if c.joint_valid)
        self.last_frame=frame;self.last_assessment=assessment

    def _reset_probe(self,time_s,reason):
        if self.probe_anchor is not None and self.probe_attempts:
            self.probe_attempts[-1].update(end_s=time_s,reason=reason)
        self.probe_anchor=None;self.stable_since=None

    def step(self,frame,*,signals=None,action_counts=None,action_goals=None,
             active_goals=None,decision_opportunity=False,continuous_commands=None,
             prepared_relations=()):
        if self.terminal:raise RuntimeError('C1 episode already finished')
        signals=signals or ActionOutcomeSignals();_validate_signals(signals,self.owner_ids)
        frame=replace(frame,pathological_collision_owner_ids=tuple(sorted(
            set(frame.pathological_collision_owner_ids)|set(signals.collision_owner_ids))))
        self.steps+=1
        gap=False
        if self.last_frame is not None:
            dt=frame.time_s-self.last_frame.time_s
            if not math.isfinite(dt) or dt<=0:raise ValueError('C1 observations must have strictly increasing simulation timestamps')
            gap=dt>1.5/30.
            old={m.module_id:m for m in self.last_frame.modules}
            if len(old)==8 and all(m.module_id in old for m in frame.modules):
                finite_difference=max(norm(tuple(m.position[k]-old[m.module_id].position[k] for k in range(3)))/dt for m in frame.modules)
                for module in frame.modules:
                    previous_ports=dict(old[module.module_id].connectors)
                    for port,position in module.connectors:
                        if port in previous_ports:
                            finite_difference=max(finite_difference,norm(tuple(position[k]-previous_ports[port][k] for k in range(3)))/dt)
                frame=replace(frame,peak_linear_speed=max(frame.peak_linear_speed,finite_difference))
        assessment=assess_structure(frame,self.bounds)
        gate=self.guided_gate.assess(frame) if self.guided_gate is not None else None
        eligible=assessment.instantaneous_valid and (gate is None or gate.target_graph_complete and gate.fold_complete)
        target_complete=gate.target_graph_complete if gate is not None else None
        fold_complete=gate.fold_complete if gate is not None else None
        potential_mode=self.settings.values.get('target_graph_shaping_mode')=='potential'
        raw={key:0. for key in self.weights}
        raw['penalty_time']=1.
        align_failed,align_retry,align_stagnant=(
            self.align_feedback.step(frame,action_goals or {})
        )
        for name,counts in (
            ('penalty_failed_alignment',align_failed),
            ('penalty_unproductive_retry',align_retry),
            ('penalty_alignment_stagnation',align_stagnant),
        ):
            if name in raw:
                raw[name]=sum(counts.values())/len(self.owner_ids)
        failed_owners=set()
        for event in frame.primitive_events:
            key=(event.goal_id,event.state)
            if event.state not in ('failed','succeeded','rejected','canceled') or key in self.seen_terminal_goals:continue
            self.seen_terminal_goals.add(key)
            if event.action=='dock' and event.state=='failed' and event.owner in self.owner_ids:failed_owners.add(event.owner)
        raw['penalty_failed_docking']=len(failed_owners)/8
        raw['penalty_conflict']=len(signals.conflict_owner_ids)/8
        raw['penalty_invalid']=len(signals.invalid_owner_ids)/8
        collision=set(signals.collision_owner_ids)|set(frame.pathological_collision_owner_ids)
        raw['penalty_collision']=len(collision)/8
        previous_signature=self.last_assessment.connection_signature if self.last_assessment else ()
        previous_joint_ids={x[0] for x in previous_signature}
        current_joint_ids={c.joint_id for c in frame.connections if c.joint_valid}
        added=current_joint_ids-self.seen_joint_ids
        event_adds={jid for action,jid in frame.connection_events if action=='added'}
        event_removes={jid for action,jid in frame.connection_events if action=='removed'}
        self.counts['docking_count']+=len((current_joint_ids|event_adds)-self.seen_joint_ids)
        removed=(previous_joint_ids-current_joint_ids)|event_removes
        self.counts['undocking_count']+=len(removed-self.removed_joint_ids)
        self.seen_joint_ids.update(current_joint_ids|event_adds);self.removed_joint_ids.update(removed)
        was_valid=self.phase=='FUNCTIONAL_PROBE'
        changed=bool(frame.topology_changed or self.last_assessment and assessment.connection_signature!=previous_signature)
        pathological=assessment.reason in ('excessive_dynamics','transient_physical_invalidity','invalid_docking_constraint_or_connector_ownership','pathological_collision')
        raw['penalty_instability']=float(bool(removed) or (was_valid and (changed or not assessment.instantaneous_valid))
            or pathological and (self.last_assessment is None or self.last_assessment.reason!=assessment.reason))
        reason=None;success=False;measurement=None
        if not assessment.safe:
            self._reset_probe(frame.time_s,assessment.reason)
            self.phase='FAILURE';reason=assessment.reason;raw['penalty_failure']=1.
        else:
            # Positive structural budgets require physically sound constraints
            # and controlled dynamics, but can reward partial components.
            sound=not pathological and not collision
            if sound:
                lcc=(max(assessment.component_sizes,default=1)-1)/7
                merges=(8-len(assessment.component_sizes))/7
                raw['reward_structural_progress']=max(0.,lcc-self.best_lcc)
                raw['reward_valid_docking']=max(0.,merges-self.best_merges) if added else 0.
                self.best_lcc=max(self.best_lcc,lcc);self.best_merges=max(self.best_merges,merges)
                if self.target_progress is not None and not potential_mode:
                    gain,self.last_target_fraction=self.target_progress.observe(frame)
                    raw['reward_target_graph_progress']=gain
            if changed or gap or not eligible:
                self._reset_probe(frame.time_s,'observation_gap' if gap else 'structural_or_dynamic_change')
            if eligible:
                if self.stable_since is None:self.stable_since=frame.time_s
                if frame.time_s-self.stable_since+1e-10>=self.settings.stability_window_s:
                    self.phase='FUNCTIONAL_PROBE'
                    if self.assembly_valid_timestamp is None:self.assembly_valid_timestamp=frame.time_s
                    if not self.stability_paid:
                        raw['reward_structural_stability']=1.;self.stability_paid=True
                    if self.probe_anchor is None:
                        self.probe_anchor=frame
                        if self.probe_start_timestamp is None:self.probe_start_timestamp=frame.time_s
                        self.probe_attempts.append(dict(start_s=frame.time_s))
                    measurement=probe_metrics(self.geometry,self.probe_anchor,frame)
                    self.last_probe=measurement
                    if not measurement.safe:
                        raw['penalty_instability']=1.
                        self._reset_probe(frame.time_s,'probe_physical_guard');self.phase='ASSEMBLY_VALIDATING'
                    else:
                        raw['reward_task_probe_progress']=max(0.,measurement.progress_fraction-self.best_probe)
                        self.best_probe=max(self.best_probe,measurement.progress_fraction)
                        elapsed=frame.time_s-self.probe_anchor.time_s
                        if elapsed+1e-10>=self.settings.probe_min_duration_s and measurement.passed_geometry:
                            success=True;self.probe_success_timestamp=frame.time_s
                            raw['reward_probe_success']=1.;self.phase='SUCCESS'
                            self.probe_attempts[-1].update(end_s=frame.time_s,reason='success')
                        elif elapsed>=self.settings.probe_max_duration_s:
                            self._reset_probe(frame.time_s,'probe_timeout_restart');self.phase='ASSEMBLY_VALIDATING'
                else:self.phase='ASSEMBLY_VALIDATING'
            else:self.phase='ASSEMBLY_NOT_VALID'
        # Report independent physical gates even when a later functional
        # stagnation failure changes the episode phase and clears the probe.
        assembly_stable_now=self.phase in ('FUNCTIONAL_PROBE','SUCCESS')
        monitor_score=self.best_lcc+self.best_merges
        if self.target_progress is not None and assessment.safe:
            monitor_score=self.target_progress.potential(frame,
                incompatible_weight=self.settings.values.get('target_incompatibility_weight',1.))
            self.last_target_fraction=self.target_progress._last_match/len(self.target_progress.target_edges)
            if fold_complete:monitor_score+=1.
        stagnant=None
        if assessment.safe:
            stagnant=self.stagnation_monitor.step(frame,structural_score=monitor_score,
                functional_progress=self.best_probe,decision_opportunity=decision_opportunity,
                active_goals=active_goals,action_goals=action_goals,
                prepared_relations=prepared_relations)
        stagnation_terminal=not success and stagnant is not None and stagnant.terminate
        if stagnation_terminal:
            self._reset_probe(frame.time_s,'stagnation')
            self.phase='FAILURE';reason='stagnation'
            if 'penalty_stagnation' in raw:raw['penalty_stagnation']=1.
        terminated=not assessment.safe or success or stagnation_terminal
        truncated=not terminated and self.steps>=self.max_steps
        if truncated:reason='time_limit'
        if potential_mode and self.target_progress is not None:
            raw.update(self.target_progress.shaping(frame,discount_factor=self.discount_factor,
                incompatible_weight=self.settings.values.get('target_incompatibility_weight',1.),terminal=terminated))
        useful=(monitor_score>self._last_monitor_score+1e-6 or raw['reward_task_probe_progress']>0)
        self._last_monitor_score=monitor_score
        redundant,redundancy=({},None)
        if assessment.safe:
            redundant,redundancy=self.redundancy_monitor.step(frame,action_goals or {},useful_progress=useful,
                continuous_commands=continuous_commands,
                already_charged_keys=[e.goal_id for e in frame.primitive_events
                    if e.action in ('align_faces','dock') or e.owner in signals.conflict_owner_ids or e.owner in signals.invalid_owner_ids])
        if 'penalty_redundant_action' in raw:
            raw['penalty_redundant_action']=sum(redundant.values())/len(self.owner_ids)
        owner_raw={owner:dict(raw) for owner in self.owner_ids}
        for owner,terms in owner_raw.items():
            terms['penalty_conflict']=float(owner in signals.conflict_owner_ids)
            terms['penalty_invalid']=float(owner in signals.invalid_owner_ids)
            terms['penalty_collision']=float(owner in collision)
            terms['penalty_failed_docking']=float(owner in failed_owners)
            if 'penalty_redundant_action' in terms:
                terms['penalty_redundant_action']=float(redundant.get(owner,0))
            for name,counts in (
                ('penalty_failed_alignment',align_failed),
                ('penalty_unproductive_retry',align_retry),
                ('penalty_alignment_stagnation',align_stagnant),
            ):
                if name in terms:
                    terms[name]=float(counts[owner])
        weighted={key:raw[key]*weight for key,weight in self.weights.items()}
        owner_rewards={owner:sum(terms[key]*self.weights[key] for key in terms) for owner,terms in owner_raw.items()}
        reward=sum(weighted.values())
        for key,value in weighted.items():self.breakdown[key]+=value
        for key,value in (action_counts or {}).items():self.counts[key]+=int(value)
        self.return_+=reward;self.terminal=bool(terminated or truncated)
        self.last_frame=frame;self.last_assessment=assessment
        result=C1Evaluation(self.phase,self.phase in ('FUNCTIONAL_PROBE','SUCCESS'),success,terminated,truncated,reason,
            assessment,measurement,weighted,owner_rewards,reward,target_graph_complete=target_complete,
            assembly_stable=assembly_stable_now,fold_complete=fold_complete,
            functional_probe_success=success,episode_success=success,
            target_assignment=gate.assignment if gate is not None else None,
            stagnation=asdict(stagnant) if stagnant is not None else None,redundancy=redundancy)
        self.last_result=result
        return result

    def episode_record(self,*,reason=None):
        frame=self.last_frame
        if frame is None or self.last_result is None:return {}
        result=self.last_result
        return dict(task=self.task,seed=self.course.seed,assembly_valid_timestamp=self.assembly_valid_timestamp,
            probe_start_timestamp=self.probe_start_timestamp,probe_success_timestamp=self.probe_success_timestamp,
            assembly_validity_achieved=self.assembly_valid_timestamp is not None,functional_probe_achieved=self.probe_success_timestamp is not None,C1_success=result.success,
            probe_attempts=self.probe_attempts,phase=self.phase,final_graph=[asdict(c) for c in frame.connections],
            canonical_topology_signature=canonical_signature(frame),unlabelled_topology_signature=canonical_signature(frame,include_ports=False),
            valid_connection_count=self.last_assessment.valid_connections,component_sizes=list(self.last_assessment.component_sizes),
            posture_summary={m.module_id:dict(position=m.position,orientation=m.orientation,joints=m.joints) for m in frame.modules},
            structural_action_count=self.counts['structural_action_count'],projected_structural_action_count=self.counts['projected_structural_action_count'],
            dropped_conflict_count=self.counts['dropped_conflict_count'],dropped_invalid_count=self.counts['dropped_invalid_count'],
            docking_count=self.counts['docking_count'],undocking_count=self.counts['undocking_count'],
            functional_probe_metric=asdict(self.last_probe) if self.last_probe else None,functional_probe_threshold=self.geometry.thresholds,
            reward_breakdown=dict(self.breakdown),final_return=self.return_,success=result.success,
            assembly_valid_now=result.assembly_valid_now,instantaneous_structural_valid=result.assessment.instantaneous_valid,physical_safety=result.assessment.safe,
            physical_bounds=asdict(self.bounds),
            terminated=result.terminated,truncated=result.truncated,success_failure_reason=reason or ('C1_success' if result.success else result.reason or 'observation_end'),
            settings_path=self.settings.path,settings_sha256=self.settings.sha256,steps=self.steps,
            target_graph_best_fraction=(
                self.target_progress.best_fraction
                if self.target_progress else None
            ),
            align_feedback_counts=dict(self.align_feedback.counts),
            episode_mode=self.episode_mode,target_graph_complete=result.target_graph_complete,
            assembly_stable=result.assembly_stable,fold_complete=result.fold_complete,
            functional_probe_success=result.functional_probe_success,episode_success=result.episode_success,
            target_assignment=result.target_assignment,stagnation=result.stagnation,redundancy=result.redundancy,
            collision_measurement='module-reference-point solid-interior penetration plus explicitly measured runtime events; no force sensor',
            energy_measurement='not instrumented; no mechanical-work reward')
