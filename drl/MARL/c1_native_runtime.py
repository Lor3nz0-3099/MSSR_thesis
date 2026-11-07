"""Loose-mode C1 evaluation/logging, independent of policy and reset control."""
from dataclasses import asdict
from collections import deque
import json
import math
from pathlib import Path

from drl.MARL.c1_config import C1Settings
from drl.MARL.c1_episode import AssemblyProbeEvaluator
from drl.MARL.runtime_signals import collect_runtime_signals


def sanitized_record(value):
    missing=[]
    def clean(item,path=''):
        if isinstance(item,float) and not math.isfinite(item):missing.append(path);return None
        if isinstance(item,dict):return {str(k):clean(v,f'{path}/{k}') for k,v in item.items()}
        if isinstance(item,(tuple,list)):return [clean(v,f'{path}/{i}') for i,v in enumerate(item)]
        return item
    result=clean(value)
    if missing:result['sanitized_nonfinite_fields']=missing
    return result


class NativeC1RewardRuntime:
    def __init__(self,*,settings_path=None,evidence_directory=None):
        self.settings=C1Settings.load(settings_path)
        self.directory=Path(evidence_directory) if evidence_directory else None
        if self.directory:self.directory.mkdir(parents=True,exist_ok=True)
        self.evaluators={};self.episode_records=[];self.recent_transitions=deque(maxlen=64);self.sealed=set()
        # Episode competence is learned from attempts, never from PPO updates.
        # Frozen evaluation is protected by the tracker's explicit split.
        from drl.MARL.c1_competence import CompetencePatience
        self.competence_policy = CompetencePatience()
        self.competence_trackers = {}

    def _start_competence(self, env, index, evaluator, timestamp_s):
        if evaluator.episode_mode != 'guided' or evaluator.target_progress is None:
            return
        from drl.MARL.c1_competence import CompetenceEpisodeTracker
        target_count = int(evaluator.target_progress._last_match)
        # In loose mode the first observation is already the first action's
        # post-step state.  Give this attempt its real preceding frame rather
        # than manufacturing a zero-duration successful edge transition.
        start_s=(timestamp_s if evaluator.last_frame is not None else
                 max(0.,timestamp_s-float(env.step_dt)))
        tracker = CompetenceEpisodeTracker(env.domain_by_env[index], target_count,
            start_s, self.competence_policy, split=env.cfg.course_split)
        self.competence_trackers[index] = (env.generation[index], tracker)
        self._set_competence_window(evaluator, tracker)

    def _set_competence_window(self, evaluator, tracker):
        # Never mutate the shared immutable settings or another env's window.
        evaluator.stagnation_monitor.cfg = dict(evaluator.stagnation_monitor.cfg)
        evaluator.stagnation_monitor.cfg['global_window_s'] = (
            self.competence_policy.patience(tracker.domain, tracker.stage))

    def prime(self,env,index,frame):
        """Create the new generation from its actual restored pre-action state."""
        key=env.generation[index]
        current=self.evaluators.get(index)
        if current is not None and current[0]==key:raise RuntimeError('Duplicate snapshot baseline')
        if current is not None:self._episode(index,*current,reason='reset_before_termination')
        if self.settings.values.get('episode_mode')!='guided':raise ValueError('Snapshot C1 requires guided contract')
        from drl.MARL.c1_guided_feedback import load_target
        evaluator=AssemblyProbeEvaluator(env.domain_by_env[index],env.courses[index],env.cfg.possible_agents,
            settings=self.settings,max_steps=int(env.max_episode_length),event_config_path=env.cfg.reward_config_path,
            target_graph=load_target(env.domain_by_env[index]),discount_factor=getattr(env.cfg,'shaping_gamma',.99))
        evaluator.prime(frame)
        self.evaluators[index]=(key,evaluator)
        self._start_competence(env,index,evaluator,frame.time_s)

    def _emit(self,name,row):
        row=sanitized_record(row)
        if self.directory:
            with (self.directory/f'{name}.jsonl').open('a') as handle:
                handle.write(json.dumps(row,allow_nan=False)+'\n')
        return row

    def _episode(self,index,generation,evaluator,reason=None):
        key=(index,generation)
        if key in self.sealed or evaluator.last_frame is None:return
        record=evaluator.episode_record(reason=reason)
        record.update(env_index=index,generation=generation,expert_actions=0)
        self.episode_records.append(self._emit('episodes',record));self.sealed.add(key)

    def evaluate(self,env,index,frame):
        mode=getattr(env.cfg,'initialization_mode',None)
        if mode not in ('loose_modules','assembled_state'):
            raise RuntimeError('Generic C1 evaluator requires policy-controlled initialization')
        runtimes=getattr(env,'assembly_runtimes',None)
        if runtimes is not None and runtimes[index] is not None:
            raise RuntimeError('Expert runtime must be absent from generic C1 control')
        generation=env.generation[index]
        current=self.evaluators.get(index)
        if mode=='assembled_state' and (current is None or current[0]!=generation):
            raise RuntimeError('Restored snapshot baseline must be primed before policy stepping')
        if current is None or current[0]!=generation:
            if current is not None:self._episode(index,*current,reason='reset_before_termination')
            graph=None
            if (self.settings.values.get('episode_mode')!='discovery'
                    and self.settings.values.get('target_graph_shaping_mode','off') in ('temporary','potential')):
                from drl.MARL.c1_guided_feedback import load_target
                graph=load_target(env.domain_by_env[index])
            evaluator=AssemblyProbeEvaluator(env.domain_by_env[index],env.courses[index],env.cfg.possible_agents,
                settings=self.settings,max_steps=int(env.max_episode_length),
                event_config_path=env.cfg.reward_config_path,target_graph=graph,
                discount_factor=getattr(env.cfg,'shaping_gamma',.99))
            self.evaluators[index]=(generation,evaluator)
            self._start_competence(env,index,evaluator,frame.time_s)
        else:evaluator=current[1]
        signals=collect_runtime_signals(owner_ids=env.cfg.possible_agents,delta_seconds=env.step_dt,
            proposal_outcome=env._outcomes[index],measured_collision_owner_ids=frame.pathological_collision_owner_ids,
            collision_source='measured module-reference-point solid-interior penetration; no contact-force sensor')
        counts={}
        outcome=env._outcomes[index]
        projection=getattr(outcome,'projection',None)
        if projection is not None:
            counts=dict(structural_action_count=sum(c.candidate_id.startswith('structural|') for c in projection.sampled_candidates),
                projected_structural_action_count=sum(c.candidate_id.startswith('structural|') for c in projection.accepted_candidates),
                dropped_conflict_count=sum(d.code=='RESOURCE_BUSY' for d in (*projection.dropped,*outcome.physical_drops)),
                dropped_invalid_count=sum(d.code=='INVALID_GOAL' for d in (*projection.dropped,*outcome.physical_drops)))
        action_events = getattr(env, 'c1_action_events', None)
        provider = getattr(env, 'runtime_context_provider', None)
        feasibility_cache = provider.feasibility_metrics() if hasattr(provider, 'feasibility_metrics') else {}
        if action_events is not None:
            action_events.feasibility_cache_metrics = feasibility_cache
            from drl.MARL.c1_action_events import event_identity
            counts.update(action_events.complete_step(outcome=outcome,
                primitive_events=frame.primitive_events, identity=event_identity(env,index,frame.time_s)))
        goals=(
            {
                goal_id:dict(metadata,
                    primitive=metadata.get('resolved_primitive',metadata.get('primitive')),
                    module_ids=metadata.get('resolved_module_ids',metadata.get('module_ids',())),
                    arguments=tuple(metadata['resolved_parameters'].items()) if 'resolved_parameters' in metadata else metadata.get('arguments',()))
                for (env_id,gen,goal_id),metadata in action_events.goals.items()
                if env_id==index and gen==generation
            }
            if action_events is not None else {}
        )
        captured=getattr(provider,'_captured',None)
        decision_opportunity=(bool(captured[index].rollout_active) if captured is not None
                              else projection is not None)
        active={};commands={};prepared=()
        dispatcher=getattr(env,'native_dispatcher',None)
        if dispatcher is not None:
            executor=dispatcher.executor_provider(index)
            snapshot=getattr(executor,'prepared_relation_snapshot',None)
            if callable(snapshot):
                prepared=tuple(((r.source_module,r.source_face),(r.target_module,r.target_face))
                    for r in snapshot() if not r.docked)
            for goal in getattr(executor,'active_goals',()):
                primitive=getattr(goal.primitive,'value',goal.primitive)
                active[goal.goal_id]=goals.get(goal.goal_id,dict(primitive=str(primitive),module_ids=tuple(goal.module_ids)))
            for owner,command in dispatcher.continuous_commands(index).items():
                # One identity per independent policy proposal, never per
                # 240-Hz controller refresh. A held command has no new identity.
                commands[owner]=(dict(request_id=f'{generation}:{frame.time_s}',command=command)
                                 if decision_opportunity else command)
        result=evaluator.step(
            frame,
            signals=signals.signals,
            action_counts=counts,
            action_goals=goals,
            active_goals=active,decision_opportunity=decision_opportunity,
            continuous_commands=commands,prepared_relations=prepared,
        )
        tracked = self.competence_trackers.get(index)
        if tracked is not None and tracked[0] == generation:
            tracker = tracked[1]
            count = int(evaluator.target_progress._last_match)
            attempts = tracker.observe(frame.time_s,count,
                primitive_events=frame.primitive_events,
                decision_opportunity=decision_opportunity,
                finished=result.terminated or result.truncated,
                truncated=result.truncated)
            for attempt in attempts:
                self._emit('competence_attempts',dict(env_index=index,generation=generation,
                    task=env.domain_by_env[index],seed=env.seed_by_env[index],**attempt))
            if attempts and not (result.terminated or result.truncated):
                self._set_competence_window(evaluator,tracker)
            if attempts and (result.terminated or result.truncated):
                self._emit('competence_summary',dict(env_index=index,generation=generation,
                    stages=self.competence_policy.snapshot()['stages']))
        for event in evaluator.align_feedback.last_events:
            self._emit(
                'align_geometry',
                dict(
                    env_index=index,
                    generation=generation,
                    task=env.domain_by_env[index],
                    seed=env.seed_by_env[index],
                    **event,
                )
            )
        if action_events is not None:
            action_events.record_transition_result(team_reward=result.team_reward,
                valid_connections=result.assessment.valid_connections,success=result.success)
        row=dict(env_index=index,generation=generation,task=env.domain_by_env[index],seed=env.seed_by_env[index],
            timestamp_s=frame.time_s,**asdict(result),signal_availability=signals.availability,action_counts=counts,
            settings_sha256=self.settings.sha256,assembly_validity_achieved=evaluator.assembly_valid_timestamp is not None,
            functional_probe_achieved=evaluator.probe_success_timestamp is not None,C1_success=result.success,
            feasibility_cache=feasibility_cache)
        self.recent_transitions.append(self._emit('transitions',row))
        telemetry=getattr(env,'training_telemetry',None)
        if telemetry is not None:
            telemetry.observe_transition(env_index=index,domain=env.domain_by_env[index],seed=env.seed_by_env[index],generation=generation,
                reward=result.team_reward,terminated=result.terminated,truncated=result.truncated,success=result.success,
                active=True,delta_s=env.step_dt,reward_breakdown=result.reward_breakdown,
                c1_state=dict(phase=result.phase,assembly_valid_now=result.assembly_valid_now,probe_success=result.success,
                    target_graph_complete=result.target_graph_complete,assembly_stable=result.assembly_stable,
                    fold_complete=result.fold_complete,functional_probe_success=result.functional_probe_success,
                    episode_success=result.episode_success,
                    functional_probe_metric=asdict(result.probe) if result.probe else None))
        if result.terminated or result.truncated:self._episode(index,generation,evaluator)
        return result

    def flush_open_episodes(self,reason='observation_end'):
        """Finalize evidence only; does not terminate, reset or command the env."""
        for index,(generation,evaluator) in self.evaluators.items():self._episode(index,generation,evaluator,reason=reason)
