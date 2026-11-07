"""User-run offline native diagnostic of calibrated sampling and projection.

No learners/optimizers/Expert. One env/process; finished legacy diagnostics are
read-only reference. --summarize never imports or starts Isaac.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from drl.MARL.structural_conflict_diagnostic import TASK_SEEDS,ROOT,serial,write_json,emit,distribution

OUTPUT = ROOT/'logs/isaaclab_c1_native/redesign/action_projection_diagnostic'
STREAMS = ('raw_proposals','projected_actions','dropped_candidates','conflicts','decisions','physics')
STRUCTURAL = ('align_faces','dock','undock','gravity_settle')


def summarize(directory):
    directory=Path(directory)
    all_rows={name:[] for name in STREAMS}
    results={}
    for task in TASK_SEEDS:
        folder=directory/task
        if (folder/'result.json').exists():results[task]=json.loads((folder/'result.json').read_text())
        for name in STREAMS:
            path=folder/f'{name}.jsonl'
            if path.exists():all_rows[name].extend(json.loads(l) for l in path.read_text().splitlines() if l)
    for name,rows in all_rows.items():
        with (directory/f'{name}.jsonl').open('w') as handle:
            for row in rows:emit(handle,row)
    decisions=all_rows['decisions']
    def total(key):return sum(d[key] for d in decisions) if decisions else None
    observed,failed=set(),set()
    for frame in all_rows['physics']:
        for status in frame['primitive_statuses']:
            if status['primitive'] not in STRUCTURAL:continue
            key=(frame['task'],status['goal_id'])
            if status['state'] in ('running','succeeded'):observed.add(key)
            if status['state']=='failed':failed.add(key)
    after=dict(captured_decisions=len(decisions),raw_structural_selections=total('raw_structural_selections'),
        raw_cardinality=total('raw_cardinality'),projected_cardinality=total('projected_cardinality'),
        native_admitted_cardinality=total('native_admitted_cardinality'),
        projected_structural_actions=total('projected_structural'),native_admitted_structural_actions=total('native_admitted_structural'),
        conflict_pairs=total('conflict_pairs'),
        projected_executable_nonempty_bundles=sum(d['projected_cardinality']>0 for d in decisions) if decisions else None,
        native_admitted_nonempty_bundles=sum(d['native_admitted_cardinality']>0 for d in decisions) if decisions else None,
        observed_executing_structural_goals=len(observed) if decisions else None,
        observed_failed_structural_goals=len(failed) if decisions else None,
        raw_cardinality_distribution=distribution([d['raw_cardinality'] for d in decisions]),
        projected_cardinality_distribution=distribution([d['projected_cardinality'] for d in decisions]),
        drop_reasons=dict(Counter(d['reason'] for d in all_rows['dropped_candidates'])),
        conflict_event_steps=sum(bool(d['conflict_owner_ids']) for d in decisions),
        invalid_event_steps=sum(bool(d['invalid_owner_ids']) for d in decisions))
    ref=directory/'reference_summary.json'
    summary=dict(status='capture_complete_pending_analysis' if len(results)==4 and all(r['status']=='pass' for r in results.values()) else 'awaiting_offline_diagnostic',
        before_reference=json.loads(ref.read_text()) if ref.exists() else None,after=after,
        results=results,optimizer_updates=0,expert_actions=0,C1_training_ready=False,
        note='Projection/native admission and observed running/succeeded primitive goals are distinct. No assembly/task success claimed; zero counters without captures are unmeasured.',
        penalty_semantics='Existing conflict/invalid owner events; loose-mode task reward/probe remains gated and unchanged.')
    write_json(directory/'summary.json',summary)
    return summary


def run(args):
    folder=args.output/args.task
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'arguments.json',vars(args))
    source_files=sorted((ROOT/'drl/MARL').glob('*.py'))+sorted((ROOT/'scripts/smores_ep/src/smores_ep/isaac').glob('*.py'))
    source_files += [ROOT/'configs/bc_v1/runtime_inference_v1.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    write_json(folder/'source_sha256.json',hashes)
    env,application=None,None
    handles={name:(folder/f'{name}.jsonl').open('w',buffering=1) for name in STREAMS}
    result=dict(status='fail',task=args.task,seed=TASK_SEEDS[args.task],num_envs=1,initialization_mode='loose_modules',
        optimizer_updates=0,expert_actions=0,decisions_completed=0)
    start=time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        application=AppLauncher(args).app
        import torch
        from drl.MARL.c1_runner import build_models,build_observation_bridge,CHECKPOINT,BC_SHA256
        from drl.MARL.isaaclab_c1_env import MSSRC1IsaacLabEnv,make_c1_isaaclab_env_cfg
        from drl.MARL.isaaclab_c1_runner import seed_native_training
        from drl.MARL.policy import sample_marl_action_bundle,compute_owner_action_log_probs
        from drl.MARL.runtime_signals import collect_runtime_signals
        seed_native_training(args.global_seed)
        owners=tuple(f'smores_{i:02d}' for i in range(1,9))
        cfg=make_c1_isaaclab_env_cfg(possible_agents=owners,num_envs=1,device=args.device)
        cfg.domains=(args.task,);cfg.seeds=(TASK_SEEDS[args.task],)
        cfg.seed=args.global_seed;cfg.initialization_mode='loose_modules';cfg.resample_train=False
        cfg.reward_config_path=str(ROOT/'logs/appo_smoke/calibration_validated_train/reward_config.initial.json')
        env=MSSRC1IsaacLabEnv(cfg);env.reset()
        bridge=build_observation_bridge(ROOT)
        actor,_=build_models(SimpleNamespace(repository_root=ROOT,checkpoint=ROOT/CHECKPOINT,device=args.device),bridge)
        native=env.initialize_policy_bridge(observation_bridge=bridge)
        def actor_hash():
            h=hashlib.sha256()
            for name,tensor in sorted(actor.state_dict().items()):
                h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
            return h.hexdigest()
        before=actor_hash()
        step_index=-1
        original_native_step=env.native_actuation.step
        def observed_physics(i):
            output=original_native_step(i)
            emit(handles['physics'],dict(task=args.task,decision_step=step_index,
                physics_step=env._sim_step_counter,primitive_statuses=serial(output.primitive_statuses),
                primitive_commands=serial(output.primitive_commands),applied_commands=serial(output.applied_commands),
                wheel_rates=serial(output.wheel_rates)))
            return output
        env.native_actuation.step=observed_physics
        for step_index in range(args.steps):
            context,=native.provider.current_action_contexts()
            if not env.handoffs[0].marl_active:raise RuntimeError('No native policy authority')
            with torch.no_grad():
                output=actor.forward_all_owners(graph=context.actor_graph,routing=context.routing_snapshot,possible_agents=owners)
                sample_args={k:output[k] for k in ('records','resource_payloads','structural_payloads','resource_log_std','structural_log_std')}
                raw=sample_marl_action_bundle(**sample_args,hard_mask=context.hard_mask)
                raw_presence=raw.presence.clone()
                log_probs=compute_owner_action_log_probs(**sample_args,action=raw,hard_mask=context.hard_mask,possible_agents=owners)
                torch.save(dict(raw_prediction=output['prediction_template'],sampled_action=raw,
                    routing=context.routing_snapshot,actor_graph=context.actor_graph,
                    decoder_batch=native.provider.decoder_batch(0,context),hard_mask=context.hard_mask,
                    calibration_sha256=actor.presence_calibration_sha256,
                    presence_threshold_logit=actor.presence_threshold_logit),folder/f'decision_{step_index:03d}.pt')
                carrier=native.provider.encode_env_actions([SimpleNamespace(action=raw,action_set_prediction_template=output['prediction_template'])])
                projection=env.staged[0][1].projection
                if projection is None or projection.raw_sampled_action is not raw:
                    raise AssertionError('Projected path lost raw policy action')
                env.step(carrier)
                outcome=env._outcomes[0]
                if outcome.projection is not projection:raise AssertionError('Native admission lost projection metadata')
                if not torch.equal(raw_presence,raw.presence):raise AssertionError('Projection rewrote sampled presence')
                reevaluated=compute_owner_action_log_probs(**sample_args,action=raw,hard_mask=context.hard_mask,possible_agents=owners)
                if any(not torch.equal(log_probs[o],reevaluated[o]) for o in owners):
                    raise AssertionError('Raw action likelihood changed during projection')
                candidates={c.candidate_id:c for c in projection.sampled_candidates}
                records={r.candidate_index:r for r in output['records']}
                projected=set(projection.accepted_candidate_ids);admitted=set(outcome.executed_candidate_ids)
                for candidate in projection.sampled_candidates:
                    record=records[candidate.candidate_index]
                    emit(handles['raw_proposals'],dict(task=args.task,decision_step=step_index,
                        candidate=serial(candidate),raw_logit=float(record.presence_logit),
                        effective_logit=float(record.effective_presence_logit),
                        calibrated_probability=float(torch.sigmoid(record.effective_presence_logit)),
                        threshold_logit=actor.presence_threshold_logit,
                        projected=candidate.candidate_id in projected,native_admitted=candidate.candidate_id in admitted))
                for candidate in projection.accepted_candidates:
                    emit(handles['projected_actions'],dict(task=args.task,decision_step=step_index,
                        candidate=serial(candidate),native_admitted=candidate.candidate_id in admitted))
                for stage,drops in [('projection',projection.dropped),('native_admission',outcome.physical_drops)]:
                    for drop in drops:emit(handles['dropped_candidates'],dict(task=args.task,decision_step=step_index,stage=stage,**serial(drop)))
                for conflict in projection.conflicts:
                    a,b=candidates[conflict.candidate_a],candidates[conflict.candidate_b]
                    align_dock=bool(a.choice and b.choice and {a.choice.action_id,b.choice.action_id}=={'align_faces','dock'}
                        and set(a.choice.connectors)==set(b.choice.connectors))
                    emit(handles['conflicts'],dict(task=args.task,decision_step=step_index,
                        align_dock_same_pair=align_dock,choice_a=serial(a.choice),choice_b=serial(b.choice),**serial(conflict)))
                signals=collect_runtime_signals(owner_ids=owners,delta_seconds=env.step_dt,proposal_outcome=outcome)
                def structural_count(ids):return sum(candidates[c].choice is not None and candidates[c].choice.action_id in STRUCTURAL for c in ids)
                emit(handles['decisions'],dict(task=args.task,decision_step=step_index,
                    raw_cardinality=int(raw.presence.sum()),projected_cardinality=len(projected),native_admitted_cardinality=len(admitted),
                    raw_structural_selections=sum(records[c.candidate_index].candidate_kind=='structural' for c in candidates.values()),
                    projected_structural=structural_count(projected),native_admitted_structural=structural_count(admitted),
                    conflict_pairs=len(projection.conflicts),conflict_owner_ids=outcome.conflict_owner_ids,invalid_owner_ids=outcome.invalid_owner_ids,
                    raw_owner_log_probs=serial(log_probs),penalty_signals=serial(signals),
                    before_snapshot=env._pre_raw[0],after_snapshot=env._post_raw[0]))
            native.provider.current_critic_graphs()
            result['decisions_completed']+=1
        env.native_actuation.step=original_native_step
        result.update(status='pass',bc_sha256=BC_SHA256,actor_unchanged=before==actor_hash(),
            calibrated_threshold_logit=actor.presence_threshold_logit,calibration_sha256=actor.presence_calibration_sha256,
            calibration_path=actor.presence_calibration_path,raw_action_log_probs_preserved=True,
            expert_assembly_records=len(env.assembly_records))
        if not result['actor_unchanged'] or env.assembly_records:raise AssertionError('Actor changed or Expert used')
    except Exception as exc:
        result.update(status='fail',error=str(exc),traceback=traceback.format_exc())
    finally:
        for handle in handles.values():handle.close()
        result['wall_seconds']=time.monotonic()-start
        result['sources_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        if not result['sources_unchanged']:result['status']='fail'
        write_json(folder/'result.json',result)
        print('C1_ACTION_PROJECTION_DIAGNOSTIC_RESULT',json.dumps(result),flush=True)
        if env is not None:env.close()
        if application is not None:application.close()
    return 0 if result['status']=='pass' else 1


def main():
    import sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--task',choices=tuple(TASK_SEEDS))
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--global-seed',type=int,default=20261029)
    if '--summarize' not in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.summarize:
        args.output.mkdir(parents=True,exist_ok=True)
        summary=summarize(args.output)
        print(json.dumps({'status':summary['status'],'after':summary['after']}))
        return 0 if summary['status']=='capture_complete_pending_analysis' else 1
    if args.task is None or not 1<=args.steps<=12:parser.error('Requires --task and 1..12 steps')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
