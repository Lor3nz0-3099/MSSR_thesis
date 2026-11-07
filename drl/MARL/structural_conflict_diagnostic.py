"""Passive ActionSet diagnostics. No optimizer, policy edits or admission repair.

``--task`` records ONE native env. ``--summarize`` only reads finished JSONL.
Threshold comparisons decode the SAME raw H2 prediction, off the execution path.
They do not replace the sampled bundle submitted to the native runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

TASK_SEEDS = dict(navigation=7017, stairs=12005, gap=16022, manipulation=20004)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'logs/isaaclab_c1_native/redesign/structural_conflict_diagnostic'


def serial(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SimpleNamespace):
        return serial(vars(value))
    if is_dataclass(value):
        return serial(asdict(value))
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [serial(v) for v in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    return value


def write_json(path, value):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + '\n')


def emit(handle, value):
    handle.write(json.dumps(serial(value), allow_nan=False) + '\n')


def claim_record(ref):
    from drl.IL.bc_v1.bc_v1_contract import ConnectorRef, ResourceRef
    if isinstance(ref, ResourceRef):
        return dict(kind='resource', module=ref.module.module_id, resource=ref.resource_id)
    if isinstance(ref, ConnectorRef):
        return dict(kind='connector', module=ref.module.module_id, connector=ref.connector_id)
    raise TypeError(f'Unexpected physical claim {type(ref)}')


def pair_conflicts(choices, claim_resolver):
    """Enumerate actual typed claim intersections, preserving composer order.

    Only the first intersecting action triggers assert_executable. Later pairs
    are potential additional conflicts in the same set, never extra rejections.
    """
    from drl.IL.bc_v1.bc_v1_contract import ResourceRef, ConnectorRef
    claims = {i: claim_resolver(a) for i, a in choices.items()}
    pairs = []
    reserved = set()
    first_trigger = None
    for j, refs in claims.items():
        if first_trigger is None and reserved.intersection(refs):
            first_trigger = j
        reserved.update(refs)
        for i, earlier in claims.items():
            if i == j:
                break
            shared = set(earlier).intersection(refs)
            if not shared:
                continue
            a, b = choices[i], choices[j]
            categories = []
            if any(isinstance(r, ResourceRef) for r in shared):
                categories.append('same physical resource claimed multiple times')
            if any(isinstance(r, ConnectorRef) for r in shared):
                categories.append('same connector claimed multiple times')
            same_pair = bool(a.connectors and b.connectors and set(a.connectors) == set(b.connectors))
            if same_pair:
                categories.append('duplicate connector pair')
            if same_pair and {a.action_id, b.action_id} == {'align_faces', 'dock'}:
                categories.append('align + dock on same pair in same decision')
            pairs.append(dict(candidate_a_index=i, candidate_b_index=j,
                candidate_a=serial(a), candidate_b=serial(b),
                shared_claims=sorted((claim_record(r) for r in shared), key=lambda r: json.dumps(r, sort_keys=True)),
                categories=categories, exact_reason='Conflicting physical resource claims',
                composer_rule='drl/IL/bc_v1/action_set_decoder.py::assert_executable: reserved.intersection(claims)',
                is_at_first_conflicting_action=(j == first_trigger)))
    return claims, pairs


def audit_prediction(*, prediction, batch, normalizers, binding, threshold):
    """Read-only decode of one fixed prediction, with no native dispatch.

    If the full decoder fails before constructing its bundle, decode its chosen
    members separately for forensic claims/argument attribution. Those members
    are NEVER dispatched. Full-set error and single-member errors stay distinct.
    """
    import torch
    from drl.IL.bc_v1.action_set import ActionBundle
    selected = torch.nonzero(prediction['presence_logits'] >= threshold).flatten().cpu().tolist()
    choices, member_errors = {}, {}
    error, phase = None, None
    try:
        bundle, = binding._action_set_decoder(prediction, batch, normalizers, threshold=threshold)
        if len(bundle.actions) != len(selected):
            raise AssertionError('Decoded membership differs from selected candidate indices')
        choices = dict(zip(selected, bundle.actions))
    except ValueError as exc:
        error, phase = str(exc), 'action_set_decoder'
        for index in selected:
            logits = torch.full_like(prediction['presence_logits'], -1e9)
            logits[index] = prediction['presence_logits'][index]
            try:
                member, = binding._action_set_decoder({**prediction, 'presence_logits': logits},
                    batch, normalizers, threshold=threshold)
                choice, = member.actions
                choices[index] = choice
            except ValueError as member_exc:
                member_errors[index] = str(member_exc)
    claim_errors = {}
    valid_choices = {}
    for index, choice in choices.items():
        try:
            binding._claim_resolver(choice)
            valid_choices[index] = choice
        except ValueError as exc:
            claim_errors[index] = str(exc)
    claims, pairs = pair_conflicts(valid_choices, binding._claim_resolver)
    if error is None:
        try:
            binding.decode_action_bundle(ActionBundle(tuple(choices.values())))
        except ValueError as exc:
            error = str(exc)
            phase = ('composer' if error in ('Conflicting physical resource claims',
                'Invalid physical claims', 'Physical resource unavailable') else 'adapter')
    return dict(selected=selected, choices=choices, claims=claims, conflicts=pairs,
        member_errors=member_errors, claim_errors=claim_errors,
        exact_reason=error, rejection_phase=phase, executable=error is None,
        conflict_enumeration_complete=not member_errors and not claim_errors)


def candidate_rows(*, output, context, action, task, step, seed, runtime_outcome, audit):
    import torch
    route = context.routing_snapshot
    count = len(route['resource_candidates'])
    records = output['records']
    # Rank is observational only: execution order remains original candidate order.
    ranks = {r.candidate_index: rank + 1 for rank, r in enumerate(sorted(records,
        key=lambda r: (-float(r.presence_logit), r.candidate_index)))}
    rows = []
    for record in records:
        index = record.candidate_index
        structural = index >= count
        if structural:
            spec, bound = route['structural_bindings'][index - count]
            action_id, initiator = spec.action_id, bound.initiator.module_id
            connectors = serial(bound.connectors)
            scope_domain = sorted(m.module_id for m in bound.scope_members)
            applicability = bound.applicability
        else:
            candidate = route['resource_candidates'][index]
            action_id = candidate.action.action_id
            initiator = candidate.resource_ref.module.module_id
            connectors, scope_domain, applicability = [], [], None
        selected = bool(action.presence[index])
        choice = audit['choices'].get(index)
        accepted = bool(runtime_outcome.accepted) if selected else None
        logit = float(record.presence_logit)
        physical_claims = None
        claims_source = 'not sampled; dynamic arguments/scope not evaluated'
        if index in audit['claims']:
            physical_claims = [claim_record(c) for c in audit['claims'][index]]
            claims_source = 'decoded sampled candidate'
        elif structural and action_id in ('align_faces','dock','undock'):
            # These claims depend only on fixed endpoints, never sampled payload.
            from drl.IL.bc_v1.bc_v1_contract import ActionChoice
            from drl.IL.bc_v1.live_policy import smores_claims
            physical_claims = [claim_record(c) for c in smores_claims(ActionChoice(
                action_id,bound.initiator,connectors=bound.connectors))]
            claims_source = 'fixed connector binding; independent of unsampled arguments'
        rows.append(dict(task=task, seed=seed, env_index=0, decision_step=step,
            candidate_index=index, candidate_id=record.candidate_id,
            candidate_kind=record.candidate_kind, action_id=action_id,
            initiator_module=initiator, participant_domain=scope_domain,
            participant_modules=None if choice is None else [p.module_id for p in choice.participants],
            connector_pairs=[connectors] if connectors else [],
            arguments=None if choice is None else serial(choice.parameters),
            raw_argument_heads=serial(output['structural_payloads'].get(index)) if structural else None,
            sampled_arguments={name: serial(getattr(action, name).get(index))
                for name in ('resource_numeric', 'resource_categorical', 'structural_numeric',
                'structural_boolean', 'structural_categorical', 'structural_reference',
                'structural_reference_set', 'structural_scope')},
            raw_logit=logit, sigmoid_probability=float(torch.sigmoid(record.presence_logit)),
            probability=0. if bool(context.hard_mask[index]) else float(torch.sigmoid(record.presence_logit)),
            score_used='Bernoulli(sigmoid(raw_logit))', threshold=None,
            threshold_status='no presence threshold in actual stochastic sampler',
            materialized_decoder_threshold_logit=0.,
            deterministic_threshold_0_selected=logit >= 0.,
            deterministic_calibrated_selected=logit >= output['calibrated_threshold'],
            masked=bool(context.hard_mask[index]), catalog_topology_applicability=applicability,
            runtime_trajectory_feasibility='not certified by candidate generation; native admission only',
            availability_provider='none in current native policy bridge',
            physical_claims=physical_claims, physical_claims_source=claims_source,
            candidate_rank=ranks[index], rank_used_for_selection=False,
            selected=selected, accepted=accepted, rejected=(not accepted) if selected else None,
            exact_rejection_reason=str(getattr(runtime_outcome, 'message', '')) if selected and not accepted else None,
            bundle_outcome_code=runtime_outcome.code,
            isolated_decoder_error=audit['member_errors'].get(index),
            isolated_claim_error=audit['claim_errors'].get(index),
            directly_in_conflict=any(index in (p['candidate_a_index'], p['candidate_b_index']) for p in audit['conflicts']),
            rejection_scope='whole ActionBundle; members not individually admitted' if selected and not accepted else None))
    return rows


def selection_stats(audit, rows):
    by_index = {r['candidate_index']: r for r in rows}
    chosen = [by_index[i] for i in audit['selected']]
    types = Counter(r['action_id'] for r in chosen)
    return dict(selected_resource=sum(r['candidate_kind'] == 'resource' for r in chosen),
        selected_structural=sum(r['candidate_kind'] == 'structural' for r in chosen),
        action_counts=dict(types), align_faces=types['align_faces'], dock=types['dock'],
        undock=types['undock'], bundle_cardinality=len(chosen),
        empty_bundle=not chosen,
        conflicting_claim_pairs=len(audit['conflicts']),
        shared_claim_occurrences=sum(len(p['shared_claims']) for p in audit['conflicts']),
        executable=audit['executable'], exact_rejection_reason=audit['exact_reason'],
        rejection_phase=audit['rejection_phase'],
        conflict_enumeration_complete=audit['conflict_enumeration_complete'])


def distribution(values):
    if not values:
        return dict(count=0, min=None, max=None, mean=None, p10=None, median=None, p90=None)
    values = sorted(values)
    def q(f):
        position = f * (len(values) - 1)
        low = int(position)
        return values[low] + (values[min(low + 1, len(values)-1)]-values[low])*(position-low)
    return dict(count=len(values), min=values[0], max=values[-1],
        mean=sum(values)/len(values), p10=q(.1), median=q(.5), p90=q(.9))


def summarize(directory):
    """Aggregate completed captures, clearly distinguishing missing task runs."""
    proposals, conflicts, decisions = [], [], []
    results = {}
    for task in TASK_SEEDS:
        folder = directory / task
        if (folder/'result.json').exists():
            results[task] = json.loads((folder/'result.json').read_text())
        for name, dest in [('proposals', proposals), ('conflicts', conflicts), ('decisions', decisions)]:
            path = folder / f'{name}.jsonl'
            if path.exists():
                dest.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    for name, rows in [('proposals', proposals), ('conflicts', conflicts), ('decisions', decisions)]:
        with (directory/f'{name}.jsonl').open('w') as handle:
            for row in rows:
                emit(handle, row)
    structural = [r for r in proposals if r['candidate_kind'] == 'structural']
    selected = [r for r in structural if r['selected']]
    actual_pairs = [p for p in conflicts if p['selection_rule'] == 'actual_stochastic']
    comparisons = {}
    for rule in ('actual_stochastic', 'deterministic_0', 'deterministic_calibrated'):
        captured = [d['comparisons'][rule] for d in decisions]
        comparisons[rule] = {key: sum(d[key] for d in captured) for key in (
            'selected_resource','selected_structural','align_faces','dock','undock',
            'bundle_cardinality','conflicting_claim_pairs','shared_claim_occurrences')}
        comparisons[rule].update(bundles=len(captured), executable_bundles=sum(d['executable'] for d in captured),
            non_executable_bundles=sum(not d['executable'] for d in captured),
            empty_bundles=sum(d['empty_bundle'] for d in captured),
            executable_nonempty_bundles=sum(d['executable'] and not d['empty_bundle'] for d in captured),
            incomplete_claim_enumeration=sum(not d['conflict_enumeration_complete'] for d in captured),
            bundle_cardinality_distribution=distribution([d['bundle_cardinality'] for d in captured]))
    examples = []
    seen = set()
    lookup = {(r['task'],r['decision_step'],r['candidate_index']):r for r in proposals}
    for pair in actual_pairs:
        key = (pair['task'], tuple(pair['categories']))
        if key in seen:
            continue
        seen.add(key)
        a = lookup[(pair['task'],pair['decision_step'],pair['candidate_a_index'])]
        b = lookup[(pair['task'],pair['decision_step'],pair['candidate_b_index'])]
        examples.append(dict(task=pair['task'], decision_step=pair['decision_step'],
            candidate_a={k:a[k] for k in ('candidate_id','raw_logit','probability','arguments')},
            candidate_b={k:b[k] for k in ('candidate_id','raw_logit','probability','arguments')},
            shared_claims=pair['shared_claims'], exact_reason=pair['exact_reason']))
        if len(examples) == 10:
            break
    coverage = {task:sum(r['task']==task for r in selected) for task in TASK_SEEDS}
    capture_complete = len(results)==4 and all(r['status']=='pass' for r in results.values())
    status = ('complete' if all(coverage.values()) else 'insufficient_structural_observations') if capture_complete else 'awaiting_offline_capture'
    if not decisions:
        for comparison in comparisons.values():
            for key in comparison:
                if key not in ('bundles','bundle_cardinality_distribution'):
                    comparison[key] = None
    summary = dict(status=status,
        note='Zero counters with no captured decisions mean unmeasured, not absence of conflicts.',
        task_results=results, captured_decisions=len(decisions),
        structural_candidates=len(structural) if decisions else None,
        structural_proposals_by_action=dict(Counter(r['action_id'] for r in selected)),
        selected_structural_by_task=coverage,
        accepted=sum(r['accepted'] is True for r in selected) if decisions else None,
        rejected=sum(r['rejected'] is True for r in selected) if decisions else None,
        exact_rejection_reasons=dict(Counter(r['exact_rejection_reason'] for r in selected if r['rejected'])),
        conflict_category_pair_counts=dict(Counter(c for p in actual_pairs for c in p['categories'])),
        count_semantics='Category counts overlap and count pairs; rejection counts count selected structural members of rejected bundles.',
        thresholds=comparisons, examples=examples,
        score_distributions={group:{field:distribution([r[field] for r in selected if
            (r['accepted'] if group=='accepted' else r['rejected'])]) for field in ('raw_logit','probability')}
            for group in ('accepted','rejected')},
        selected_structural_logits=distribution([r['raw_logit'] for r in selected]),
        diagnosis='pending analysis of finished native captures; code proves a documented deterministic-BC versus stochastic-RL decision-rule mismatch',
        expert_actions=0, optimizer_updates=0, behavior_fix_applied=False)
    for name in ('historical_trial','checkpoint_metrics','dataset_rarity','threshold_finding'):
        path = directory/f'{name}.json'
        if path.exists():
            summary[f'{name}_artifact'] = str(path)
    write_json(directory/'summary.json', summary)
    return summary


def replay(directory):
    """Repeat threshold/claim comparisons from saved H2, without Isaac or actors."""
    import torch
    from drl.MARL.c1_runner import build_observation_bridge
    from drl.MARL.smores_runtime_binding import SMORESMARLRuntimeBinding, materialize_marl_action_set_prediction
    from drl.IL.bc_v1.live_policy import smores_claims
    bridge = build_observation_bridge(ROOT)
    binding = SMORESMARLRuntimeBinding(adapter=bridge.adapter,claim_resolver=smores_claims)
    threshold = float(json.loads((ROOT/'configs/bc_v1/runtime_inference_v1.json').read_text())[
        'deterministic_presence_threshold_logit'])
    with (directory/'threshold_replay.jsonl').open('w') as handle:
        for task in TASK_SEEDS:
            for path in sorted((directory/task).glob('decision_*.pt')):
                # Trusted local capture, contains repository dataclasses.
                capture = torch.load(path,map_location='cpu',weights_only=False)
                prediction,batch = capture['raw_prediction'],capture['decoder_batch']
                routing = capture['routing']
                from drl.MARL.policy import bind_marl_candidates
                records = bind_marl_candidates(routing=routing,presence_logits=prediction['presence_logits'])
                nr = len(routing['resource_candidates'])
                rows = [dict(candidate_index=r.candidate_index,candidate_kind=r.candidate_kind,
                    action_id=(routing['resource_candidates'][r.candidate_index].action.action_id
                        if r.candidate_index<nr else routing['structural_bindings'][r.candidate_index-nr][0].action_id))
                    for r in records]
                sampled = materialize_marl_action_set_prediction(action=capture['sampled_action'],
                    prediction_template=prediction,routing=routing,batch=batch)
                for rule,pred,t in [('actual_stochastic',sampled,0.),
                                     ('deterministic_0',prediction,0.),
                                     ('deterministic_calibrated',prediction,threshold)]:
                    audit = audit_prediction(prediction=pred,batch=batch,normalizers=bridge.normalizers,
                        binding=binding,threshold=t)
                    emit(handle,dict(task=task,capture=path.name,selection_rule=rule,
                        threshold_logit=None if rule=='actual_stochastic' else t,
                        comparison=selection_stats(audit,rows),conflicts=audit['conflicts']))


def run_native(args):
    # AppLauncher must precede native simulator imports. One task/process only.
    from isaaclab.app import AppLauncher
    folder = args.output / args.task
    folder.mkdir(parents=True, exist_ok=False)
    files = sorted((ROOT/'drl/MARL').glob('*.py'))
    files += sorted((ROOT/'scripts/smores_ep/src/smores_ep/isaac').glob('*.py'))
    files += [ROOT/'configs/bc_v1/runtime_inference_v1.json']
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write_json(folder/'source_sha256.json',hashes)
    write_json(folder/'arguments.json',vars(args))
    application = None
    env = None
    result = dict(status='fail', task=args.task, num_envs=1, initialization_mode='loose_modules',
        expert_actions=0, optimizer_updates=0, decisions_completed=0)
    started = time.monotonic()
    try:
        application = AppLauncher(args).app
        import torch
        from drl.MARL.c1_runner import CHECKPOINT, BC_SHA256, build_models, build_observation_bridge
        from drl.MARL.isaaclab_c1_env import MSSRC1IsaacLabEnv, make_c1_isaaclab_env_cfg
        from drl.MARL.isaaclab_c1_runner import seed_native_training
        from drl.MARL.policy import sample_marl_action_bundle
        from drl.MARL.smores_runtime_binding import materialize_marl_action_set_prediction
        seed_native_training(args.global_seed)
        owners = tuple(f'smores_{i:02d}' for i in range(1,9))
        cfg = make_c1_isaaclab_env_cfg(possible_agents=owners,num_envs=1,device=args.device)
        cfg.domains = (args.task,)
        cfg.seeds = (TASK_SEEDS[args.task],)
        cfg.initialization_mode = 'loose_modules'
        cfg.seed = args.global_seed
        cfg.resample_train = False
        cfg.reward_config_path = str(ROOT/'logs/appo_smoke/calibration_validated_train/reward_config.initial.json')
        env = MSSRC1IsaacLabEnv(cfg)
        env.reset()
        bridge = build_observation_bridge(ROOT)
        actor, _ = build_models(SimpleNamespace(repository_root=ROOT,checkpoint=ROOT/CHECKPOINT,
            device=args.device),bridge)
        native = env.initialize_policy_bridge(observation_bridge=bridge)
        provider = native.provider
        binding = native.decision_bridge.binding.binding
        threshold = float(json.loads((ROOT/'configs/bc_v1/runtime_inference_v1.json').read_text())[
            'deterministic_presence_threshold_logit'])
        def actor_hash():
            digest = hashlib.sha256()
            for name, value in sorted(actor.state_dict().items()):
                digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
            return digest.hexdigest()
        before = actor_hash()
        with (folder/'proposals.jsonl').open('w',buffering=1) as proposals, \
             (folder/'conflicts.jsonl').open('w',buffering=1) as conflicts, \
             (folder/'decisions.jsonl').open('w',buffering=1) as decisions:
            for step in range(args.steps):
                context, = provider.current_action_contexts()
                if not env.handoffs[0].marl_active:
                    raise RuntimeError('Policy authority unavailable in loose_modules diagnostic')
                with torch.no_grad():
                    output = actor.forward_all_owners(graph=context.actor_graph,
                        routing=context.routing_snapshot,possible_agents=owners)
                    bundle = sample_marl_action_bundle(**{key:output[key] for key in (
                        'records','resource_payloads','structural_payloads','resource_log_std','structural_log_std')},
                        hard_mask=context.hard_mask)
                    decision = SimpleNamespace(action=bundle, action_set_prediction_template=output['prediction_template'])
                    batch = provider.decoder_batch(0,context)
                    # Save replayable pre-admission data BEFORE semantic decode.
                    torch.save(dict(actor_graph=context.actor_graph,routing=context.routing_snapshot,
                        hard_mask=context.hard_mask,raw_prediction=output['prediction_template'],
                        sampled_action=bundle,decoder_batch=batch),folder/f'decision_{step:03d}.pt')
                    actions = provider.encode_env_actions([decision])
                    env.step(actions)
                    outcome = env._outcomes[0]
                    # Offline comparisons: these predictions never reach dispatch.
                    cpu_rng = torch.random.get_rng_state().clone()
                    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                    sampled = materialize_marl_action_set_prediction(action=bundle,
                        prediction_template=output['prediction_template'],routing=context.routing_snapshot,batch=batch)
                    actual = audit_prediction(prediction=sampled,batch=batch,normalizers=bridge.normalizers,
                        binding=binding,threshold=0.)
                    output['calibrated_threshold'] = threshold
                    rows = candidate_rows(output=output,context=context,action=bundle,task=args.task,
                        step=step,seed=TASK_SEEDS[args.task],runtime_outcome=outcome,audit=actual)
                    audits = {'actual_stochastic':actual}
                    for rule, value in [('deterministic_0',0.),('deterministic_calibrated',threshold)]:
                        audits[rule] = audit_prediction(prediction=output['prediction_template'],batch=batch,
                            normalizers=bridge.normalizers,binding=binding,threshold=value)
                    if not torch.equal(cpu_rng,torch.random.get_rng_state()) or any(
                        not torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())):
                        raise AssertionError('Diagnostic observer consumed policy RNG')
                    for row in rows:
                        emit(proposals,row)
                    for rule, audit in audits.items():
                        for pair in audit['conflicts']:
                            emit(conflicts,dict(task=args.task,decision_step=step,selection_rule=rule,
                                full_bundle_rejection_reason=audit['exact_reason'],
                                original_composer_reached=audit['rejection_phase']!='action_set_decoder',**pair))
                    emit(decisions,dict(task=args.task,decision_step=step,calibrated_threshold=threshold,
                        comparisons={rule:selection_stats(audit,rows) for rule,audit in audits.items()},
                        runtime_outcome=serial(outcome),physics_step=env._sim_step_counter,
                        raw_snapshot=env._pre_raw[0]))
                provider.current_critic_graphs()
                result['decisions_completed'] += 1
        result.update(status='pass',bc_checkpoint=str(ROOT/CHECKPOINT),bc_sha256=BC_SHA256,
            actor_unchanged=before==actor_hash(),calibrated_threshold=threshold,
            actual_presence_rule='independent Bernoulli(raw logits), no threshold',
            physical_expert_assembly_records=len(env.assembly_records))
        if not result['actor_unchanged'] or env.assembly_records:
            raise AssertionError('Diagnostic changed actor or used Expert')
    except Exception as exc:
        result.update(status='fail',error=str(exc),traceback=traceback.format_exc())
    finally:
        result['wall_seconds'] = time.monotonic()-started
        result['runtime_sources_unchanged'] = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        if not result['runtime_sources_unchanged']:
            result['status'] = 'fail'
        write_json(folder/'result.json',result)
        print('STRUCTURAL_DIAGNOSTIC_RESULT',json.dumps(result),flush=True)
        if env is not None:
            env.close()
        if application is not None:
            application.close()
    return 0 if result['status']=='pass' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--task',choices=tuple(TASK_SEEDS))
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--global-seed',type=int,default=20261029)
    # Keep offline aggregation independent of Isaac installation/launcher.
    import sys
    if '--summarize' not in sys.argv and '--replay' not in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.replay:
        replay(args.output)
        return 0
    if args.summarize:
        args.output.mkdir(parents=True,exist_ok=True)
        summary = summarize(args.output)
        print(json.dumps({key:summary[key] for key in ('status','captured_decisions','structural_proposals_by_action','accepted','rejected')}))
        return 0 if summary['status']=='complete' else 1
    if args.task is None or not 1 <= args.steps <= 12:
        parser.error('Requires --task and 1..12 steps; long experiments excluded')
    return run_native(args)


if __name__ == '__main__':
    raise SystemExit(main())
