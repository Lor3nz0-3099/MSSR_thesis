"""Read frozen aggregate files only. No torch, dataset, inference or tuning."""
from pathlib import Path
import sys,json,math,hashlib,subprocess
A=Path(__file__).resolve().parent;R=Path('/home/lorenzo/MSSR_thesis')
def load(n):return json.loads((A/n).read_text())
def save(n,v):(A/n).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
def git(*args):return subprocess.run(['git',*args],capture_output=True,check=True).stdout

def main():
    verdict=sys.argv[1]
    classifications={
        'A':('OFFICIAL TEST EVALUATION COMPLETE; BC-v4 SUITABLE FOR MAPPO WARM START','SUITABLE_FOR_MAPPO_WARM_START'),
        'B':('OFFICIAL TEST EVALUATION COMPLETE; BC-v4 GENERALIZES BUT IS MARGINAL AS MAPPO WARM START','MARGINAL_FOR_MAPPO_WARM_START'),
        'C':('OFFICIAL TEST EVALUATION COMPLETE; BC-v4 NOT SUITABLE AS MAPPO WARM START','NOT_SUITABLE_FOR_MAPPO_WARM_START')}
    title,warm=classifications[verdict]
    log=load('test_access_log.json');g=load('test_global_metrics.json');loss=load('test_loss_families.json')
    assert log['status']=='COMPLETE' and log['result_frozen']
    assert log['official_test_evaluation_number']==1 and log['official_TEST_evaluation_consumed']
    assert log['TEST_sample_count']==log['number_of_valid_model_predictions']==g['samples']==18536
    assert log['TEST_episode_count']==12 and log['number_of_batches']==290
    assert all(x['attempts']==1 and x['completed'] for x in log['episode_payload_loads'].values())
    assert sum(x['samples_metadata'] for x in log['episode_payload_loads'].values())==18536
    assert (A/'parameter_hash_before.txt').read_bytes()==(A/'parameter_hash_after.txt').read_bytes()
    config=load('test_evaluation_config.json');identity=load('checkpoint_identity.json')
    assert sha(identity['path'])==identity['sha256']==config['checkpoint_sha256']==log['checkpoint_sha256']
    action=load('test_per_action_metrics.json');tasks=load('test_task_breakdown.json');episodes=load('test_episode_breakdown.json')
    errors=load('test_set_error_distribution.json');patterns=load('test_fp_fn_breakdown.json');comparison=load('val_vs_test.json')
    assert sum(v['samples'] for v in tasks.values())==sum(v['samples'] for v in episodes.values())==18536
    assert sum(x['count'] for x in errors.values())==sum(x['count'] for x in patterns.values())==18536
    assert sum(v['tp'] for v in action['actions'].values())==g['true_positive']
    assert sum(v['fp'] for v in action['actions'].values())==g['false_positive']
    assert sum(v['fn'] for v in action['actions'].values())==g['false_negative']
    num=action['normalized_numeric_by_action'];global_num=g['normalized_numeric']['payload']['1']
    assert sum(v['1']['support'] for v in num.values())==global_num['support']
    sq=math.fsum(v['1']['support']*(v['1']['rmse'] or 0)**2 for v in num.values())
    assert math.isclose(math.sqrt(sq/global_num['support']),g['normalized_payload_rmse'],rel_tol=1e-12)
    # Byte preservation of the original dirty source/config/run state.
    prior=load('preservation_before.json');fail=[p for p,h in prior['files'].items() if sha(R/p)!=h]
    assert not fail, 'Preservation failure: '+repr(fail)
    status=git('status','--short','--untracked-files=all');diff=git('diff','--binary');index=git('diff','--cached','--binary')
    kept=[s for s in status.decode().splitlines() if str(A.relative_to(R)) not in s]
    assert kept==prior['status_lines']
    assert hashlib.sha256(diff).hexdigest()==prior['tracked_diff_sha']
    assert hashlib.sha256(index).hexdigest()==prior['index_sha']
    check=subprocess.run(['git','diff','--check'],capture_output=True);assert check.returncode==0
    (A/'git_after.txt').write_bytes(status+b'\n--- TRACKED DIFF ---\n'+diff+b'\n--- INDEX DIFF ---\n'+index)
    save('preservation.json',{'PASS':True,'source_config_run_files_hashed':len(prior['files']),
        'all_original_hashes_equal':True,'original_git_status_preserved':True,'tracked_diff_unchanged':True,
        'index_unchanged':True,'index_empty':not bool(index),'diff_check_exit_code':check.returncode,
        'writes':'Only this new audit directory','no_reset_restore_clean_stash_checkout_add_commit_push':True,
        'NO_PRODUCTION_CODE_MODIFIED':True,'checkpoint_bytes_unchanged':True})
    top=sorted(action['actions'].items(),key=lambda kv:kv[1]['support'],reverse=True)[:5]
    worst_task=min(tasks,key=lambda k:tasks[k]['exact_match']);best_task=max(tasks,key=lambda k:tasks[k]['exact_match'])
    worst_episode=min(episodes,key=lambda k:episodes[k]['exact_match']);best_episode=max(episodes,key=lambda k:episodes[k]['exact_match'])
    summary={'classification':verdict+' — '+title,'warm_start_verdict':warm,'checkpoint':identity,
        'global':g,'loss_families':loss,'error_distribution':errors,'FP_FN_distribution':patterns,
        'top5_by_support':dict(top),'rare_structural':{k:action['actions'][k] for k in ('align_faces','dock','undock','gravity_settle')},
        'best_task':{'id':best_task,**tasks[best_task]},'worst_task':{'id':worst_task,**tasks[worst_task]},
        'best_episode':{'id':best_episode,**episodes[best_episode]},'worst_episode':{'id':worst_episode,**episodes[worst_episode]},
        'VAL_vs_TEST':comparison,'official_TEST_evaluation_number':1,'official_TEST_result_frozen':True,
        'audit_pre_access_incident':load('pre_access_incident.json'),'preservation':'PASS',
        'NO_THRESHOLD_TUNING':True,'NO_CHECKPOINT_COMPARISON':True,'NO_RETRAINING':True,
        'NO_PRODUCTION_CODE_MODIFIED':True}
    save('final_release_summary.json',summary)
    lines=['# Final BC-v4 official one-shot TEST assessment','',summary['classification'],'',
        f"Checkpoint: `{identity['path']}`; epoch5; SHA256 `{identity['sha256']}`.",
        'Official TEST evaluation number=1.18536samples,12episodes,290batches; one model prediction per sample.',
        'Threshold logit0.0; FP32 eager; TF32/AMP/compileOFF; eval/no_grad; no optimizer/backward.',
        'Frozen membership metrics use exactly production VAL raw candidate identities before physical feasibility/claim validation. Exact-set ignores numeric values; matched normalized payload metrics evaluate values separately.',
        '', '## A. Generalization','', '| Metric | VAL best epoch5 | TEST | TEST minus VAL |','|---|---:|---:|---:|']
    lines += [f"| {k} | {v['VAL_best']:.9g} | {v['TEST']:.9g} | {v['TEST_minus_VAL']:+.9g} |" for k,v in comparison['metrics'].items()]
    lines += ['', 'Task and episode dispersion is fully preserved in test_task_breakdown.json and test_episode_breakdown.json; current task names are retained without inferring assembly/reconfiguration labels from episode names.',
        '', '## B. Action-set understanding','',
        f"TEST precision={g['precision']:.6f},recall={g['recall']:.6f},F1={g['f1']:.6f},Jaccard={g['jaccard']:.6f}.",
        f"Target cardinality mean={g['target_cardinality']:.6f} [{g['target_cardinality_min']},{g['target_cardinality_max']}]; predicted mean={g['predicted_cardinality']:.6f} [{g['predicted_cardinality_min']},{g['predicted_cardinality_max']}]; absolute cardinality error={g['cardinality_absolute_error']:.6f}.",
        f"INITIAL RANDOM VAL REFERENCE ONLY: F1={comparison['INITIAL_RANDOM_VAL_REFERENCE']['metrics']['f1']:.6f},exact-set=0,predicted cardinality={comparison['INITIAL_RANDOM_VAL_REFERENCE']['metrics']['predicted_cardinality']:.6f}. This is not a random TEST baseline and no random TEST model was evaluated.",
        '', '| Action | Support | TP | FP | FN | Precision | Recall | F1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {k} | {v['support']} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']:.6f} | {v['recall']:.6f} | {v['f1']:.6f} |" for k,v in sorted(action['actions'].items())]
    lines += ['', 'Support0 means no positive TEST evidence; inherited empty-set scoring does not prove that action learned.',
        '', '## C. Dominant failure mode','',
        f"FP={g['false_positive']},FN={g['false_negative']}; FP/sample={g['mean_FP_per_sample']:.6f},FN/sample={g['mean_FN_per_sample']:.6f}.",
        'Largest FP contributions: '+', '.join(f"{k}:{v['fp']}" for k,v in sorted(action['actions'].items(),key=lambda kv:kv[1]['fp'],reverse=True)[:5])+'.',
        f"Normalized matched payload MAE={global_num['mae']:.6f},RMSE={global_num['rmse']:.6f},valid scalar support={global_num['support']}. Numeric statistics are conditional on matched actions and cannot compensate for FP/FN or establish physical task success.",
        f"Worst task exact-set: {worst_task}={tasks[worst_task]['exact_match']:.6f} (n={tasks[worst_task]['samples']}); best: {best_task}={tasks[best_task]['exact_match']:.6f} (n={tasks[best_task]['samples']}).",
        f"Worst episode: {worst_episode}={episodes[worst_episode]['exact_match']:.6f} (n={episodes[worst_episode]['samples']}); best: {best_episode}={episodes[best_episode]['exact_match']:.6f} (n={episodes[best_episode]['samples']}).",
        '', '## D. Exact-set interpretation','', '| FP+FN | Samples | Fraction |','|---|---:|---:|']
    lines += [f"| {k} | {errors[k]['count']} | {errors[k]['fraction']:.6%} |" for k in ('0','1','2','3','4+')]
    lines += ['',f"Exact={errors['0']['count']}; off-by-one={errors['1']['count']}; at least2errors={18536-errors['0']['count']-errors['1']['count']}.",
        'This distinguishes exact membership and near misses from larger bundle errors; it does not equate any non-exact sample with random behavior.',
        '', '| Pattern | Samples | Fraction |','|---|---:|---:|']
    lines += [f"| {k} | {v['count']} | {v['fraction']:.6%} |" for k,v in patterns.items()]
    lines += ['', '## E. Warm start','', warm,
        (A/'assessment_rationale.md').read_text(),
        'This is an actor-initialization assessment, not evidence of autonomous execution success. No changes to BC checkpoint/threshold/training are made or recommended from TEST.',
        '', '## Integrity and access','',
        f"TEST wall time={g['wall_time_s']:.3f}s; throughput={g['samples_per_s']:.3f}samples/s, metadata only; no profiling.",
        f"Model state SHA256 before/after identical: `{g['parameter_hash_before']}`. Checkpoint bytes unchanged.",
        'Streaming G1 TEST episode loading and in-memory actor projection/tensorization reused frozen F2/current candidates; no persistent TEST sidecar. All12episode payload loaders called once. Hash authentication reads bytes separately from deserialization; there is one logical TEST iterator and no second inference/metric pass.',
        'Pre-access incident: audit-local SyntaxError before Python execution/import/TEST access; corrected and syntax-checked. No execution claim, payload or prediction existed for that parser failure. The only actual evaluation completed once, with no inference retry.',
        'Only audit-local scripts/documents/results created; all production/tests/config/source datasets/normalizers/run artifacts preserved. See preservation.json and git snapshots.',
        'NO THRESHOLD TUNING. NO CHECKPOINT COMPARISON. NO RETRAINING. NO PRODUCTION CODE MODIFIED. OFFICIAL TEST RESULT FROZEN.']
    text='\n'.join(lines)+'\n';(A/'final_bc_assessment.md').write_text(text);(A/'README.md').write_text(text)
    print(json.dumps({'classification':summary['classification'],'integrity':'PASS','preservation':'PASS',
        'source_config_run_files_checked':len(prior['files']),'global':g,'families':loss,'errors':errors,
        'patterns':patterns,'top5':dict(top),'rare':summary['rare_structural'],'worst_task':worst_task,'best_task':best_task,
        'worst_episode':worst_episode,'best_episode':best_episode,'comparison':comparison['metrics']},indent=2))

if __name__=='__main__':main()
