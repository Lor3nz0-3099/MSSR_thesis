"""Explicit authorized one-shot TEST. Exclusive claim; never auto-retry."""
from audit_common import *
import os
import time
import collections
import traceback
import math
import subprocess
import torch

def main():
    # The production TRAIN/VAL seal stays intact. Only this audit owns TEST.
    claim=A/'official_execution_claim.json'
    with claim.open('x') as f:json.dump({'claimed_UTC':now(),'PID':os.getpid(),'number':1},f)
    log={'TEST_access_authorized':True,'official_test_evaluation_number':1,
         'official_TEST_evaluation_consumed':False,'status':'PREFLIGHT',
         'start_timestamp':None,'end_timestamp':None,'checkpoint_sha256':None,
         'TEST_sample_count':0,'TEST_episode_count':0,'number_of_batches':0,
         'number_of_valid_model_predictions':0,'model_forward_calls':0,
         'episode_payload_loads':{},'no_retry_after_predictions':True}
    save('test_access_log.json',log)
    started=None
    try:
        guard=subprocess.run(['pgrep','-af',r'drl\.IL\.bc_v1\.train_action_set.*g2_v4_actor_visible_r1m'],capture_output=True,text=True)
        assert guard.returncode==1, 'Training process guard failed: '+guard.stdout
        assert load(A/'composition_preflight.json')['PASS'] is True
        assert load(A/'composition_preflight.json')['TEST_payload_access']==0
        frozen=load(A/'test_evaluation_config.json')
        assert sha(BEST)==frozen['checkpoint_sha256']
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        rt=runtime()
        model=rt['model']
        assert not torch.cuda.is_initialized()
        assert torch.cuda.is_available(), 'Expected GPU unavailable; TEST not accessed'
        model=model.to('cuda:0').eval()
        assert all(not m.training for m in model.modules())
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        before=model_hash(model)
        (A/'parameter_hash_before.txt').write_text(before+'\n')
        assert before==load(A/'composition_preflight.json')['parameter_hash']
        from drl.IL.bc_v1 import dataset as g1,train_action_set as runner
        from drl.IL.bc_v1.action_set_metrics import ActionSetMetrics,_scores
        from drl.IL.bc_v1.action_set_targets import collate_action_set_samples
        from drl.IL.bc_v1.trainer import FamilyAccumulator
        from drl.IL.bc_v1.training import move_batch_to_device,family_statistics
        aggregate=FamilyAccumulator({k:1. for k in runner.FAMILIES})
        task_metrics={};episode_metrics={};cards={};errors=collections.Counter();patterns=collections.Counter()
        payload_by_action=ActionSetMetrics()

        class AllMetrics(ActionSetMetrics):
            contexts=();position=0
            def update(self,target,predicted,*,numeric=True):
                assert self.position<len(self.contexts)
                ctx=self.contexts[self.position];self.position+=1
                oldfp,oldfn=self.fp,self.fn
                super().update(target,predicted,numeric=numeric)
                fp,fn=self.fp-oldfp,self.fn-oldfn
                task=ctx['task'] if ctx['task'] is not None else 'UNKNOWN'
                for d,k in ((task_metrics,task),(episode_metrics,ctx['episode'])):
                    d.setdefault(k,ActionSetMetrics()).update(target,predicted,numeric=False)
                n=len(target.actions)
                c=cards.setdefault(n,{'samples':0,'exact':0,'fp':0,'fn':0})
                c['samples']+=1;c['exact']+=int(fp+fn==0);c['fp']+=fp;c['fn']+=fn
                errors[min(fp+fn,4)]+=1
                patterns['both' if fp and fn else 'FP_only' if fp else 'FN_only' if fn else 'exact']+=1

        metrics=AllMetrics()
        # Dataset construction authenticates manifest/counts only, before payload.
        data=g1.BCDataset(G1,'test',expected_identity=rt['physical']['identity'])
        assert len(data)==18536 and len(data.episode_ids)==12
        assert list(data.episode_ids)==rt['split']['splits']['test']
        original_load=g1._load_episode
        def audited_load(root,eid,entry):
            assert eid in data.episode_ids
            assert eid not in log['episode_payload_loads'], 'Second TEST episode payload load forbidden'
            log['episode_payload_loads'][eid]={'attempts':1,'samples_metadata':entry['samples'],'completed':False}
            save('test_access_log.json',log)
            result=original_load(root,eid,entry)
            log['episode_payload_loads'][eid]['completed']=True
            log['TEST_episode_count']+=1
            save('test_access_log.json',log)
            return result
        g1._load_episode=audited_load
        log.update(status='RUNNING',start_timestamp=now(),checkpoint_sha256=sha(BEST))
        save('test_access_log.json',log);started=time.perf_counter()
        print('OFFICIAL TEST START; frozen best epoch5; 18536 samples /12 episodes /290 batches',flush=True)
        try:
            with torch.no_grad():
                for lo in range(0,len(data),64):
                    rows=[];contexts=[]
                    for i in range(lo,min(lo+64,len(data))):
                        sample=data[i]
                        contexts.append({'episode':sample.audit.episode_id,'task':sample.inputs.policy.context.task_id})
                        rows.append(tensorize(sample,rt))
                        log['TEST_sample_count']+=1
                    host=collate_action_set_samples(rows)
                    batch=move_batch_to_device(host,'cuda:0')
                    prediction=model(batch['inputs'])
                    log['model_forward_calls']+=1
                    assert torch.isfinite(prediction['presence_logits']).all()
                    log['number_of_valid_model_predictions']+=len(rows)
                    # Persist consumption immediately; any subsequent failure stops permanently.
                    save('test_access_log.json',log)
                    _,stats=runner.compute_action_set_loss(prediction,batch)
                    aggregate.add(family_statistics(stats))
                    metrics.contexts=contexts;metrics.position=0
                    runner.measure_batch(prediction,batch,metrics,threshold=0.0)
                    assert metrics.position==len(rows)
                    # Same existing matched numeric metric contract as measure_batch,
                    # with action_id attribution; no new units or loss semantics.
                    num=prediction['parameters']['resource'];y=batch['supervision']['resource']
                    lookup={int(q):j for j,q in enumerate(num['payload_candidate'].tolist())}
                    candidates=[c for r in host['routing']['samples'] for c in r['resource_candidates']]
                    selected=prediction['presence_logits'].detach().cpu()>=0.0
                    for j,q in enumerate(y['candidate'].tolist()):
                        if q not in lookup or not selected[q] or not y['supervised'][j]:continue
                        a,b=y['payload_ptr'][j:j+2].tolist();v,w=num['payload_ptr'][lookup[q]:lookup[q]+2].tolist()
                        valid=y['payload_valid'][a:b]
                        payload_by_action.update_numeric(candidates[q].action.action_id+'.payload',
                            num['payload'][v:w][valid],y['payload'][a:b][valid],'1',normalized=True)
                    log['number_of_batches']+=1
                    save('test_access_log.json',log)
                    if log['number_of_batches']%25==0:
                        print(f"Progress: {log['TEST_sample_count']}/18536; batches={log['number_of_batches']}/290; elapsed={time.perf_counter()-started:.1f}s",flush=True)
                    del prediction,batch,host,rows,contexts,sample
        finally:
            g1._load_episode=original_load
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        after=model_hash(model)
        (A/'parameter_hash_after.txt').write_text(after+'\n')
        assert after==before, 'INTEGRITY FAILURE: model state changed'
        assert log['TEST_sample_count']==metrics.samples==18536
        assert log['TEST_episode_count']==len(episode_metrics)==12
        assert log['number_of_batches']==log['model_forward_calls']==290
        assert log['number_of_valid_model_predictions']==18536
        assert sum(errors.values())==sum(patterns.values())==18536
        global_result=metrics.compute();loss=aggregate.compute()
        save('test_global_metrics.json',{**global_result,'total_loss':loss['total_loss'],
            'wall_time_s':elapsed,'samples_per_s':18536/elapsed,
            'mean_FP_per_sample':metrics.fp/18536,'mean_FN_per_sample':metrics.fn/18536,
            'membership_semantics':'Same raw candidate identity/logit>=0 VAL contract; values/feasibility excluded from exact-set',
            'parameter_hash_before':before,'parameter_hash_after':after,'parameter_hash_equal':True})
        save('test_loss_families.json',loss)
        actions=dict(global_result['per_action'])
        required=('wheel_velocity','pan_velocity','tilt_velocity','pan_target','tilt_target','pan_delta','tilt_delta','align_faces','dock','undock','gravity_settle')
        for action in required:
            actions.setdefault(action,{'support':0,'tp':0,'fp':0,'fn':0,**_scores(0,0,0)})
        save('test_per_action_metrics.json',{'actions':actions,
            'normalized_numeric_by_action':payload_by_action.compute().get('normalized_numeric',{}),
            'support_zero_note':'No positive TEST evidence; empty-set score convention is inherited from existing metrics, not a learned-action success claim',
            'payload_support_unit':'valid scalar components on matched resource actions; same selection and masking as global VAL numeric metrics'})
        save('test_task_breakdown.json',{k:v.compute() for k,v in sorted(task_metrics.items())})
        save('test_episode_breakdown.json',{k:v.compute() for k,v in sorted(episode_metrics.items())})
        save('test_cardinality_breakdown.json',{str(k):{**v,'exact_set_rate':v['exact']/v['samples'],
            'mean_FP':v['fp']/v['samples'],'mean_FN':v['fn']/v['samples']} for k,v in sorted(cards.items())})
        save('test_set_error_distribution.json',{str(k) if k<4 else '4+':{'count':errors[k],'fraction':errors[k]/18536} for k in range(5)})
        save('test_fp_fn_breakdown.json',{k:{'count':patterns[k],'fraction':patterns[k]/18536} for k in ('exact','FP_only','FN_only','both')})
        val=rt['summary']['validation'];vm=val['metrics']
        comparison={k:{'VAL_best':vm[k],'TEST':global_result[k],'TEST_minus_VAL':global_result[k]-vm[k]}
            for k in ('precision','recall','f1','exact_match','predicted_cardinality')}
        comparison['total_loss']={'VAL_best':val['total_loss'],'TEST':loss['total_loss'],'TEST_minus_VAL':loss['total_loss']-val['total_loss']}
        save('val_vs_test.json',{'checkpoint_epoch':5,'metrics':comparison,
            'INITIAL_RANDOM_VAL_REFERENCE':rt['history'][0]['validation'],
            'not_a_random_TEST_baseline':True})
        log.update(status='COMPLETE',official_TEST_evaluation_consumed=True,end_timestamp=now(),wall_time_s=elapsed,
            parameter_hash_before=before,parameter_hash_after=after,parameter_hash_equal=True,
            result_frozen=True,no_threshold_tuning=True,no_checkpoint_comparison=True,no_retraining=True)
        save('test_access_log.json',log)
        print(json.dumps({'status':'COMPLETE','samples':18536,'episodes':12,'batches':290,
            'loss':loss['total_loss'],'precision':global_result['precision'],'recall':global_result['recall'],
            'F1':global_result['f1'],'exact_set':global_result['exact_match'],'parameter_hash_equal':True,
            'wall_time_s':elapsed},indent=2),flush=True)
    except BaseException as exc:
        log.update(status='FAILED_STOP_NO_AUTOMATIC_RETRY',end_timestamp=now(),error=repr(exc),
            traceback=traceback.format_exc(),official_TEST_evaluation_consumed=bool(log['number_of_valid_model_predictions']))
        save('test_access_log.json',log)
        raise

if __name__=='__main__':main()
