from audit_common import *
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
rt=runtime()
assert not torch.cuda.is_initialized()
digest=sha(BEST)
(A/'checkpoint_sha256.txt').write_text(digest+'\n')
save('checkpoint_identity.json',{'path':str(BEST),'sha256':digest,'size':BEST.stat().st_size,
    'epoch':5,'global_step':6120,'VAL_loss':rt['summary']['validation']['total_loss'],
    'identity':rt['identity'],'summary':rt['summary'],'CPU_load_strict':True,
    'no_last_or_other_checkpoint_loaded':True,'no_cuda_initialized':True})
save('test_evaluation_config.json',{'checkpoint':str(BEST),'checkpoint_sha256':digest,
    'epoch':5,'threshold_logit':0.0,'batch_size':64,'expected_samples':18536,
    'expected_episodes':12,'expected_batches':290,'FP32_eager':True,'TF32':False,
    'AMP':False,'compile':False,'model_eval':True,'optimizer':None,'backward':False,
    'identity':rt['identity'],'actor_visibility':rt['visibility'].document,
    'actor_visibility_fingerprint':rt['visibility'].fingerprint,'h1_config':rt['h1'],'h2_config':rt['h2'],
    'G1_source':str(G1),'G1_fingerprint':rt['physical']['cache_fingerprint'],
    'F2':str(F2),'F2_fingerprint':rt['identity']['source_identity']['f2_fingerprint'],
    'candidate_policy':'Unchanged current G1 candidate tables and production tensorization/order; no label filtering or feasibility pruning',
    'metrics_semantics':'Exact production train_action_set.measure_batch raw membership logit>=0; physical feasibility not applied, same as VAL',
    'test_materialization':'Streaming in-memory tensorization only, no persistent G2 TEST or source modification',
    'task_routing':'Current sample.inputs.policy.context.task_id; audit only, no taxonomy inference',
    'parameterized_payload':'Same matched numeric contract/validity as official VAL, additionally grouped by action_id using existing ActionSetMetrics.update_numeric',
    'frozen_before_TEST':True,'created_UTC':now()})
from drl.IL.bc_v1.dataset import BCDataset
from drl.IL.bc_v1.sidecar import SidecarDataset,assert_exact
from drl.IL.bc_v1.action_set_targets import collate_action_set_samples
data=BCDataset(G1,'train');sample=data[0]
fresh=tensorize(sample,rt)
with SidecarDataset(G2,'train',expected_identity=rt['manifest']['identity']) as source:
    old=source[0]
    fresh_batch=collate_action_set_samples([fresh]);old_batch=collate_action_set_samples([old])
    assert_exact(fresh_batch['inputs'],old_batch['inputs'])
    assert_exact(fresh_batch['supervision'],old_batch['supervision'])
    for key in ('resource_candidates','structural_bindings','resource_targets','structural_targets','physical_supervision','audit'):
        assert_exact(fresh['routing'][key],old['routing'][key])
save('composition_preflight.json',{'PASS':True,'TEST_payload_access':0,'model_predictions':0,
    'TRAIN_probe_samples':1,'TRAIN_episode':sample.audit.episode_id,
    'checks':['strict best epoch5 CPU checkpoint','all expected fingerprints/counts via metadata',
        'frozen actor-visible tensorization exact input/supervision/candidate routing equality with G2 on one TRAIN sample'],
    'parameter_hash':model_hash(rt['model']),'CUDA_initialized':torch.cuda.is_initialized()})
print(json.dumps({'preflight':'PASS','best_epoch':5,'checkpoint_sha':digest,'TEST_count_metadata':18536,
    'TEST_episodes_metadata':12,'TRAIN_composition_parity':'PASS','TEST_payload_access':0},indent=2))
