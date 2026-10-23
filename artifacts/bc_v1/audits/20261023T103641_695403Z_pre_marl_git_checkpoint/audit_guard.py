"""Task-scoped sealing; runtime-only instrumentation, not production code."""
from pathlib import Path
import atexit,json,os,sys
A=Path(__file__).resolve().parent
POLICY=json.loads((A/'sealing_policy.json').read_text());PATHS=POLICY['paths']
ROOT=Path(POLICY['root']);PHASE='integrity_mapping';COUNTS={'TEST_payload_access':0,'denied_attempts':[], 'real_episode_opens':{'train':0,'validation':0},'sidecar_split_opens':[]}
def set_phase(phase):
 global PHASE
 PHASE=phase

def audit(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
 path=Path(os.fsdecode(args[0])).resolve();name=str(path);part=PATHS.get(name)
 if part is None:
  for base in ['datasets/expert_v1_canonical_v6/episodes','artifacts/bc_v1/cache/g1_v1_optimized/episodes']:
   if path.is_relative_to(ROOT/base):part='unassigned'
 if part is not None:
  if part not in ('train','validation') or (part=='validation' and PHASE!='integrity_mapping'):
   COUNTS['denied_attempts'].append({'path':name,'split':part,'phase':PHASE});raise PermissionError('Sealed episode access')
  COUNTS['real_episode_opens'][part]+=1
sys.addaudithook(audit)
from drl.IL.bc_v1.sidecar import SidecarDataset
_original=SidecarDataset.__init__
def sealed_init(self,root,split,*args,**kwargs):
 real=Path(root).resolve() in [ROOT/'artifacts/bc_v1/sidecar/g2_v3_resource_contract',ROOT/'artifacts/bc_v1/sidecar/g2_v4_actor_visible_r1m']
 if real:
  if split not in ('train','validation') or (split=='validation' and PHASE!='integrity_mapping'):
   COUNTS['denied_attempts'].append({'sidecar':str(root),'split':split,'phase':PHASE});raise PermissionError('Sealed sidecar split')
  COUNTS['sidecar_split_opens'].append({'root':str(root),'split':split,'phase':PHASE})
 return _original(self,root,split,*args,**kwargs)
SidecarDataset.__init__=sealed_init

def publish():
 (A/('seal_'+str(os.getpid())+'.json')).write_text(json.dumps(COUNTS,indent=2)+'\n')
atexit.register(publish)
