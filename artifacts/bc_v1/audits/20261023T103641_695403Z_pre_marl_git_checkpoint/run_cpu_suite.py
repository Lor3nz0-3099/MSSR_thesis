"""The previously certified CPU selection; output/evidence confined to this audit."""
from pathlib import Path
import os,sys,json
A=Path(__file__).resolve().parent;R=Path('/home/lorenzo/MSSR_thesis')
sys.path.insert(0,str(R))
import audit_guard
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
import pytest
events=[];deselected=[]
class Capture:
 def pytest_deselected(self,items):deselected.extend(x.nodeid for x in items)
 def pytest_runtest_logreport(self,report):
  if report.failed:events.append({'node':report.nodeid,'when':report.when,'detail':str(report.longrepr)})
 def pytest_terminal_summary(self,terminalreporter,exitstatus,config):
  doc={'counts':{k:len(terminalreporter.stats.get(k,[])) for k in ('passed','failed','skipped','deselected','error')},
   'exit_code':int(exitstatus),'failure_events':events,'TEST_payload_access':audit_guard.COUNTS['TEST_payload_access'],
   'denied_attempts':audit_guard.COUNTS['denied_attempts'],
   'skipped_details':[{'node':r.nodeid,'reason':str(r.longrepr)} for r in terminalreporter.stats.get('skipped',[])],
   'deselected_details':deselected}
  (A/'test_results.json').write_text(json.dumps(doc,indent=2)+'\n')
if __name__=='__main__':
 args=['drl/IL/tests/bc_v1','-q','-p','no:cacheprovider','--basetemp',str(A/'pytest_work'),
  '--deselect=drl/IL/tests/bc_v1/test_dataset_counts.py::test_g1_materialized_cache_has_frozen_v6_split_counts']
 assert not (A/'pytest_work').exists()
 (A/'test_command.json').write_text(json.dumps({'python':sys.executable,'pytest_argv':args,
  'env':{k:os.environ.get(k) for k in ('PYTEST_DISABLE_PLUGIN_AUTOLOAD','MSSR_BC_V1_SKIP_REAL_CENSUS','MSSR_K2_REAL_TRAIN_SMOKE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
  'same_previous_selection':True,'only_basetemp_and_audit_evidence_location_changed':True},indent=2)+'\n')
 sys.exit(pytest.main(args,plugins=[Capture()]))
