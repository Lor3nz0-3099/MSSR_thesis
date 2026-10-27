"""Additional existing CPU unit gates for the seven included BC runtime deltas."""
from pathlib import Path
import sys,json
A=Path(__file__).resolve().parent;R=Path('/home/lorenzo/MSSR_thesis')
# Match the documented project source-layout environment; no package installation.
for source in (R, R/'scripts/smores_ep/src', R/'mssr_ws/src/mssr_expert'):
 assert source.is_dir(), source
 sys.path.insert(0,str(source))
import audit_guard
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
import pytest
class Capture:
 def pytest_terminal_summary(self,terminalreporter,exitstatus,config):
  (A/'runtime_gate_recovery_results.json').write_text(json.dumps({
   'counts':{k:len(terminalreporter.stats.get(k,[])) for k in ('passed','failed','skipped','deselected','error')},
   'exit_code':int(exitstatus),'TEST_payload_access':audit_guard.COUNTS['TEST_payload_access'],
   'denied_attempts':audit_guard.COUNTS['denied_attempts'],
   'failures':[str(r.longrepr) for k in ('failed','error') for r in terminalreporter.stats.get(k,[])]},indent=2)+'\n')
if __name__=='__main__':
 args=['scripts/smores_ep/tests/test_bc_live_smoke.py','scripts/smores_ep/tests/test_bc_runtime.py',
  'scripts/smores_ep/tests/test_command_router.py','-q','-p','no:cacheprovider',
  '--basetemp',str(A/'runtime_gate_recovery_work')]
 assert not (A/'runtime_gate_recovery_work').exists()
 (A/'runtime_gate_recovery_command.json').write_text(json.dumps(args,indent=2)+'\n')
 sys.exit(pytest.main(args,plugins=[Capture()]))
