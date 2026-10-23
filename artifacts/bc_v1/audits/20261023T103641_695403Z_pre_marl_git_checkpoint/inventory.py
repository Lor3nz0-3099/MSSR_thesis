"""Packaging inventory: paths/stats only for dataset payloads, never deserialization."""
from pathlib import Path
import subprocess,json,csv,collections
A=Path(__file__).resolve().parent;R=Path('/home/lorenzo/MSSR_thesis')
RUN=R/'artifacts/bc_v1/runs/20261022T232532_854802073Z_207309_bc_v4_actor_visible_r1m_scratch_final'
TEST=R/'artifacts/bc_v1/audits/20261023T073725_980736Z_bc_v4_official_one_shot_test_evaluation'
HELPER=R/'artifacts/bc_v1/manual_helpers/20261022T194614_079486Z_bc_v4_reader_final_full_training'
LOCAL_POST={'commit_sha.txt','commit_show.txt','git_status_after.txt','post_commit_integrity.json','staged_diff_check_initial.txt'}
UNRELATED={
 'mssr_ws/src/mssr_expert/mssr_expert/execution/parallel_assembly_executor.py':'Unconnected policy-alignment executor extension; separate existing runtime work; no caller from final actor/training path',
 'mssr_ws/src/mssr_expert/test/test_parallel_assembly_executor.py':'Tests of that separate executor extension; preserve together outside release',
 'mssr_ws/src/mssr_expert/mssr_expert/nodes/smores_parallel_self_assembly_node.py':'Separate recorder same-stamp event persistence fix',
 'scripts/dataset_tools/build_expert_v1.py':'Separate assembly-only dataset import support',
 'scripts/smores_ep/run_button_expert_to_ik.py':'Separate expert button controller/reposition changes',
 'scripts/smores_ep/src/smores_ep/control/button_ik_math.py':'Separate MM8 expert reposition planning',
 'scripts/smores_ep/tests/test_button_ik_source_contract.py':'Separate expert button source-contract tests'}
RUNTIME={
 'scripts/smores_ep/run_bc_live_smoke.py','scripts/smores_ep/src/smores_ep/isaac/command_router.py',
 'scripts/smores_ep/src/smores_ep/isaac/dynamic_stage.py','scripts/smores_ep/src/smores_ep/primitives/bc_runtime.py',
 'scripts/smores_ep/tests/test_bc_live_smoke.py','scripts/smores_ep/tests/test_bc_runtime.py',
 'scripts/smores_ep/tests/test_command_router.py'}
def git(*a):return subprocess.check_output(['git',*a],cwd=R)
def rel(p):return str(p.relative_to(R))
def metadata():
 names=['normalization/f2_actor_visible_r1m_v1/normalizers.json','normalization/f2_actor_visible_r1m_v1/fit_context.json',
 'sidecar/g2_v4_actor_visible_r1m/manifest.json','sidecar/g2_v4_actor_visible_r1m/SUCCESS.json',
 'cache/g1_v1_optimized/manifest.json','splits/f1_v1/split.json','splits/f1_v1/episodes.json',
 'splits/f1_v1/audit.json','splits/f1_v1/report.md']
 return {str(Path('artifacts/bc_v1')/n) for n in names}
def category(path):
 p=R/path
 if path in UNRELATED:
  return ('AMBIGUOUS_DO_NOT_TOUCH' if '/parallel_assembly_executor.py' in path or '/test_parallel_assembly_executor.py' in path else 'UNRELATED_DO_NOT_TOUCH',UNRELATED[path])
 if path.startswith(('drl/IL/bc_v1/','drl/IL/tests/bc_v1/','configs/bc_v1/')):
  if '__pycache__' in p.parts or p.suffix=='.pyc':return 'KEEP_LOCAL_IGNORE','Generated Python cache; no deletion needed'
  return 'OFFICIAL_COMMIT','Final BC implementation/config/regressions including prerequisite legacy contracts'
 if path in RUNTIME:return 'OFFICIAL_COMMIT','Existing BC checkpoint/lease/coactuation runtime support and associated CPU tests'
 if path in ('.gitignore','docs/bc_v4_actor_visible_release.md'):return 'OFFICIAL_COMMIT','Release artifact policy/documentation'
 if p.parent==RUN:return ('KEEP_LOCAL_IGNORE','Final last.pt must remain local') if p.name=='last.pt' else ('OFFICIAL_COMMIT','Official selected checkpoint or lightweight final run metadata')
 if path=='artifacts/bc_v1/manual_runs/'+RUN.name+'.log':return 'OFFICIAL_COMMIT','Official final training record, about12KiB'
 if p.is_relative_to(TEST):return 'OFFICIAL_COMMIT','Complete frozen official one-shot TEST evidence, byte-identical'
 if p.parent==HELPER:return 'OFFICIAL_COMMIT','Final pinned training helper preserved as provenance, not rerun'
 if path in metadata():return 'OFFICIAL_COMMIT','Frozen actor F2 or upstream source/F1 identity metadata; no payload'
 if p.is_relative_to(A):
  if p.parent!=A or p.name in LOCAL_POST or p.suffix=='.tmp':return 'KEEP_LOCAL_IGNORE','Generated test working files or intrinsically post-commit evidence; not a release payload'
  return 'OFFICIAL_COMMIT','Pre-MARL packaging audit, no recursive self-commit SHA'
 if path.startswith('artifacts/bc_v1/'):return 'KEEP_LOCAL_IGNORE','Superseded diagnostic/scientific history or generated data; preserve on disk'
 return 'UNRELATED_DO_NOT_TOUCH','Existing non-BC change/untracked material; preserve untouched'
def main():
 tracked=set(git('ls-files','-z').decode().split('\0'))-{''}
 modified=(set(git('diff','--name-only','-z').decode().split('\0')) | set(git('diff','--cached','--name-only','-z').decode().split('\0')))-{''}
 untracked=set(git('ls-files','--others','--exclude-standard','-z').decode().split('\0'))-{''}
 paths=set(modified)|untracked|{'.gitignore'}
 for folder in ['drl/IL/bc_v1','drl/IL/tests/bc_v1','configs/bc_v1','artifacts/bc_v1']:
  paths.update(rel(p) for p in (R/folder).rglob('*') if p.is_file())
 if (R/'.gitattributes').exists():paths.add('.gitattributes')
 paths.update(p for p in tracked if p.startswith('docs/') and 'bc' in p.lower())
 payload='\0'.join(sorted(paths))+'\0'
 ignored=subprocess.run(['git','check-ignore','--stdin','-z'],input=payload.encode(),cwd=R,capture_output=True)
 assert ignored.returncode in (0,1)
 ignored=set(ignored.stdout.decode().split('\0'))-{''}
 rows=[];selected=[]
 for path in sorted(paths):
  p=R/path
  if not p.is_file():continue
  c,reason=category(path);new=path not in tracked
  decision='STAGE_EXPLICIT' if c=='OFFICIAL_COMMIT' and (new or path in modified) else 'ALREADY_TRACKED' if c=='OFFICIAL_COMMIT' else 'PRESERVE_UNSTAGED'
  rows.append(dict(path=path,size=p.stat().st_size,tracked=path in tracked,modified=path in modified,
      ignored=path in ignored,category=c,decision=decision,reason=reason))
  if decision=='STAGE_EXPLICIT':selected.append(path)
 for name,subset in [('file_classification.tsv',rows),('artifact_inventory.tsv',[x for x in rows if x['path'].startswith('artifacts/bc_v1/')])]:
  with (A/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(subset)
 (A/'selected_paths.json').write_text(json.dumps(selected,indent=2)+'\n')
 (A/'cleanup_actions.tsv').write_text('action\tpath\treason\nIGNORE_POLICY\tartifacts/bc_v1/\tGenerated and historical files local-only; exact release exceptions force-added\n')
 old=json.loads((A/'snapshot.json').read_text())
 counts={'categories':dict(collections.Counter(x['category'] for x in rows)),
  'selected_paths':len(selected),'historical_untracked_from_git':0,'temporary_deleted':0,
  'old_run_directories':len([p for base in ['runs','failed_runs','frozen_runs'] for p in (R/'artifacts/bc_v1'/base).iterdir() if p.is_dir() and p!=RUN]),
  'historical_pt_files_including_caches_and_synthetic_fixtures':sum(Path(x).suffix=='.pt' and str(RUN.relative_to(R))+'/' not in x for x in old['artifact_stats']),
  'old_checkpoint_files':sum(Path(x).suffix=='.pt' and '/cache/' not in x and '/pytest' not in x
      and str(RUN.relative_to(R))+'/' not in x for x in old['artifact_stats']),
  'old_audit_directories':len([p for p in (R/'artifacts/bc_v1/audits').iterdir() if p.is_dir() and p not in (A,TEST)])}
 (A/'inventory_summary.json').write_text(json.dumps(counts,indent=2)+'\n')
 print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
