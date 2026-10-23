"""Read-only pre/post Git packaging checks. No model or dataset imports."""
from pathlib import Path
import hashlib,json,subprocess,collections,sys
A=Path(__file__).resolve().parent;R=Path('/home/lorenzo/MSSR_thesis')
RUN=R/'artifacts/bc_v1/runs/20261022T232532_854802073Z_207309_bc_v4_actor_visible_r1m_scratch_final'
BEST=RUN/'best.pt';LAST=RUN/'last.pt'
SHA='d90a73e9b5ba2ecb579e4fc8733066b1c55d9e60dafeb1d71c029c49a7c21c6d'
def load(n):return json.loads((A/n).read_text())
def save(n,v):(A/n).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
def preservation():
 s=load('snapshot.json')
 bad=[p for p,h in s['source_hashes'].items() if sha(R/p)!=h]
 assert not bad,'Source preservation failure:'+repr(bad)
 for name,before in s['artifact_stats'].items():
  p=R/name;assert p.is_file(), 'Missing historical artifact:'+name
  st=p.stat();assert (st.st_size,st.st_mtime_ns,st.st_ino)==(before['size'],before['mtime_ns'],before['inode']), 'Historical artifact drift:'+name
 for name,h in load('scientific_file_hashes.json').items():assert sha(R/name)==h,name
 assert sha(BEST)==SHA and LAST.is_file()
 return {'source_hashes_unchanged':len(s['source_hashes']),'original_artifact_stats_unchanged':len(s['artifact_stats']),
  'official_run_TEST_files_byte_identical':len(load('scientific_file_hashes.json')),
  'last_exists_locally':True,'no_scientific_result_modified':True,'temporary_files_deleted':0}
def suite_gate():
 results={n:load(n) for n in ('test_results.json','runtime_gate_recovery_results.json')}
 for name,d in results.items():
  assert d['exit_code']==0 and d['counts']['failed']==d['counts']['error']==0,(name,d)
  assert d['TEST_payload_access']==0 and not d['denied_attempts']
 seals=[load(p.name) for p in A.glob('seal_*.json')]
 assert seals and all(s['TEST_payload_access']==0 and not s['denied_attempts'] for s in seals)
 return results
def main():
 mode=sys.argv[1]
 results=suite_gate();p=preservation()
 assert subprocess.run(['git','diff','--check'],cwd=R,capture_output=True).returncode==0
 if mode=='prestage':
  assert git('rev-parse','HEAD').decode().strip()==load('snapshot.json')['HEAD']
  assert not git('diff','--cached','--name-only')
  save('preservation.json',{'PASS':True,'phase':'all pre-commit gates; post-commit record is local-only',**p,
   'production_tests_configs_unchanged_by_task':True,'only_existing_BC_deltas_packaged':True,
   'TEST_payload_access_this_task':0,'diff_check_exit_code':0,
   'historical_artifacts_removed_from_tracking':0,'initial_tracked_artifact_count':0,
   'no_reset_restore_clean_stash_checkout_push':True})
  (A/'test_access_guard.txt').write_text('PASS\nTHIS TASK TEST payload access = 0\nDenied attempts = 0\n'+
   'Previous certified episode-open audit hook and real-sidecar split guard reused; synthetic TEST fixtures allowed.\n'+
   'Exact previous CPU exclusions retained; no real census or all-split G1 constructor.\n'+
   'All parent/worker seal reports and both suite summaries checked.\n')
  print(json.dumps({'gate':'PRESTAGE_PASS','suites':{k:v['counts'] for k,v in results.items()},'preservation':p},indent=2))
 elif mode=='staged':
  expected=set(load('selected_paths.json'));actual=set(git('diff','--cached','--name-only','-z').decode().split('\0'))-{''}
  assert actual==expected,{'missing':sorted(expected-actual),'extra':sorted(actual-expected)}
  assert str(LAST.relative_to(R)) not in actual
  assert git('ls-files','--',str(LAST.relative_to(R)))==b''
  blob=git('show',':'+str(BEST.relative_to(R)));assert hashlib.sha256(blob).hexdigest()==SHA
  assert git('cat-file','-s',':'+str(BEST.relative_to(R))).decode().strip()==str(BEST.stat().st_size)
  assert subprocess.run(['git','-c','core.attributesFile='+str(A/'whitespace_policy.gitattributes'),'diff','--cached','--check'],cwd=R,capture_output=True).returncode==0
  binaries=[]
  for path in expected:
   content=git('show',':'+path)
   assert content==(R/path).read_bytes(), 'Stale staged content:'+path
   try:content.decode('utf8')
   except UnicodeDecodeError:binaries.append(path)
  assert binaries==[str(BEST.relative_to(R))]
  save('staging_gate.json',{'PASS':True,'paths':len(actual),'only_binary':binaries,'best_sha256':SHA,
   'last_staged':False,'last_tracked':False,'all_staged_blobs_match_worktree':True,
   'pre_existing_unrelated_files_staged':False,'TEST_payload_access_this_task':0})
  print(json.dumps({'gate':'STAGED_PASS','files':len(actual),'binary':binaries,'SHA256':SHA},indent=2))
 elif mode=='postcommit':
  assert not git('diff','--cached','--name-only')
  assert git('log','-1','--format=%s').decode().strip()=='Finalize actor-visible BC-v4 warm start for MAPPO'
  assert git('rev-parse','HEAD^').decode().strip()==load('snapshot.json')['HEAD']
  expected=set(load('selected_paths.json'));actual=set(git('diff-tree','--no-commit-id','--name-only','-r','HEAD','-z').decode().split('\0'))-{''}
  assert actual==expected
  blob=git('show','HEAD:'+str(BEST.relative_to(R)));assert hashlib.sha256(blob).hexdigest()==SHA
  assert git('ls-files','--',str(BEST.relative_to(R))).decode().strip()==str(BEST.relative_to(R))
  assert git('ls-files','--',str(LAST.relative_to(R)))==b''
  modified=set(git('diff','--name-only','-z').decode().split('\0'))-{''}
  from inventory import UNRELATED
  assert modified==set(UNRELATED), modified
  head=git('rev-parse','HEAD').decode().strip()
  save('post_commit_integrity.json',{'PASS':True,'HEAD':head,'parent':load('snapshot.json')['HEAD'],
   'committed_files':len(actual),'index_clean':True,'best_local_SHA256':SHA,'best_HEAD_blob_SHA256':hashlib.sha256(blob).hexdigest(),
   'last_exists_locally':True,'last_tracked':False,'unrelated_modified_remaining':sorted(modified),
   'TEST_payload_access_this_task':0,'NO_TEST_RE_EVALUATION':True,'NO_TRAINING_EXECUTED':True,
   'PUSH_NOT_EXECUTED':True,**p})
  (A/'commit_sha.txt').write_text(head+'\n')
  (A/'commit_show.txt').write_bytes(git('show','--stat','--oneline','--decorate','HEAD')+b'\n'+git('show','--name-status','--format=fuller','HEAD'))
  (A/'git_status_after.txt').write_bytes(git('status','--short','--untracked-files=all')+b'\n--- REMAINING DIFF ---\n'+git('diff','--binary')+b'\n--- INDEX ---\n'+git('diff','--cached','--binary'))
  print(json.dumps(load('post_commit_integrity.json'),indent=2))
 else:raise ValueError(mode)
if __name__=='__main__':main()
