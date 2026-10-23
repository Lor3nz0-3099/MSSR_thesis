import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import pytest
from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize
from drl.IL.tests.bc_v1.test_resource_decisions import contract
from drl.IL.bc_v1.sidecar import SidecarWriter,SidecarDataset,_hash,assert_exact
from drl.IL.bc_v1.structural_decisions import add_no_structural_supervision


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.resource_prepare'), 'Resource TRAIN/VAL converter missing'
    from drl.IL.bc_v1 import resource_prepare
    return resource_prepare


def source(tmp_path):
    from drl.IL.tests.bc_v1.test_sidecar import identity
    s,kw=fixture();one=tensorize(s,kw);ids=identity();inventory={k:[] for k in ('train','validation','test')}
    for split in ('train','validation'):inventory[split]=[dict(episode_id=split,samples=1,transition_indices_fingerprint=_hash([0]))]
    w=SidecarWriter(tmp_path/'old',ids,inventory)
    for split in ('train','validation'):
        sample={**one,'routing':{**one['routing'],'audit':replace(s.audit,episode_id=split)}}
        w.add(add_no_structural_supervision(sample),split,0)
    d=w.finalize();return tmp_path/'old',d,kw


def test_tiny_roundtrip_preserves_inputs_structure_checksums_and_seals_test(tmp_path):
    a=api();root,doc,kw=source(tmp_path);calls=[]
    def factory(path,split,**args):
        assert split in ('train','validation');calls.append(split);return SidecarDataset(path,split,**args)
    c=contract(kw)
    from drl.IL.bc_v1.resource_decisions import add_resource_decisions
    report=a.convert_resource_sidecar(root,tmp_path/'v3',c,label_provider=lambda row:add_resource_decisions(row,c,authoritative_resources={1}),dataset_factory=factory)
    assert report['TEST_payload_access']==0 and report['sample_counts']=={'train':1,'validation':1}
    assert set(calls)=={'train','validation'}
    new=json.loads((tmp_path/'v3/manifest.json').read_text())
    assert new['identity']['resource_contract_fingerprint']==c.fingerprint and not new['inventory']['test']
    assert new['resource_decision_schema']=='mssr.bc_v1.resource_decisions.v1'
    for split in ('train','validation'):
        with SidecarDataset(root,split,expected_identity=doc['identity']) as old,SidecarDataset(tmp_path/'v3',split,expected_identity=new['identity']) as out:
            x,y=old[0],out[0];assert_exact(y['supervision']['structural'],x['supervision']['structural'])
            for key in x['inputs']:
                if key!='resource_candidates':assert_exact(y['inputs'][key],x['inputs'][key])
    assert a.verify_resource_sidecar(tmp_path/'v3',c)['TEST_payload_access']==0
    with pytest.raises(ValueError):a.convert_resource_sidecar(root,tmp_path/'v3',c,label_provider=lambda r:r)
    # A v3 identity is part of ordinary stale-cache validation, never ignored.
    bad={**new['identity'],'resource_contract_fingerprint':'foreign'}
    with pytest.raises(ValueError):SidecarDataset(tmp_path/'v3','train',expected_identity=bad)


def test_manual_helper_help_dry_run_and_preflight_never_build(tmp_path):
    import subprocess,os
    from tempfile import TemporaryDirectory
    a=api();root=Path(__file__).resolve().parents[4]
    config=root/'configs/bc_v1/resource_prepare_v3.json'
    assert config.is_file(),'Durable manual build configuration not prepared'
    cfg=json.loads(config.read_text())
    helper=root/cfg['manual_helper']
    if not helper.is_file():pytest.skip('Local durable manual helper is an external artifact')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
    for args in (['--help'],['--invalid']):
        result=subprocess.run(['bash',helper,*args],cwd=root,env=env,text=True,capture_output=True,timeout=30)
        assert result.returncode==(2 if args==['--invalid'] else 0),result.stdout+result.stderr
    # The durable helper is a frozen historical build receipt. Exercise current
    # admission with a separate local receipt and an owned, absent output.
    with TemporaryDirectory(prefix='test-resource-preflight-',dir=root/'artifacts/bc_v1') as owned:
        cfg={**cfg,'output':str(Path(owned)/'v3'),
             'code_pins':{name:a.file_sha(root/name) for name in cfg['code_pins']}}
        local=tmp_path/'prepare.json';local.write_text(json.dumps(cfg))
        _,source,output,c,doc=a.preflight(local)
        result=subprocess.run([os.sys.executable,'-B','-m','drl.IL.bc_v1.resource_prepare','--config',str(local)],
                              cwd=root,env=env,text=True,capture_output=True,timeout=30)
        assert result.returncode==0,result.stdout+result.stderr
        report=json.loads(result.stdout)
        assert report['mode']=='DRY_RUN' and report['TEST_payload_access']==0
        assert report['output']==str(output) and report['source']==str(source)
        assert not output.exists() and not output.with_name(output.name+'.building').exists()
        bad={**cfg,'resource_contract_fingerprint':'wrong'};p=tmp_path/'bad.json';p.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='Resource contract pin mismatch'):a.preflight(p)
        bad={**cfg,'code_pins':{'drl/IL/bc_v1/resource_decisions.py':'0'*64}};p.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='Build code identity mismatch'):a.preflight(p)
        output.mkdir()
        with pytest.raises(ValueError,match='Output already exists'):a.main(['--config',str(local)])


def test_builder_refuses_payload_supervision_on_inactive_outcome(tmp_path):
    a=api();root,doc,kw=source(tmp_path);c=contract(kw)
    from drl.IL.bc_v1.resource_decisions import add_resource_decisions
    def invalid(old):
        row=add_resource_decisions(old,c);row['supervision']['resource']['decision_candidate'][0]=0
        return row
    with pytest.raises(ValueError,match='Payload supervision'):
        a.convert_resource_sidecar(root,tmp_path/'bad',c,label_provider=invalid)


def test_preflight_rejects_changed_build_code_before_build(tmp_path):
    a=api();cfg=json.loads(a.CONFIG.read_text());cfg['code_pins']={'drl/IL/bc_v1/resource_prepare.py':'foreign'}
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='Build code identity'):a.preflight(path)


def test_preflight_rejects_unsupported_configuration_schema(tmp_path):
    a=api();cfg=json.loads(a.CONFIG.read_text());cfg['schema_version']='unsupported.v99'
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='prepare configuration schema'):a.preflight(path)
