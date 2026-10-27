"""Safe manual launch contract; tests never run an epoch or open split payloads."""
import importlib.util
import json
import pytest


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.train_resource_optimized'), 'Optimized v3 entrypoint missing'
    from drl.IL.bc_v1 import train_resource_optimized
    return train_resource_optimized


def test_default_is_preflight_only_and_never_initializes_cuda_or_opens_payload(monkeypatch,capsys):
    a=api()
    monkeypatch.setattr(a,'preflight',lambda: {'TEST_payload_access':0,'samples':{'train':78302,'validation':22762}})
    monkeypatch.setattr(a,'run',lambda *_a,**_kw:pytest.fail('dry-run must not launch'))
    a.main([])
    result=json.loads(capsys.readouterr().out)
    assert result['mode']=='PREFLIGHT_ONLY' and result['TEST_payload_access']==0


def test_manual_launch_requires_new_output_and_explicit_run(monkeypatch,tmp_path):
    a=api();calls=[]
    monkeypatch.setattr(a,'preflight',lambda:{'TEST_payload_access':0})
    monkeypatch.setattr(a,'run',lambda *args,**kw:calls.append((args,kw)))
    with pytest.raises(SystemExit):a.main(['--run'])
    a.main(['--run','--output',str(tmp_path/'new'),'--epochs','1'])
    assert len(calls)==1
    with pytest.raises(ValueError,match='exists|overwrite'):a.main(['--run','--output',str(tmp_path)])
    with pytest.raises(SystemExit):a.main(['--run','--output',str(tmp_path/'new'),'--epochs','101'])


def test_manual_orchestration_publishes_finite_status_and_checkpoint_metadata(monkeypatch,tmp_path):
    import torch
    from contextlib import nullcontext
    a=api()
    monkeypatch.setattr(a,'preflight',lambda:{'TEST_payload_access':0})
    monkeypatch.setattr(a,'load_sources',lambda:{'kwargs':{'spec':None,'feature_schema':None,'normalizers':None}})
    monkeypatch.setattr(a.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(a.torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(a.torch,'set_num_threads',lambda *_:None)
    monkeypatch.setattr(a.torch,'set_num_interop_threads',lambda *_:None)
    original_read=a.read
    monkeypatch.setattr(a,'read',lambda p:{'identity':{}} if p==a.G2/'manifest.json' else original_read(p))
    monkeypatch.setattr(a,'build_scratch_actor',lambda *_a,**_kw:(torch.nn.Linear(1,1),None,{'frozen':'identity'}))
    class Data:
        def __init__(self,*_a,**_kw):pass
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def train_index(self):return None
        def stats(self):return {}
    class Sampler:
        def __len__(self):return 2
        def state_dict(self):return {'position':2}
    monkeypatch.setattr(a,'TrainingDataset',Data)
    monkeypatch.setattr(a,'JointActionBatchSampler',lambda *_a,**_kw:Sampler())
    monkeypatch.setattr(a,'BCMetrics',lambda *_:None)
    monkeypatch.setattr(a,'train_epoch',lambda *_a,**_kw:{'steps':2,'global_step':2,'wall_seconds':1.})
    monkeypatch.setattr(a,'evaluate_validation',lambda *_a,**_kw:{'samples':22762,'total_loss':.5})
    saved=[]
    def save(path,*args,**kw):
        # JSON/checkpoint metadata must never persist an infinite initial VAL.
        json.dumps(kw['selection'],allow_nan=False);saved.append((path.name,kw['summary']))
    monkeypatch.setattr(a,'save_checkpoint',save)
    report={'cache_budgets_bytes':dict.fromkeys(('train_dynamic','train_template','validation_dynamic','validation_template'),0),
        'seed':20261006,'learning_rate':.0003,'weight_decay':.0001}
    output=tmp_path/'new'
    a.run(output,epochs=1,cpu_threads=1,report=report)
    assert json.loads((output/'status.json').read_text())['phase']=='COMPLETED'
    assert json.loads((output/'SUCCESS.json').read_text())['TEST_payload_access']==0
    assert [n for n,_ in saved]==['best_checkpoint.pt','last_checkpoint.pt']
    assert saved[0][1]['epoch']==1 and saved[0][1]['global_step']==2
