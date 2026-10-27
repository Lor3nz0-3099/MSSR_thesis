"""Opt-in acceleration preserves the default optimizer/aggregation contract."""
import copy
import torch

from drl.IL.tests.bc_v1.test_resource_heads_v3 import example,threads
from drl.IL.bc_v1.training import train_step,build_optimizer
from drl.IL.bc_v1.losses import load_loss_config
from drl.IL.bc_v1.losses_bulk import compute_bc_loss_bulk
from drl.IL.tests.bc_v1.g2_fixtures import ROOT


def test_opt_in_bulk_step_matches_default_loss_and_optimizer_update():
    batch,model,_,_=example()
    other=copy.deepcopy(model)
    cfg=load_loss_config(ROOT/'configs/bc_v1/loss_v1.json')
    first,second=build_optimizer(model),build_optimizer(other)
    torch.manual_seed(711)
    old=train_step(model,batch,first,cfg,audit=True)
    torch.manual_seed(711)
    new=train_step(other,batch,second,cfg,audit=True,loss_fn=compute_bc_loss_bulk)
    assert old['families'].keys()==new['families'].keys()
    for name in old['families']:
        assert old['families'][name]['support']==new['families'][name]['support']
        torch.testing.assert_close(torch.tensor(old['families'][name]['mean']),
                                   torch.tensor(new['families'][name]['mean']),rtol=2e-6,atol=2e-7)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,other.state_dict()[name],rtol=2e-5,atol=2e-6)


def test_epoch_uses_borrowed_rows_and_injected_loss_without_changing_sampler():
    from drl.IL.bc_v1.trainer import train_epoch
    from drl.IL.bc_v1.sampling import JointActionBatchSampler,TrainRowIndex
    # A synthetic v3 sample (no real split IO).
    from drl.IL.tests.bc_v1.test_resource_heads_v3 import example
    batch,model,_,_=example()
    class Data:
        split='train'
        reads=0
        def __getitem__(self,index):raise AssertionError('owned path must not be used')
        def get_for_collate(self,index):self.reads+=1;return index
    data=Data();calls=[]
    def loss(*args):calls.append(1);return compute_bc_loss_bulk(*args)
    index=TrainRowIndex({'a':(0,1)}, {}, {'split_name':'train'})
    sampler=JointActionBatchSampler(index,batch_size=1,ordinary_quota=1,structural_quota=0)
    cfg=load_loss_config(ROOT/'configs/bc_v1/loss_v1.json')
    result=train_epoch(model,data,sampler,build_optimizer(model),cfg,device='cpu',epoch=1,
        max_steps=1,loss_fn=loss,collate=lambda rows:batch)
    assert calls==[1] and data.reads==1 and sampler.position==1 and result['steps']==1


def test_validation_injects_loss_and_borrowed_reader_preserving_aggregation():
    from drl.IL.bc_v1.trainer import evaluate_validation
    batch,model,_,_=example()
    class Data:
        split='validation'
        def __len__(self):return 2
        def __getitem__(self,index):raise AssertionError('owned path must not be used')
        def get_for_collate(self,index):return index
    class Metrics:
        def reset(self):self.n=0
        def update(self,prediction,batch):self.n+=1
        def compute(self):return {'batches':self.n}
    calls=[]
    def loss(*args):calls.append(1);return compute_bc_loss_bulk(*args)
    cfg=load_loss_config(ROOT/'configs/bc_v1/loss_v1.json')
    result=evaluate_validation(model,Data(),cfg,Metrics(),device='cpu',batch_size=1,
        max_batches=1,loss_fn=loss,collate=lambda rows:batch)
    assert calls==[1] and result['metrics']=={'batches':1} and result['samples']==1
