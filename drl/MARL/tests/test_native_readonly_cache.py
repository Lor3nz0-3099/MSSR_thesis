"""Actual-read snapshots per completed physics tick; targets always remain live."""
from types import SimpleNamespace
import numpy as np
import pytest
import warp as wp
from drl.MARL.tests.test_isaaclab_articulation import TensorView,ROOT
from drl.MARL.isaaclab_articulation import NativeArticulationAccess


class CountingView(TensorView):
    def __init__(self,root=ROOT):
        super().__init__(root);self.reads={}
        self.put('dof_positions',[[1.,2.,3.]])
        self.put('dof_velocities',[[.1,.2,.3]])
        self.put('root_transforms',[[0.,0.,1.,0.,0.,0.,1.]])
        self.put('root_velocities',[[0.]*6])
        self.put('link_transforms',[[[0.,0.,1.,0.,0.,0.,1.]]])
    def __getattr__(self,name):
        if name.startswith('get_') and name!='get_link_transforms':
            def get():
                field=name[4:];self.reads[field]=self.reads.get(field,0)+1
                return self.arrays[field]
            return get
        return super().__getattr__(name)


def access(view,clock):
    sim=SimpleNamespace(create_articulation_view=lambda path:view,update_articulations_kinematic=lambda:None)
    return NativeArticulationAccess(sim,view.prim_paths[0],state_clock=clock)


def test_one_read_per_stamp_readonly_value_and_retained_snapshot():
    stamp=[0];view=CountingView();a=access(view,lambda:stamp[0])
    first=a.get_dof_positions().numpy()
    for _ in range(8):np.testing.assert_array_equal(a.view.get_dof_positions().numpy(),[[1.,2.,3.]])
    assert view.reads['dof_positions']==1
    assert not first.flags.writeable
    view.put('dof_positions',[[4.,5.,6.]]);stamp[0]+=1
    np.testing.assert_array_equal(a.get_dof_positions().numpy(),[[4.,5.,6.]])
    np.testing.assert_array_equal(first,[[1.,2.,3.]])
    assert view.reads['dof_positions']==2


def test_same_policy_counter_pre_and_post_physics_have_distinct_snapshot():
    # DirectMARLEnv increments policy-owned counter BEFORE applying actions.
    # The SimulationContext count advances in sim.step after _apply_action.
    physics=[7];policy_counter=8;view=CountingView();a=access(view,lambda:physics[0])
    pre=a.get_dof_positions().numpy()
    view.put('dof_positions',[[8.,9.,10.]]);physics[0]=8
    assert policy_counter==8
    np.testing.assert_array_equal(a.get_dof_positions().numpy(),[[8.,9.,10.]])
    np.testing.assert_array_equal(pre,[[1.,2.,3.]])


@pytest.mark.parametrize('field',['dof_positions','dof_velocities','root_transforms','root_velocities'])
def test_explicit_state_mutation_invalidates_at_same_stamp(field):
    view=CountingView();a=access(view,lambda:(0,1))
    original=getattr(a.view,'get_'+field)().numpy().copy()
    changed=wp.array(original+1.,dtype=wp.float32,device='cpu');indices=wp.array([0],dtype=wp.uint32,device='cpu')
    getattr(a.view,'set_'+field)(changed,indices)
    np.testing.assert_array_equal(getattr(a.view,'get_'+field)().numpy(),original+1.)


def test_reset_pose_invalidates_same_counter_and_sibling_stays_cached():
    stamp=[0];va,vb=CountingView(),CountingView(root=ROOT+'_sibling')
    a,b=access(va,lambda:stamp[0]),access(vb,lambda:stamp[0])
    a.get_dof_positions().numpy();b.get_dof_positions().numpy()
    a.reset_pose((2.,3.,4.),0.)
    np.testing.assert_array_equal(a.get_dof_positions().numpy(),[[0.,0.,0.]])
    np.testing.assert_array_equal(b.get_dof_positions().numpy(),[[1.,2.,3.]])
    assert vb.reads['dof_positions']==1


def test_targets_are_never_cached_and_target_writes_do_not_stale_actuals():
    view=CountingView();a=access(view,lambda:0)
    a.get_dof_positions().numpy();a.get_dof_position_targets().numpy()
    a.set_dof_position_targets([4.],dof_indices=[0])
    np.testing.assert_array_equal(a.get_dof_position_targets().numpy(),[[4.,0.,0.]])
    a.set_dof_position_targets([9.],dof_indices=[1])
    np.testing.assert_array_equal(a.get_dof_position_targets().numpy(),[[4.,9.,0.]])
    np.testing.assert_array_equal(a.get_dof_positions().numpy(),[[1.,2.,3.]])
    assert view.reads['dof_positions']==1


def test_environment_wires_actual_simulation_counter_not_preincremented_env_counter():
    import ast
    from pathlib import Path
    tree=ast.parse((Path(__file__).resolve().parents[1]/'isaaclab_c1_env.py').read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_ensure_state_readers')
    calls=[n for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='NativeArticulationAccess']
    assert calls and any(k.arg=='state_clock' for k in calls[0].keywords)
    assert any(isinstance(n,ast.Attribute) and n.attr=='get_physics_step_count' for n in ast.walk(calls[0]))
