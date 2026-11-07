"""Exercise native training preparation without importing Isaac or stepping physics."""
from copy import deepcopy
from types import SimpleNamespace
import torch

from drl.IL.bc_v1.bc_v1_contract import TaskContext, ObservationBundle
from drl.MARL.isaaclab_c1_tasks import NativeC1TaskHooks
from drl.MARL.tests.test_bc_v4_runtime_observation import make_bridge
from drl.IL.tests.bc_v1.test_live_observation import snapshot
from drl.MARL.tests.test_c1_runtime_static_cache import equal_tree


def test_training_cache_preserves_both_views_and_reset_terminal_state(monkeypatch):
    import drl.MARL.c1_runner as runner
    monkeypatch.setattr(runner,'physical_task_view',lambda *args:(TaskContext('navigation'),ObservationBundle()))
    raw=snapshot(n=8)
    raw['state']['course']={}
    env=SimpleNamespace(observation_bridge=make_bridge(),_prepared_cache={},generation=[1,1],
        domain_by_env=['navigation','navigation'],task_context_config={},_marl_pre=[True,True],
        handoffs=[SimpleNamespace(marl_active=True),SimpleNamespace(marl_active=True)],
        _post_raw=[deepcopy(raw),deepcopy(raw)],cached_raw_snapshot=lambda i:raw,
        _c1_runtime_static_caches={})
    prepare=lambda i=0,**kw:NativeC1TaskHooks.prepare_env(env,i,**kw)
    reference=prepare();env._prepared_cache.clear()
    # The training boundary explicitly opts in; ordinary viewers retain their own cache.
    from drl.MARL.isaaclab_c1_training import enable_static_preparation_cache
    enable_static_preparation_cache(env)
    first=prepare();env._prepared_cache.clear();warm=prepare()
    equal_tree(reference.actor_graph,warm.actor_graph)
    equal_tree(reference.privileged_graph,warm.privileged_graph)
    cache=env._c1_runtime_static_caches[0]
    assert cache.metrics()['catalog_hits']>=1 and cache.metrics()['layout_hits']>=1
    builds=cache.metrics()['layout_builds']
    for _ in range(10):
        env._prepared_cache.clear()
        prepare(include_candidates_override=False)
    assert cache.metrics()['layout_builds']==builds
    # Dynamic physical features must not be cached along with catalog routing.
    raw=deepcopy(raw)
    raw['state']['modules'][0]['linear_velocity'][0]=5.
    raw['state']['modules'][0]['twist']['linear'][0]=5.
    raw['graph']['nodes'][0]['attributes']=deepcopy(raw['state']['modules'][0])
    env._prepared_cache.clear();changed=prepare()
    env._c1_static_preparation_enabled=False;env._prepared_cache.clear();oracle=prepare()
    equal_tree(changed.actor_graph,oracle.actor_graph);equal_tree(changed.privileged_graph,oracle.privileged_graph)
    assert not torch.equal(first.actor_graph['module_x'],changed.actor_graph['module_x'])
    env._c1_static_preparation_enabled=True;env._prepared_cache.clear()
    prepare(1);sibling=env._c1_runtime_static_caches[1]
    sibling_stats=sibling.metrics().copy()
    # Autoreset changes generation while terminal post-step raw data still belongs to the old step.
    env.generation[0]=2
    terminal=prepare(post_step=True,include_candidates_override=False)
    assert terminal.privileged_graph is not None
    assert len(terminal.actor_graph['structural']['initiator'])==0
    env._c1_static_preparation_enabled=False;env._prepared_cache.clear()
    terminal_oracle=prepare(post_step=True,include_candidates_override=False)
    equal_tree(terminal.actor_graph,terminal_oracle.actor_graph)
    equal_tree(terminal.privileged_graph,terminal_oracle.privileged_graph)
    assert cache.metrics()['epoch_invalidations']==1
    assert sibling.metrics()==sibling_stats
    # Cached templates must never become aliases of policy/rollout storage tensors.
    env._c1_static_preparation_enabled=True;env._prepared_cache.clear()
    changed.actor_graph['module_x'].fill_(-999)
    fresh=prepare();equal_tree(fresh.actor_graph,oracle.actor_graph)
    # A reset current state with another topology must not reuse terminal layout/features.
    raw=snapshot(n=8,attached=True);raw['state']['course']={}
    env._prepared_cache.clear();reset_current=prepare()
    env._c1_static_preparation_enabled=False;env._prepared_cache.clear()
    reset_oracle=prepare()
    equal_tree(reset_current.actor_graph,reset_oracle.actor_graph)
    equal_tree(reset_current.privileged_graph,reset_oracle.privileged_graph)
    assert len(reset_current.routing_snapshot['structural_bindings'])!=len(first.routing_snapshot['structural_bindings'])
    assert sibling.metrics()==sibling_stats
