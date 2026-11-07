"""Cached-vs-uncached current-state preprocessing; no Isaac launch."""
from dataclasses import replace
import copy
import torch
import pytest
from drl.MARL.tests.test_bc_v4_runtime_observation import make_bridge,ROOT
from drl.IL.tests.bc_v1 import test_live_observation as live_test
from drl.IL.tests.bc_v1 import test_candidates as fixtures
from drl.IL.bc_v1.candidates import enumerate_candidates
from drl.IL.bc_v1 import bc_v1_contract as c


def equal_tree(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_tree(a[k],b[k])
    else:assert a==b


def test_actor_only_does_not_prepare_privileged_tensorization(monkeypatch):
    from drl.MARL import bc_v4_runtime_observation as runtime
    bridge=make_bridge();raw=live_test.snapshot(n=8)
    reference=bridge.prepare_snapshot(raw,task_context=c.TaskContext('navigation'))
    def forbidden(*a,**kw):pytest.fail('Privileged tensorization invoked in actor-only mode')
    monkeypatch.setattr(runtime,'tensorize_policy_sample',forbidden)
    result=bridge.prepare_snapshot(raw,task_context=c.TaskContext('navigation'),include_privileged=False)
    assert result.privileged_graph is None
    equal_tree(result.actor_graph,reference.actor_graph)
    assert result.routing_snapshot==reference.routing_snapshot


def test_catalog_cache_oracle_and_every_admission_change():
    from drl.IL.bc_v1.runtime_static_cache import RuntimeStaticCache
    spec=fixtures.catalog(ports=2);schema=fixtures.schema();graph=fixtures.graph(modules=3,ports=2)
    cache=RuntimeStaticCache()
    first=cache.enumerate(graph,spec,schema)
    assert first==enumerate_candidates(graph,spec,schema)
    moved=replace(graph,modules=tuple(replace(m,position=(2.,3.,4.),linear_velocity=(7.,0.,0.)) for m in graph.modules))
    assert cache.enumerate(moved,spec,schema) is first
    a,b=graph.connectors[0].ref,graph.connectors[2].ref
    attached=replace(graph,connectors=tuple(replace(p,status='connected',connected_to=b if p.ref==a else a) if p.ref in (a,b) else p for p in graph.connectors),connections=(c.ConnectionState(a,b,'current_connection'),))
    variants=[attached,replace(graph,connectors=tuple(replace(p,status='unknown') for p in graph.connectors)),
        replace(graph,modules=tuple(reversed(graph.modules))),replace(graph,resources=tuple(reversed(graph.resources)))]
    for current in variants:
        result=cache.enumerate(current,spec,schema)
        assert result==enumerate_candidates(current,spec,schema)
    changed=replace(spec,actions=tuple(replace(a,enabled=False) if a.action_id=='dock' else a for a in spec.actions))
    assert cache.enumerate(graph,changed,schema)==enumerate_candidates(graph,changed,schema)
    cache.set_epoch(2)
    after=cache.enumerate(graph,spec,schema)
    assert after==first and after is not first
    assert cache.metrics()['catalog_hits']>=1


def test_cached_layout_is_exact_fresh_and_dynamic_features_are_not_cached():
    from drl.IL.bc_v1.runtime_static_cache import RuntimeStaticCache
    bridge=make_bridge();cache=RuntimeStaticCache();raw=live_test.snapshot(n=8)
    def prepare(value,cached):return bridge.prepare_snapshot(value,task_context=c.TaskContext('navigation'),include_privileged=False,**({'static_cache':cache} if cached else {}))
    reference=prepare(raw,False);first=prepare(raw,True);second=prepare(raw,True)
    equal_tree(first.actor_graph,reference.actor_graph);equal_tree(second.actor_graph,reference.actor_graph)
    for key in reference.routing_snapshot:assert first.routing_snapshot[key]==reference.routing_snapshot[key]
    assert first.routing_snapshot['runtime_candidate_layout_token']==second.routing_snapshot['runtime_candidate_layout_token']
    first.actor_graph['structural']['argument_domain_index'].fill_(-999)
    fresh=prepare(raw,True);equal_tree(fresh.actor_graph,reference.actor_graph)
    raw=copy.deepcopy(raw);raw['state']['modules'][0]['linear_velocity'][0]=5.
    raw['state']['modules'][0]['twist']['linear'][0]=5.
    raw['graph']['nodes'][0]['attributes']=copy.deepcopy(raw['state']['modules'][0])
    changed=prepare(raw,True);oracle=prepare(raw,False)
    equal_tree(changed.actor_graph,oracle.actor_graph)
    assert not torch.equal(changed.actor_graph['module_x'],reference.actor_graph['module_x'])
    assert cache.metrics()['layout_hits']>=2 and cache.metrics()['layout_builds']==1


def test_native_actor_only_provider_explicit_guard_keeps_training_contract():
    from drl.MARL.tests.test_native_policy_bridge import make_bridge as native_bridge
    bridge,env,_,_=native_bridge();source=bridge.provider.observation_source
    original=source.prepare_env
    source.prepare_env=lambda i:replace(original(i),privileged_graph=None)
    bridge.provider.reset()
    with pytest.raises(ValueError,match='privileged'):bridge.provider.current_action_contexts()
    bridge.provider.actor_only=True;bridge.provider.reset()
    assert all(context.critic_graph is None for context in bridge.provider.current_action_contexts())
    with pytest.raises(RuntimeError,match='actor-only'):bridge.provider.current_critic_graphs()


def test_cache_is_bounded_and_sibling_epoch_has_no_effect_on_other_cache():
    from drl.IL.bc_v1.runtime_static_cache import RuntimeStaticCache
    a,b=RuntimeStaticCache(max_layouts=2),RuntimeStaticCache(max_layouts=2)
    spec,schema=fixtures.catalog(ports=2),fixtures.schema();graph=fixtures.graph(modules=3,ports=2)
    original=b.enumerate(graph,spec,schema);a.enumerate(graph,spec,schema);a.set_epoch(2)
    assert b.enumerate(graph,spec,schema) is original
    for n in (2,3,4,5):a.enumerate(fixtures.graph(modules=n,ports=2),spec,schema)
    assert a.metrics()['catalog_entries']==2


def test_real_checkpoint_logits_identical_with_actor_only_cached_layout():
    from drl.IL.bc_v1.runtime_static_cache import RuntimeStaticCache
    from drl.MARL.c1_structural_physics_diagnostic import build_frozen_actor,configuration,parse_args
    bridge=make_bridge();raw=live_test.snapshot(n=8);cache=RuntimeStaticCache()
    original=bridge.prepare_snapshot(raw,task_context=c.TaskContext('navigation'))
    optimized=bridge.prepare_snapshot(raw,task_context=c.TaskContext('navigation'),include_privileged=False,static_cache=cache)
    actor=build_frozen_actor(bridge,configuration(parse_args(['--preflight','--device','cpu'])))
    with torch.no_grad():
        a=actor.policy(original.actor_graph);b=actor.policy(optimized.actor_graph)
    equal_tree(a,b)


def test_resource_mode_availability_and_peer_only_change_invalidate_catalog():
    from drl.IL.bc_v1.runtime_static_cache import RuntimeStaticCache
    cache=RuntimeStaticCache();spec=fixtures.catalog(ports=2);schema=fixtures.schema();graph=fixtures.graph(modules=3,ports=2)
    base=cache.enumerate(graph,spec,schema)
    disabled=replace(spec,actions=tuple(replace(a,enabled=False) if a.action_id=='pace' else a for a in spec.actions))
    result=cache.enumerate(graph,disabled,schema)
    assert result==enumerate_candidates(graph,disabled,schema)
    assert len(result.resource_candidates)<len(base.resource_candidates)
    # Resource mode catalog changes are part of the spec fingerprint as well.
    typ=spec.module_types[0];resource=typ.resources[0]
    changed=replace(spec,module_types=(replace(typ,resources=(replace(resource,modes=resource.modes[:1]),)),))
    assert cache.enumerate(graph,changed,schema)==enumerate_candidates(graph,changed,schema)
    peer=replace(graph,connectors=tuple(replace(p,status='connected',connected_to=graph.connectors[2].ref) if i==0 else p for i,p in enumerate(graph.connectors)))
    assert cache.enumerate(peer,spec,schema)==enumerate_candidates(peer,spec,schema)
    assert cache.metrics()['catalog_builds']==4
