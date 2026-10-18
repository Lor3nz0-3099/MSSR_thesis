"""F2: train-only preprocessing at the immutable E2 feature boundary."""

from dataclasses import replace
import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.feature_schema import (
    FeatureField, FeatureRow, FeatureView, extract_features, load_feature_schema,
)
from drl.IL.bc_v1.features import PhysicalFrame, PhysicalFrameConfig, transform_inputs
from drl.IL.bc_v1.split import build_split_manifest, make_episode_split, read_split_manifest
from drl.IL.tests.bc_v1 import fixtures as f

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = load_feature_schema(ROOT/'configs/bc_v1/features.json')
FRAME = PhysicalFrame((0.,0.,0.), (1.,0.,0.,0.,1.,0.,0.,0.,1.), 'world', 'world', True)


def api():
    name = 'drl.IL.bc_v1.normalization'
    assert importlib.util.find_spec(name) is not None, 'Missing F2 normalization engine'
    return importlib.import_module(name)


def spec(bounds=None):
    base = f.robot_spec()
    resource = c.ResourceSpec('joint', (c.CoordinateSpec('rotation','rad',(-2.,2.)),), (
        c.ResourceModeSpec('rate',1,('rad/s',),codec='continuous'),
        c.ResourceModeSpec('position',1,('rad',),kind='discrete',codec='continuous'),
        c.ResourceModeSpec('delta',1,('rad',),kind='discrete',codec='continuous'),
        c.ResourceModeSpec('category',1,('1',),kind='discrete',codec='categorical',categories=('a','b')),
    ))
    actions = tuple(c.ActionSpec(mode.mode_id,mode.kind,'resource',(
        c.ParameterSpec('value','scalar',units=mode.units[0],limits=bounds if mode.mode_id=='rate' else None,
                        module_type_id='unit',resource_id='joint'),
    )) for mode in resource.modes if mode.codec!='categorical')
    skill = c.ActionSpec('recover','skill','modules',(
        c.ParameterSpec('duration','scalar',units='s'),
        c.ParameterSpec('uv','vector',size=2,units='1',limits=(-1.,1.)),
        c.ParameterSpec('frozen','scalar',units='m',learnable=False),
        c.ParameterSpec('category','category',categories=('a','b')),
    ))
    return replace(base,module_types=(replace(base.module_types[0],resources=(resource,)),),actions=actions+(skill,))


def policy():
    value = api().load_normalization_policy(ROOT/'configs/bc_v1/normalization.json',SCHEMA)
    value = copy.deepcopy(value)
    value['resource_payloads'] = [
        {'module_type_id':'unit','resource_id':'joint','mode_id':name,
         'components':[{'parameter':'value','component':0}]} for name in ('rate','position','delta')
    ]
    return value


def manifest():
    rows=[{'episode_id':str(i),'lineage_id':str(i),'sha256':f'{i+1:064x}',
           'bc_valid_transitions':10,'tag':'all'} for i in range(4)]
    cfg={'schema_version':'mssr.bc_v1.strata.v1','eligible_count_field':'bc_valid_transitions',
         'strata':[{'name':'all','match':{'tag':'all'},'total':4,
                    'quotas':{'train':2,'validation':1,'test':1}}]}
    splits=make_episode_split(rows,cfg)
    return build_split_manifest(splits,rows,cfg,20261006,
        {'dataset_name':'fixture','schema_version':'fixture.v1','manifest_sha256':'a'*64})


def view(value=2.,valid=True):
    inputs=c.PolicyInputs(f.state_graph(1,1),c.ObservationBundle(),c.TaskContext())
    result=extract_features(inputs,transform_inputs(inputs,PhysicalFrameConfig()),spec(),SCHEMA)
    row=result.modules[0]
    row=replace(row,values=(value,)+row.values[1:],validity=(valid,)+row.validity[1:])
    return replace(result,modules=(row,))


def sample(eid,index=0,value=2.,target=2.,valid=True):
    supervision=c.SupervisionBundle(resource_targets=(c.ResourceTarget(c.ResourceRef(c.ModuleRef('unit-0'),'joint'),'rate',(target,)),))
    return api().NormalizationSample(eid,index,view(value,valid),supervision)


def streams(split=None,**kwargs):
    split=split or manifest()
    result={eid:[sample(eid,value=v,target=v,**kwargs)]
            for eid,v in zip(split['splits']['train'],(2.,4.))}
    result.update({eid:[sample(eid,value=1e100,target=1e100)]
                   for part in ('validation','test') for eid in split['splits'][part]})
    return result


def fit(data=None,robot=None,p=None,split=None):
    split=split or manifest()
    return api().fit_normalizers(data if data is not None else streams(split),split,robot or spec(),SCHEMA,p or policy(),
                                expected_split_fingerprint=split['manifest_fingerprint'])


def feature_key(stats, family='module',field='relative_position',component=0,unit='m',**context):
    return api().feature_key(family,field,component,unit,**context)


def target_key(mode='rate',unit='rad/s',component=0):
    return api().resource_target_key('unit','joint',mode,component,unit)


def test_population_mean_std_only_train():
    stats=fit(); scaler=stats.get(feature_key(stats))
    assert (scaler.offset,scaler.scale,scaler.count)==(3.,1.,2)
    assert stats.get(target_key()).offset==3.


@pytest.mark.parametrize('part',['validation','test'])
def test_altering_holdout_cannot_change_stats_or_fingerprint(part):
    split=manifest(); a=streams(split); b=streams(split)
    for eid in split['splits'][part]:
        b[eid]=[sample(eid,value=-1e200,target=-1e200)]
    assert fit(a)==fit(b)
    assert api().normalizer_fingerprint(fit(a))==api().normalizer_fingerprint(fit(b))


def test_fitter_never_opens_holdout_factories():
    split=manifest(); data=streams(split)
    def forbidden():
        raise AssertionError('Holdout accessed during fit')
    for part in ('validation','test'):
        for eid in split['splits'][part]:data[eid]=forbidden
    assert fit(data).get(target_key()).offset==3.


def test_order_invariance_including_adversarial_cancellation():
    split=manifest(); data=streams(split); eid=split['splits']['train'][0]
    data[eid]=[sample(eid,i,x,x) for i,x in enumerate((1e16,1.,-1e16,3.))]
    original=fit(data)
    data={eid:list(reversed(rows)) for eid,rows in reversed(list(data.items()))}
    assert fit(data)==original
    assert original.get(target_key()).offset==pytest.approx(1.6)


def test_stable_variance_for_large_offset_small_spread():
    split=manifest(); data={eid:[sample(eid,value=v,target=v)] for eid,v in zip(split['splits']['train'],(1e12+2,1e12+4))}
    scaler=fit(data).get(target_key())
    assert scaler.offset==1e12+3 and scaler.scale==1.


def test_invalid_missing_and_padding_ignored_and_preserved():
    split=manifest(); data=streams(split)
    eid=split['splits']['train'][1]; data[eid]=[sample(eid,value=float('nan'),valid=False)]
    stats=fit(data);scaler=stats.get(feature_key(stats))
    assert scaler.count==1 and scaler.offset==2. and scaler.scale==1.
    source=data[eid][0].features
    result=api().normalize_features(source,stats,SCHEMA,policy(),spec())
    assert math.isnan(result.modules[0].values[0])
    assert result.modules[0].validity==source.modules[0].validity
    # E2 packs padding as invalid components: zero is retained, not shifted.
    padded=replace(source,modules=(replace(source.modules[0],values=(0.,)+source.modules[0].values[1:]),))
    assert api().normalize_features(padded,stats,SCHEMA,policy(),spec()).modules[0].values[0]==0.


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_valid_nonfinite_rejected(bad):
    split=manifest(); data=streams(split); eid=split['splits']['train'][0]
    data[eid]=[sample(eid,value=bad)]
    with pytest.raises(ValueError,match='finite'):fit(data)


def test_zero_variance_and_no_support_identity_diagnostic():
    stats=fit()
    zero=stats.get(feature_key(stats,component=1))
    assert zero.scale==1. and zero.count==2
    missing=stats.get(feature_key(stats,field='linear_velocity',unit='m/s'))
    assert missing.offset==0. and missing.scale==1. and missing.no_train_support
    absent=stats.get(target_key('delta','rad'))
    assert absent.offset==0. and absent.scale==1. and absent.no_train_support


def test_modes_and_units_not_fused():
    split=manifest();data=streams(split)
    for eid,rows in data.items():
        s=rows[0]
        targets=s.supervision.resource_targets+(c.ResourceTarget(c.ResourceRef(c.ModuleRef('unit-0'),'joint'),'position',(20.,)),)
        data[eid]=[replace(s,supervision=c.SupervisionBundle(resource_targets=targets))]
    stats=fit(data)
    assert stats.get(target_key()).offset==3.
    assert stats.get(target_key('position','rad')).offset==20.
    assert stats.get(target_key('delta','rad')).no_train_support


def test_bounds_come_only_from_declared_parameter_limits_no_clipping():
    stats=fit(robot=spec((-2.,6.)))
    scaler=stats.get(target_key())
    assert scaler.offset==2. and scaler.scale==4. and scaler.strategy=='physical_bounds'
    target=c.ResourceTarget(c.ResourceRef(c.ModuleRef('unit-0'),'joint'),'rate',(10.,))
    result=api().normalize_target(target,stats,module_type_id='unit')
    assert result.payload==(2.,)
    assert api().denormalize_values(result.payload,target_key(),stats)==(10.,)


def test_extreme_train_values_do_not_change_declared_bounds():
    split=manifest();data=streams(split)
    for eid in split['splits']['train']:data[eid]=[sample(eid,target=1e100)]
    assert fit(data,robot=spec((-2.,6.))).get(target_key()).scale==4.


@pytest.mark.parametrize('bounds',[(2.,2.),(4.,2.)])
def test_degenerate_or_incoherent_bounds_fail_explicitly(bounds):
    with pytest.raises(ValueError,match='bound|limit'):fit(robot=spec(bounds))


def test_coordinate_limits_do_not_imply_resource_mode_bounds():
    stats=fit()
    assert stats.get(target_key('position','rad')).strategy=='train_standard'
    key=feature_key(stats,'actuator_coordinate','position',unit='rad',module_type_id='unit',resource_id='joint',coordinate_id='rotation')
    assert stats.get(key).strategy=='physical_bounds' and stats.get(key).scale==2.


def test_normalize_feature_view_keeps_all_structure_and_special_channels():
    stats=fit();source=view(2.)
    result=api().normalize_features(source,stats,SCHEMA,policy(),spec())
    assert result.modules[0].values[0]==-1.
    assert result.modules[0].values[3:9]==source.modules[0].values[3:9] # R6
    assert result.frame is source.frame
    for family in ('modules','connectors','resources','coordinates','connections','environment','relations'):
        old,new=getattr(source,family),getattr(result,family)
        assert len(old)==len(new)
        for a,b in zip(old,new):
            assert replace(a,values=b.values)==b
    for a,b in zip(source.connectors,result.connectors):
        assert a.values[3:]==b.values[3:] # normal/tangent/category/capabilities
    assert source==view(2.), 'Input was mutated'


@pytest.mark.parametrize('kind,width,preserved',[
    ('line',6,range(3,6)),('plane',4,range(3)),('heading',3,range(3)),
    ('planar_pose',9,range(3,9)),('angle_constraint',10,range(1,10)),
    ('vector3',3,range(3)),
])
def test_mixed_geometry_preserves_direction_heading_and_basis(kind,width,preserved):
    split=manifest();data=streams(split)
    categories=tuple(SCHEMA['entities']['environment']['fields'][1]['categories'])
    fields=(FeatureField('geometry',width,tuple(SCHEMA['geometry_kinds'][kind]['units']),'robot_centroid'),
            FeatureField('object_type',len(categories),('1',)*len(categories),'intrinsic',categories))
    for j,eid in enumerate(split['splits']['train']):
        env=FeatureRow(('object','c_static','world.shape',0),tuple(float(j+2+i) for i in range(width))+(0.,)*len(categories),(True,)*(width+len(categories)),fields,
                       source_scope='c_static',geometry_kind=kind)
        data[eid]=[replace(data[eid][0],features=replace(data[eid][0].features,environment=(env,)))]
    stats=fit(data);source=data[split['splits']['train'][0]][0].features
    result=api().normalize_features(source,stats,SCHEMA,policy(),spec())
    for i in preserved:assert result.environment[0].values[i]==source.environment[0].values[i]


def test_policy_cannot_standardize_special_codecs_or_direction_components():
    p=policy();p['fields']['module']['orientation_r6']='train'
    with pytest.raises(ValueError,match='preserv|codec'):fit(p=p)
    p=policy();p['geometry']['line']=['train']*6
    with pytest.raises(ValueError,match='preserv|direction|geometry'):fit(p=p)


def test_continuous_targets_only_and_no_invented_missing_values():
    stats=fit(); ref=c.ResourceRef(c.ModuleRef('unit-0'),'joint')
    target=c.ResourceTarget(ref,'rate',(2.,))
    assert api().normalize_target(target,stats,module_type_id='unit')==replace(target,payload=(-1.,))
    for target in (c.ResourceTarget(ref,'rate',(),False),c.ResourceTarget(ref,'category',(1.,))):
        assert api().normalize_target(target,stats,module_type_id='unit') is target
    empty=c.SupervisionBundle()
    data=streams();data={eid:[replace(s,supervision=empty) for s in rows] for eid,rows in data.items()}
    assert fit(data).get(target_key()).no_train_support


def test_structural_arguments_numeric_learnable_only():
    split=manifest();data=streams(split)
    for j,eid in enumerate(split['splits']['train']):
        target=c.StructuralTarget(c.ModuleRef('unit-0'),'recover',parameters=(
            c.ParameterValue('duration',2.+2*j),c.ParameterValue('uv',(.5,-.5)),
            c.ParameterValue('frozen',100.),c.ParameterValue('category','a'),))
        data[eid]=[replace(data[eid][0],supervision=c.SupervisionBundle(structural_targets=(target,)))]
    stats=fit(data);target=data[split['splits']['train'][0]][0].supervision.structural_targets[0]
    result=api().normalize_structural_target(target,stats)
    assert result.parameters[0].value==-1.
    assert result.parameters[1:]==target.parameters[1:]
    assert replace(result,parameters=target.parameters)==target
    assert api().normalize_structural_target(replace(target,supervised=False),stats).parameters==target.parameters


def test_denormalization_numpy_shape_and_roundtrip():
    stats=fit(); physical=np.array([[2.,3.],[4.,5.]])
    normalized=(physical-3.)
    restored=api().denormalize_values(normalized,target_key(),stats)
    assert restored.shape==(2,2) and np.allclose(restored,physical)


def test_multi_component_resource_mode_keeps_independent_units_and_order():
    robot=spec();mt=robot.module_types[0];resource=mt.resources[0]
    resource=replace(resource,modes=resource.modes+(c.ResourceModeSpec('twist',3,('m/s','m/s','rad/s'),codec='vector'),))
    action=c.ActionSpec('twist','continuous','resource',(
        c.ParameterSpec('xy','vector',size=2,units='m/s'),
        c.ParameterSpec('turn','scalar',units='rad/s',limits=(-2.,6.)),))
    robot=replace(robot,module_types=(replace(mt,resources=(resource,)),),actions=robot.actions+(action,))
    p=policy();p['resource_payloads'].append({'module_type_id':'unit','resource_id':'joint','mode_id':'twist',
        'components':[{'parameter':'xy','component':1},{'parameter':'xy','component':0},{'parameter':'turn','component':0}]})
    split=manifest();data=streams(split)
    for i,eid in enumerate(split['splits']['train']):
        target=c.ResourceTarget(c.ResourceRef(c.ModuleRef('unit-0'),'joint'),'twist',(2.+2*i,10.+4*i,2.))
        data[eid]=[replace(data[eid][0],supervision=c.SupervisionBundle(resource_targets=(target,)))]
    stats=fit(data,robot=robot,p=p)
    target=data[split['splits']['train'][0]][0].supervision.resource_targets[0]
    normalized=api().normalize_target(target,stats,module_type_id='unit')
    assert normalized.payload==(-1.,-1.,0.)
    for i,unit in enumerate(('m/s','m/s','rad/s')):
        key=api().resource_target_key('unit','joint','twist',i,unit)
        assert api().denormalize_values(normalized.payload[i],key,stats)==pytest.approx(target.payload[i])


def test_two_coordinate_catalogs_with_same_units_are_not_fused():
    robot=spec();mt=robot.module_types[0];r=mt.resources[0]
    r=replace(r,coordinates=r.coordinates+(c.CoordinateSpec('unbounded','rad'),))
    robot=replace(robot,module_types=(replace(mt,resources=(r,)),))
    split=manifest();data=streams(split)
    template=data[split['splits']['train'][0]][0].features.coordinates[0]
    for j,eid in enumerate(split['splits']['train']):
        fields=template.fields
        other=replace(template,ref=(template.ref[0],'unbounded'),values=(10.+4*j,0.,0.,0.,0.),validity=(True,True,False,False,True),fields=fields)
        data[eid]=[replace(data[eid][0],features=replace(data[eid][0].features,coordinates=data[eid][0].features.coordinates+(other,)))]
    stats=fit(data,robot=robot)
    key=api().feature_key('actuator_coordinate','position',0,'rad',module_type_id='unit',resource_id='joint',coordinate_id='unbounded')
    assert stats.get(key).offset==12. and stats.get(key).scale==2.
    assert stats.get(feature_key(stats,'actuator_coordinate','position',unit='rad',module_type_id='unit',resource_id='joint',coordinate_id='rotation')).scale==2.


def test_implicit_dimensionless_parameter_unit_uses_contract_default():
    robot=spec();mt=robot.module_types[0];r=mt.resources[0]
    r=replace(r,modes=r.modes+(c.ResourceModeSpec('fraction',1,('1',),codec='continuous'),))
    robot=replace(robot,module_types=(replace(mt,resources=(r,)),),actions=robot.actions+(
        c.ActionSpec('fraction','continuous','resource',(c.ParameterSpec('value','scalar'),)),))
    p=policy();p['resource_payloads'].append({'module_type_id':'unit','resource_id':'joint','mode_id':'fraction',
        'components':[{'parameter':'value','component':0}]})
    split=manifest();data=streams(split)
    for i,eid in enumerate(split['splits']['train']):
        target=c.ResourceTarget(c.ResourceRef(c.ModuleRef('unit-0'),'joint'),'fraction',(.2+.2*i,))
        data[eid]=[replace(data[eid][0],supervision=c.SupervisionBundle(resource_targets=(target,)))]
    stats=fit(data,robot=robot,p=p)
    assert stats.get(api().resource_target_key('unit','joint','fraction',0,'1')).offset==pytest.approx(.3)


@pytest.mark.parametrize('corruption',['units','frame','channels'])
def test_claimed_schema_fingerprint_cannot_hide_forged_field_metadata(corruption):
    split=manifest();data=streams(split);eid=split['splits']['train'][0];s=data[eid][0]
    row=s.features.modules[0]
    fields=list(row.fields)
    if corruption=='units':fields[0]=replace(fields[0],units=('rad',)*3)
    if corruption=='frame':fields[0]=replace(fields[0],frame='world')
    if corruption=='channels':fields[0]=replace(fields[0],channels=('id-1','id-2','id-3'))
    data[eid]=[replace(s,features=replace(s.features,modules=(replace(row,fields=tuple(fields)),)))]
    with pytest.raises(ValueError,match='layout|units|frame|schema'):fit(data)


def test_absent_vector_width_in_train_is_explicit_identity_without_new_stats():
    stats=fit();source=view();categories=tuple(SCHEMA['entities']['environment']['fields'][1]['categories'])
    env=FeatureRow(('object','o_dynamic','dimensions',0),(2.,4.,6.)+(0.,)*len(categories),(True,)*(3+len(categories)),
        (FeatureField('geometry',3,('m',)*3,'intrinsic'),FeatureField('object_type',len(categories),('1',)*len(categories),'intrinsic',categories)),
        source_scope='o_dynamic',geometry_kind='intrinsic_vector')
    source=replace(source,environment=(env,));before=api().normalizer_fingerprint(stats)
    assert api().normalize_features(source,stats,SCHEMA,policy(),spec()).environment[0].values==env.values
    key=api().feature_key('environment','geometry',2,'m',geometry_kind='intrinsic_vector')
    assert stats.get(key).no_train_support
    assert api().normalizer_fingerprint(stats)==before


def test_partial_supervision_false_never_contributes_to_target_stats():
    split=manifest();data=streams(split)
    eid=split['splits']['train'][1];s=data[eid][0]
    data[eid]=[replace(s,supervision=c.SupervisionBundle(resource_targets=(replace(s.supervision.resource_targets[0],payload=(1e200,),supervised=False),)))]
    scaler=fit(data).get(target_key())
    assert scaler.offset==2. and scaler.count==1


def test_tiny_population_std_does_not_underflow_to_zero_scale():
    split=manifest();data={eid:[sample(eid,value=x,target=x)] for eid,x in zip(split['splits']['train'],(1e-300,3e-300))}
    scaler=fit(data).get(target_key())
    assert scaler.offset==2e-300 and scaler.scale==pytest.approx(1e-300,rel=1e-14,abs=0.)


def test_bounded_without_train_supervision_is_identity_with_diagnostic():
    data=streams();data={eid:[replace(s,supervision=c.SupervisionBundle()) for s in rows] for eid,rows in data.items()}
    stats=fit(data,robot=spec((-2.,6.)));scaler=stats.get(target_key())
    assert (scaler.offset,scaler.scale,scaler.count,scaler.no_train_support)==(0.,1.,0,True)
    target=c.ResourceTarget(c.ResourceRef(c.ModuleRef('unit-0'),'joint'),'rate',(2.,))
    assert api().normalize_target(target,stats,module_type_id='unit').payload==(2.,)


def test_bounded_coordinate_invalid_in_all_train_is_identity_with_diagnostic():
    data=streams()
    for eid,rows in data.items():
        s=rows[0];row=s.features.coordinates[0]
        row=replace(row,values=(0.,)+row.values[1:],validity=(False,)+row.validity[1:])
        data[eid]=[replace(s,features=replace(s.features,coordinates=(row,)))]
    stats=fit(data)
    key=feature_key(stats,'actuator_coordinate','position',unit='rad',module_type_id='unit',resource_id='joint',coordinate_id='rotation')
    assert (stats.get(key).offset,stats.get(key).scale,stats.get(key).no_train_support)==(0.,1.,True)
    # Later valid inference is unchanged; no support is not filled retrospectively.
    source=view();normalized=api().normalize_features(source,stats,SCHEMA,policy(),spec())
    assert normalized.coordinates[0].values[0]==source.coordinates[0].values[0]


@pytest.mark.parametrize('domain',['feature','resource','argument'])
def test_representable_affine_roundtrip_survives_intermediate_overflow(domain):
    split=manifest();data=streams(split)
    for eid,x in zip(split['splits']['train'],(1e308,1.7e308)):
        s=sample(eid,value=x,target=x)
        argument=c.StructuralTarget(c.ModuleRef('unit-0'),'recover',parameters=(c.ParameterValue('duration',x),))
        data[eid]=[replace(s,supervision=replace(s.supervision,structural_targets=(argument,)))]
    stats=fit(data);physical=-1e308
    if domain=='feature':
        result=api().normalize_features(view(physical),stats,SCHEMA,policy(),spec()).modules[0].values[0]
        key=feature_key(stats)
    elif domain=='resource':
        target=c.ResourceTarget(c.ResourceRef(c.ModuleRef('unit-0'),'joint'),'rate',(physical,))
        result=api().normalize_target(target,stats,module_type_id='unit').payload[0]
        key=target_key()
    else:
        target=c.StructuralTarget(c.ModuleRef('unit-0'),'recover',parameters=(c.ParameterValue('duration',physical),))
        result=api().normalize_structural_target(target,stats).parameters[0].value
        key=api().argument_key('recover','duration',0,'s')
    assert result==pytest.approx(-6.714285714285715)
    assert api().denormalize_values(result,key,stats)==pytest.approx(physical)


def test_numpy_denormalization_overflow_fallback_is_finite_and_warning_free():
    split=manifest();data={eid:[sample(eid,target=x)] for eid,x in zip(split['splits']['train'],(1e308,1.7e308))}
    stats=fit(data)
    with np.errstate(all='raise'):
        result=api().denormalize_values(np.array([[-6.714285714285715,0.]]),target_key(),stats)
    assert result.shape==(1,2) and np.allclose(result,[[-1e308,1.35e308]])


def test_numpy_zero_dimensional_denormalization_preserves_shape():
    split=manifest();data={eid:[sample(eid,target=x)] for eid,x in zip(split['splits']['train'],(1e308,1.7e308))}
    stats=fit(data)
    result=api().denormalize_values(np.array(-6.714285714285715),target_key(),stats)
    assert isinstance(result,np.ndarray) and result.shape==()
    assert result.item()==pytest.approx(-1e308)


def test_artifact_deterministic_and_bound_to_context(tmp_path):
    stats=fit();blob=api().serialize_normalizers(stats)
    assert blob==api().serialize_normalizers(fit())
    value=json.loads(blob)
    assert value['metadata']['train_episode_count']==2
    assert value['metadata']['train_episode_ids']==manifest()['splits']['train']
    assert len(value['metadata']['train_membership_fingerprint'])==64
    path=tmp_path/'normalization.json'
    api().write_normalizers(stats,path,source_root=ROOT/'datasets/expert_v1_canonical_v6')
    assert api().read_normalizers(path,expected_metadata=stats.metadata)==stats
    api().write_normalizers(stats,path,source_root=ROOT/'datasets/expert_v1_canonical_v6')
    for key in ('split_fingerprint','dataset_manifest_sha256','robot_spec_fingerprint','feature_schema_fingerprint','policy_fingerprint'):
        changed=dict(stats.metadata);changed[key]='b'*64
        with pytest.raises(ValueError,match='incompatib'):api().read_normalizers(path,expected_metadata=changed)
    value['scalers'][0]['scale']=2.
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='fingerprint'):api().read_normalizers(path,expected_metadata=stats.metadata)


def test_writer_refuses_source_and_different_artifact(tmp_path):
    stats=fit()
    with pytest.raises(ValueError,match='source'):api().write_normalizers(stats,tmp_path/'x.json',source_root=tmp_path)
    path=tmp_path/'normalization.json';path.write_text('{}')
    with pytest.raises(ValueError,match='overwrite|differ'):api().write_normalizers(stats,path,source_root=tmp_path/'source')


def test_reject_wrong_sample_identity_duplicate_or_missing_train():
    split=manifest();data=streams(split);eid=split['splits']['train'][0]
    data[eid]=[sample('wrong')]
    with pytest.raises(ValueError,match='episode'):fit(data)
    data=streams(split);data[eid]*=2
    with pytest.raises(ValueError,match='duplicate'):fit(data)
    data=streams(split);del data[eid]
    with pytest.raises(ValueError,match='train|missing'):fit(data)


def test_wrong_split_pin_and_schema_rejected():
    split=manifest()
    with pytest.raises(ValueError,match='split'):api().fit_normalizers(streams(split),split,spec(),SCHEMA,policy(),expected_split_fingerprint='b'*64)
    data=streams(split);eid=split['splits']['train'][0]
    data[eid]=[replace(data[eid][0],features=replace(data[eid][0].features,schema_fingerprint='b'*64))]
    with pytest.raises(ValueError,match='schema'):fit(data)


def official_split():
    config=json.loads((ROOT/'configs/bc_v1/training.json').read_text())
    inventory=json.loads((ROOT/config['split']['inventory_path']).read_text())
    return read_split_manifest(ROOT/config['split']['manifest_path'],inventory['episodes'],config['split']['strata_config'],inventory['dataset_identity'],
        expected_seed=config['split']['seed'],expected_fingerprint=config['split']['manifest_fingerprint'])


def test_official_49_train_membership_holdout_poison_cannot_change_fingerprint():
    split=official_split(); data={eid:[sample(eid,value=2.+(i%2)*2)] for i,eid in enumerate(split['splits']['train'])}
    for part in ('validation','test'):
        for eid in split['splits'][part]:data[eid]=lambda: (_ for _ in ()).throw(AssertionError('holdout read'))
    stats=fit(data,split=split)
    assert stats.metadata['train_episode_count']==49
    assert stats.metadata['split_fingerprint']=='aee975a2f8b044ad3f57f331fc659d6cfa91d7c1e493b90b812634fb6364e81b'
    assert stats.metadata['dataset_manifest_sha256']=='31765272b6ecd7506c54b25e214b75d7b1b28691803ddfc286dafcce8b4ec562'
    changed=dict(data)
    for part in ('validation','test'):
        for eid in split['splits'][part]:changed[eid]=[sample(eid,value=1e200)]
    assert api().normalizer_fingerprint(fit(changed,split=split))==api().normalizer_fingerprint(stats)
    assert stats.metadata['complete_train_coverage'] is False


@pytest.fixture(scope='module')
def real_rows(tmp_path_factory):
    from drl.IL.tests.bc_v1.test_policy_inputs import real_policy_rows
    yield from real_policy_rows.__wrapped__(tmp_path_factory)


def test_real_e2_feature_views_smoke_preserve_refs_cardinality_validity(real_rows):
    robot=c.load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json')
    p=api().load_normalization_policy(ROOT/'configs/bc_v1/normalization.json',SCHEMA)
    split=official_split();train=set(split['splits']['train'])
    data={eid:[] for eid in train};views=[]
    for row,_,_,_,inputs in real_rows.values():
        fv=extract_features(inputs,transform_inputs(inputs,PhysicalFrameConfig()),robot,SCHEMA)
        views.append(fv)
        eid=row['provenance']['logical_episode_id'];idx=row['provenance']['canonical_transition_index']
        if eid in train:data[eid].append(api().NormalizationSample(eid,idx,fv,c.SupervisionBundle()))
    stats=api().fit_normalizers(data,split,robot,SCHEMA,p,expected_split_fingerprint=split['manifest_fingerprint'])
    assert stats.metadata['complete_train_coverage'] is False
    for fv in views:
        result=api().normalize_features(fv,stats,SCHEMA,p,robot)
        for family in ('modules','connectors','resources','coordinates','connections','environment','relations'):
            assert len(getattr(fv,family))==len(getattr(result,family))
            for old,new in zip(getattr(fv,family),getattr(result,family)):
                assert replace(old,values=new.values)==new
        for old,new in zip(fv.modules,result.modules):
            for i in range(3):
                if old.validity[i]:
                    assert api().denormalize_values(new.values[i],feature_key(stats,component=i),stats)==pytest.approx(old.values[i])


def test_normalization_core_import_isolation():
    api()
    script='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'numpy','torch','rclpy','isaacsim','isaaclab','omni','smores_ep','mssr_expert'} or fullname in {'drl.IL.bc_v1.smores_adapter','drl.IL.bc_v1.features','drl.IL.bc_v1.policy_inputs','drl.IL.dataset_loader'}:
   raise ImportError(fullname)
sys.meta_path.insert(0,Block())
from drl.IL.bc_v1.normalization import Scaler,NormalizerStats,denormalize_values
stats=NormalizerStats((),(('x',Scaler(3.,2.,2,'train_standard',False)),))
assert denormalize_values((1.,-1.),'x',stats)==(5.,1.)
'''
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
