"""CPU-only regression tests for integration added during the interrupted Phase D."""
from pathlib import Path

import pytest


def test_stage_classification_matches_expert_parallel_waves():
    from drl.MARL.c1_phase_d_capture import stage_for_count
    assert [stage_for_count(k) for k in range(8)] == [
        'S0','S1','S1','S2','S2','S3','S3',None]


def test_native_docking_snapshot_api_is_exposed_without_isaac_import():
    from smores_ep.isaac.docking import IsaacDockingManager
    for name in ('export_connections','validate_connection_states','restore_connections'):
        assert callable(getattr(IsaacDockingManager,name))


def test_snapshot_runner_guard_accepts_valid_explicit_bank_and_weights(tmp_path):
    from drl.MARL.isaaclab_c1_runner import build_parser,configure_run_arguments
    bank=tmp_path/'bank.json';bank.write_text('{}')
    args=build_parser().parse_args([
        '--train','--c1-staged-training','--curriculum','c1',
        '--initialization-mode','assembled_state',
        '--reward-config','configs/marl/c1_guided_phase_c_v1.json',
        '--c1-probe-config','configs/marl/c1_guided_phase_c_v1.json',
        '--snapshot-bank',str(bank),
        '--snapshot-stage-weights','{"S0":0.25,"S1":0.2,"S2":0.25,"S3":0.3}',
        '--episode-seconds','120','--num-envs','4','--total-transitions','32',
        '--early-stopping-patience','0','--checkpoint-every-transitions','64',
        '--output',str(tmp_path/'new'),
    ])
    assert configure_run_arguments(args).snapshot_stage_weights == {
        'S0':.25,'S1':.2,'S2':.25,'S3':.3}
    assert args.initialization_mode == 'assembled_state'


def test_snapshot_runner_rejects_invalid_weights_before_simulator(tmp_path):
    from drl.MARL.isaaclab_c1_runner import build_parser,configure_run_arguments
    bank=tmp_path/'bank.json';bank.write_text('{}')
    for weights in ('{"S0":1}', '{"S0":0,"S1":0,"S2":0,"S3":0}',
                    '{"S0":-1,"S1":0,"S2":1,"S3":1}', 'broken'):
        args=build_parser().parse_args([
            '--train','--c1-staged-training','--curriculum','c1',
            '--initialization-mode','assembled_state',
            '--reward-config','configs/marl/c1_guided_phase_c_v1.json',
            '--c1-probe-config','configs/marl/c1_guided_phase_c_v1.json',
            '--snapshot-bank',str(bank),'--snapshot-stage-weights',weights,
            '--episode-seconds','120','--num-envs','4','--total-transitions','32',
        '--early-stopping-patience','0','--checkpoint-every-transitions','64',
            '--output',str(tmp_path/'new'),
        ])
        with pytest.raises(ValueError,match='weights|JSON'):
            configure_run_arguments(args)


def test_docking_connection_preflight_rejects_duplicate_and_wrong_body_without_usd():
    from smores_ep.docking.model import DockingFace
    from smores_ep.isaac.docking import IsaacDockingManager
    manager=object.__new__(IsaacDockingManager)
    manager._module_roots={'m0':'/World/envs/env_0/SMORES_0',
                           'm1':'/World/envs/env_0/SMORES_1'}
    manager._faces={owner:(DockingFace(module_id=owner,face_name=port,
        rigid_body_path=f'{root}/body',frame_path=f'{root}/body/{port}'),)
        for owner,root,port in [('m0',manager._module_roots['m0'],'RIGHT'),
                                ('m1',manager._module_roots['m1'],'LEFT')]}
    edge=dict(first_owner='m0',first_port='RIGHT',second_owner='m1',second_port='LEFT',
        first_body_relative_path='body',second_body_relative_path='body',
        local_pos0=[0.,0.,0.],local_rot0_xyzw=[0.,0.,0.,1.],
        local_pos1=[0.,0.,0.],local_rot1_xyzw=[0.,0.,0.,1.],
        collision_enabled=False,exclude_from_articulation=True)
    manager.validate_connection_states([edge])
    with pytest.raises(ValueError,match='(?i)duplicate'):
        manager.validate_connection_states([edge,edge])
    with pytest.raises(ValueError,match='body path'):
        manager.validate_connection_states([{**edge,'first_body_relative_path':'wrong'}])
