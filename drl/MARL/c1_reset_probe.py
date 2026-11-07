"""Bounded physical reset diagnostic. No optimizer or C1 reward/probe training.

loose: two manipulation slots, actual BC actions, reset env0 only, sibling proof.
legacy_regression: retain old Expert initial state only to reproduce the recorded
four-domain post-BC failure, then reset all four to policy-controlled loose state.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('loose','legacy_regression','legacy_isolation'),default='loose')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bc-steps',type=int,default=30)
    parser.add_argument('--physics-ticks',type=int,default=56)
    parser.add_argument('--global-seed',type=int,default=20261029)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.bc_steps<1 or args.physics_ticks<50:parser.error('requires BC actions and at least50physics ticks')
    args.output.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    source_files=sorted((repo/'drl/MARL').glob('*.py'))
    source_files+=sorted((repo/'scripts/smores_ep/src/smores_ep/isaac').glob('*.py'))
    hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (args.output/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (args.output/'arguments.json').write_text(json.dumps(vars(args),default=str,indent=2)+'\n')
    application=AppLauncher(args).app
    env=None;result=dict(status='fail',case=args.case,optimizer_updates=0)
    trace=(args.output/'trace.jsonl').open('w',buffering=1)
    try:
        import numpy as np
        import torch
        from types import SimpleNamespace
        from drl.MARL.c1_initialization import PolicyControlledHandoff
        from drl.MARL.isaaclab_c1_env import MSSRC1IsaacLabEnv,make_c1_isaaclab_env_cfg
        from drl.MARL.isaaclab_c1_runner import seed_native_training,create_native_learner,collect_native_rollout
        from drl.MARL.c1_runner import CHECKPOINT,BC_SHA256,build_models,build_observation_bridge
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from smores_ep.config.geometry import SmoresGeometry
        from smores_ep.config.physics import SMORES_EP_MAX_WHEEL_SPEED_RAD_S,SMORES_DOF_SIMULATION_MAX_SPEED_RAD_S
        seed_native_training(args.global_seed)
        full=args.case=='legacy_regression'
        legacy=args.case!='loose'
        n=4 if full else 2
        domains=('navigation','stairs','gap','manipulation') if full else ('manipulation',)
        owners=tuple(f'smores_{i:02d}' for i in range(1,9))
        cfg=make_c1_isaaclab_env_cfg(possible_agents=owners,num_envs=n,device=args.device)
        cfg.domains=domains;cfg.seeds=(7017,12005,16022,20004) if full else (20004,20004)
        cfg.initialization_mode='expert_validation' if legacy else 'loose_modules'
        cfg.seed=args.global_seed;cfg.train_sampler_seed=args.global_seed
        cfg.resample_train=True;cfg.smoke_seed_cycle=True;cfg.episode_length_s=60.
        cfg.reward_config_path=str(repo/'logs/appo_smoke/calibration_validated_train/reward_config.initial.json')
        if full:
            cfg.train_seed_pools=json.loads((repo/'logs/isaaclab_c1_native/training_readiness/learning_seed_manifest.json').read_text())['seed_pools']
        else:cfg.train_seed_pools={'manipulation':(20004,20273)}
        env=MSSRC1IsaacLabEnv(cfg);wrapper=SkrlVecEnvWrapper(env,ml_framework='torch')
        observations,_=wrapper.reset()
        if legacy:
            env.assemble_until_ready(max_steps=18000)
            observations,*_=wrapper.step({o:torch.zeros(n,1,device=args.device) for o in owners})
        bridge=build_observation_bridge(repo)
        model_args=SimpleNamespace(repository_root=repo,checkpoint=repo/CHECKPOINT,device=args.device,
            num_envs=n,rollout_steps=args.bc_steps,learning_epochs=2,mini_batches=2,
            gamma=.99,alpha=.5,altruism_radius_m=.5,train=False)
        actor,critic=build_models(model_args,bridge)
        env.initialize_policy_bridge(observation_bridge=bridge)
        learner=create_native_learner(model_args,bridge,actor,critic,owners,env.runtime_context_provider)
        def actor_hash():
            digest=hashlib.sha256()
            for key,tensor in sorted(actor.state_dict().items()):
                digest.update(key.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
            return digest.hexdigest()
        initial_hash=actor_hash()
        accepted_nonempty=[]
        original_step=wrapper.step
        def measured_step(actions):
            proposals=[x for x in env.staged]
            response=original_step(actions)
            for i,proposal in enumerate(proposals):
                if proposal and proposal[0] and getattr(env._outcomes[i],'accepted',False):
                    accepted_nonempty.append(dict(env_index=i,commands=len(proposal[0])))
            return response
        wrapper.step=measured_step
        def forbidden_update(*a,**k):raise AssertionError('Reset diagnostic attempted an optimizer update')
        learner.coordinated_update=forbidden_update
        collected=collect_native_rollout(wrapper,learner,rollout_steps=args.bc_steps,
            initial_observations=observations,initial_states=wrapper.state())
        samples=learner._structured_rollout_sidecar._samples
        decisions=[]
        for i,s in enumerate(samples):
            resource_count=len(s.routing_snapshot['resource_candidates'])
            chosen=torch.nonzero(s.action.presence[resource_count:],as_tuple=False).reshape(-1).cpu().tolist()
            bindings=s.routing_snapshot['structural_bindings']
            decisions.append(dict(row=i,structural_selected=len(chosen),
                structural_action_ids=[bindings[j][0].action_id for j in chosen]))
        (args.output/'policy_actions.json').write_text(json.dumps(decisions,default=str,indent=2)+'\n')
        result.update(bc_steps=args.bc_steps,bc_checkpoint=str(model_args.checkpoint),bc_sha256=BC_SHA256,
            accepted_nonempty_bc_proposals=len(accepted_nonempty),
            structural_policy_actions=sum(d['structural_selected'] for d in decisions),
            initial_expert_assembly_used_for_regression=legacy)
        if not accepted_nonempty:raise RuntimeError('BC emitted no accepted nonempty physical action; reset gate not demonstrated')
        selected=tuple(range(n)) if full else (0,)
        def measured(i):
            state={}
            for o,reader in env.state_readers_by_env[i].items():
                v=reader._articulation.view
                state[o]=dict(root=v.get_root_transforms().numpy().copy(),velocity=v.get_root_velocities().numpy().copy(),
                    q=v.get_dof_positions().numpy().copy(),qd=v.get_dof_velocities().numpy().copy(),
                    position_targets=v.get_dof_position_targets().numpy().copy(),velocity_targets=v.get_dof_velocity_targets().numpy().copy(),
                    efforts=v.get_dof_actuation_forces().numpy().copy(),stiffness=v.get_dof_stiffnesses().numpy().copy(),
                    damping=v.get_dof_dampings().numpy().copy(),links=v.get_link_transforms().numpy().copy())
            return state
        def serial(value):
            if isinstance(value,np.ndarray):return value.tolist()
            if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [serial(v) for v in value]
            return value
        def emit(event,**fields):
            payload=dict(event=event,physics_step=env._sim_step_counter,seeds=list(env.seed_by_env),
                generations=list(env.generation),selected=list(selected),**fields)
            trace.write(json.dumps(serial(payload),allow_nan=False)+'\n')
        before=[measured(i) for i in range(n)]
        sibling_backend=env.physical_expert_backends[1] if not full else None
        sibling_identity=(env.seed_by_env[1],env.generation[1],env.marl_steps[1]) if not full else None
        emit('before_reset',states=before)
        expert_records_before=len(env.assembly_records)
        if legacy:
            # This conversion is diagnostic only: Expert generated a controlled
            # starting state, never actions in the ensuing policy-controlled run.
            env.cfg.initialization_mode='loose_modules'
            handoffs=list(env.handoffs)
            for i in selected:handoffs[i]=PolicyControlledHandoff()
            env.handoffs=tuple(handoffs)
            env.morphology_by_env=(None,)*n
        env._reset_idx(torch.tensor(selected,device=env.device))
        after=[measured(i) for i in range(n)]
        emit('after_reset',states=after)
        sibling_unchanged=None if full else True
        if not full:
            sibling_unchanged=(env.physical_expert_backends[1] is sibling_backend and sibling_identity==
                (env.seed_by_env[1],env.generation[1],env.marl_steps[1]) and all(
                    (np.allclose(before[1][o][k],after[1][o][k],rtol=0.,atol=1e-6) if k=='links'
                     else np.array_equal(before[1][o][k],after[1][o][k]))
                    for o in owners for k in before[1][o]))
            if not sibling_unchanged:raise RuntimeError('Selected reset changed sibling physical/controller state')
        for i in selected:
            if env.docking_managers[i].connections:raise RuntimeError('Docking constraints survived reset')
            for o,state in after[i].items():
                for key in ('velocity','q','qd','position_targets','velocity_targets','efforts','stiffness'):
                    if np.count_nonzero(state[key]):raise RuntimeError(f'Reset left nonzero {key} in env{i}/{o}')
        geometry=SmoresGeometry();height=max(float(s['root'][0,2]) for i in selected for s in after[i].values())
        speed_bound=geometry.wheel_radius_m*SMORES_EP_MAX_WHEEL_SPEED_RAD_S+math.sqrt(2*9.81*height)
        angular_bound=SMORES_DOF_SIMULATION_MAX_SPEED_RAD_S+speed_bound/geometry.wheel_radius_m
        start=env._sim_step_counter;peak_speed=peak_angular=0.;ticks=[]
        previous_physics=env.sim.step
        def observe_physics(*a,**kw):
            nonlocal peak_speed,peak_angular
            response=previous_physics(*a,**kw)
            states=[measured(i) for i in selected]
            speed=max(float(np.linalg.norm(s['velocity'][0,:3])) for row in states for s in row.values())
            angular=max(float(np.linalg.norm(s['velocity'][0,3:])) for row in states for s in row.values())
            finite=all(np.isfinite(v).all() for row in states for s in row.values() for v in s.values())
            peak_speed=max(peak_speed,speed);peak_angular=max(peak_angular,angular)
            ticks.append(dict(step=env._sim_step_counter,speed=speed,angular=angular,finite=finite))
            emit('physics_tick',states=states,speed=speed,angular=angular,finite=finite)
            if not finite or speed>speed_bound or angular>angular_bound:
                raise RuntimeError(f'Post-reset physical divergence: v={speed}, omega={angular}')
            return response
        env.sim.step=observe_physics
        env._suspend_marl=True
        while env._sim_step_counter-start<args.physics_ticks:
            env.step({o:torch.zeros(n,1,device=env.device) for o in owners})
        env.sim.step=previous_physics
        if len(env.assembly_records)!=expert_records_before:raise RuntimeError('Expert controlled post-reset physics')
        if actor_hash()!=initial_hash:raise RuntimeError('BC actor changed without optimizer')
        result.update(status='pass',bc_checkpoint=str(model_args.checkpoint),bc_sha256=BC_SHA256,
            actor_unchanged=True,accepted_nonempty_bc_proposals=len(accepted_nonempty),
            structural_policy_actions=sum(d['structural_selected'] for d in decisions),
            bc_steps=args.bc_steps,rollout=collected,initialization_mode_after_reset=env.cfg.initialization_mode,
            seeds_after_reset=list(env.seed_by_env),selected=list(selected),sibling_immediate_unchanged=sibling_unchanged,
            physics_ticks=env._sim_step_counter-start,peak_linear_speed_m_s=peak_speed,peak_angular_speed_rad_s=peak_angular,
            linear_speed_bound_m_s=speed_bound,angular_speed_bound_rad_s=angular_bound,
            bound_reason='Free fall from initial root height plus maximum configured wheel surface speed; angular equivalent plus configured internal motor speed. Diagnostic detection only, never a clamp.',
            post_reset_expert_actions=0,initial_expert_assembly_used_for_regression=legacy)
    except Exception as error:
        result.update(error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        print('C1_RESET_PROBE_FAIL:',str(error),flush=True)
    finally:
        trace.close()
        result['runtime_unchanged']=all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        if not result['runtime_unchanged']:result['status']='fail'
        (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('C1_RESET_PROBE_RESULT:',json.dumps(result),flush=True)
        if env is not None:env.close()
        application.close()
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':
    raise SystemExit(main())
