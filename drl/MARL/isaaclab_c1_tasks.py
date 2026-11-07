"""C1 robot/task hooks for native Isaac Lab; no stepping/transport abstraction."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import math
import time
import torch
from drl.MARL.c1_initialization import InitializationMode, PolicyControlledHandoff, mode_for


class NativeC1TaskHooks:
    def _setup_native_courses(self):
        from pxr import PhysxSchema,UsdShade,UsdPhysics
        from drl.MARL.c1_runner import make_course,MORPHOLOGIES
        from smores_ep.isaac.obstacle_course import _install_course_boxes,_install_button_prismatic_joint
        from smores_ep.isaac.phase1_training_course import _install_phase1_static_cones
        domains=tuple(self.cfg.domains)
        starts={'navigation':7000,'stairs':12000,'gap':16000,'manipulation':20000}
        if not domains or any(d not in starts for d in domains):raise ValueError('Invalid C1 domains')
        self.domain_by_env=tuple(domains[i%len(domains)] for i in range(self.num_envs))
        if self.cfg.course_split != 'train' and (self.cfg.resample_train or not self.cfg.seeds):
            raise ValueError('non-TRAIN courses require explicit fixed evaluation seeds, without TRAIN resampling')
        self.train_sampler = None
        if mode_for(self) is InitializationMode.ASSEMBLED_STATE:
            from drl.MARL.c1_snapshot_runtime import initialize_snapshot_sampling
            initialize_snapshot_sampling(self)
        elif self.cfg.resample_train:
            from drl.MARL.c1_train_sampling import NativeC1TrainSampler
            self.train_sampler = NativeC1TrainSampler(self.domain_by_env, self.cfg.train_sampler_seed,
                self.cfg.train_seed_pools, cycle=self.cfg.smoke_seed_cycle,
                initial_seeds=tuple(self.cfg.seeds) if self.cfg.seeds else None)
            self.seed_by_env = tuple(self.train_sampler.sample(i).seed for i in range(self.num_envs))
        else:
            self.seed_by_env=tuple(self.cfg.seeds) if self.cfg.seeds else tuple(starts[d]+i for i,d in enumerate(self.domain_by_env))
        if len(self.seed_by_env)!=self.num_envs:raise ValueError('One curriculum seed per native environment required')
        self.courses=tuple(make_course(d,s,self.cfg.course_split) for d,s in zip(self.domain_by_env,self.seed_by_env,strict=True))
        from drl.MARL.reward_config import RewardConfig
        self.reward_config=RewardConfig.load(self.cfg.reward_config_path)
        self.morphology_by_env=(tuple(MORPHOLOGIES[d] for d in self.domain_by_env)
            if mode_for(self) is InitializationMode.EXPERT_VALIDATION else (None,)*self.num_envs)
        material=UsdShade.Material.Define(self.sim.stage,'/World/materials/dynamic_ground')
        physics=UsdPhysics.MaterialAPI.Apply(material.GetPrim());physics.CreateStaticFrictionAttr(.9);physics.CreateDynamicFrictionAttr(.75);physics.CreateRestitutionAttr(0.)
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr('multiply')
        self.plunger_paths={}
        for i,(domain,course) in enumerate(zip(self.domain_by_env,self.courses,strict=True)):
            root=f'/World/envs/env_{i}/C1Course'
            _install_course_boxes(self.sim.stage,root,course.support_boxes)
            if domain=='navigation':_install_phase1_static_cones(self.sim.stage,root+'/Cones',course.cones)
            if domain=='manipulation':
                center=tuple(float(course.button_center_xyz_m[k]+self.scene.env_origins[i,k]) for k in range(3))
                _install_button_prismatic_joint(self.sim.stage,root_path=root,plunger_name=course.button_plunger.name,
                    center_xyz_m=center,press_direction_world_xy=course.press_direction_world_xy,
                    plunger_yaw_deg=course.button_plunger.yaw_deg,joint_name='ButtonPrismaticJoint')
                self.plunger_paths[i]=root+'/'+course.button_plunger.name
        self.scene.filter_collisions(global_prim_paths=[])

    def _initialize_task_buffers(self):
        from drl.MARL.runtime_handoff import DirectMARLHandoff
        from drl.MARL.runtime_signals import StagnationDetector
        authority=(DirectMARLHandoff if mode_for(self) is InitializationMode.EXPERT_VALIDATION
                   else PolicyControlledHandoff)
        self.handoffs=tuple(authority() for _ in range(self.num_envs))
        self.marl_steps=[0]*self.num_envs
        self.generation=[0]*self.num_envs
        self.intended_connections=[frozenset() for _ in range(self.num_envs)]
        self.end_effector_ids=[None]*self.num_envs
        self.stagnation=[StagnationDetector() for _ in range(self.num_envs)]
        self.staged=[None]*self.num_envs
        self._prepared_cache={}
        self._raw_current=None;self._post_raw=None
        self.owner_scalar_rewards={owner:torch.zeros(self.num_envs,1,device=self.device) for owner in self.cfg.possible_agents}
        self.team_scalar_rewards=torch.zeros(self.num_envs,device=self.device)
        self._terminal=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self._truncated=self._terminal.clone()
        self._last_evaluations=[None]*self.num_envs
        self._marl_pre=[False]*self.num_envs
        self.assembly_runtimes=[None]*self.num_envs
        self.assembly_verified=[False]*self.num_envs
        self.assembly_started_wall=[None]*self.num_envs
        self.assembly_records=[]
        self.reset_records=[]

    def _resample_course(self, i):
        if mode_for(self) is InitializationMode.ASSEMBLED_STATE:
            from drl.MARL.c1_snapshot_runtime import resample_snapshot_course
            resample_snapshot_course(self,i)
            return
        if self.train_sampler is None or self.generation[i] == 0:
            return
        from drl.MARL.c1_runner import make_course
        from drl.MARL.isaaclab_course_reset import update_native_course
        sample = self.train_sampler.sample(i)
        course = make_course(sample.domain, sample.seed, sample.split)
        update_native_course(self.sim.stage, env_root=f'/World/envs/env_{i}',
            domain=sample.domain, course=course, origin=self.scene.env_origins[i].cpu().tolist())
        seeds, courses = list(self.seed_by_env), list(self.courses)
        seeds[i], courses[i] = sample.seed, course
        self.seed_by_env, self.courses = tuple(seeds), tuple(courses)

    def _reset_idx(self,env_ids):
        mode=mode_for(self)
        reset_started = time.monotonic()
        super()._reset_idx(env_ids)
        if not hasattr(self,'handoffs'):self._initialize_task_buffers()
        if mode is InitializationMode.EXPERT_VALIDATION:self._ensure_physical_expert_backends()
        else:self._ensure_native_actuation_backends()
        if not hasattr(self,'native_dispatcher'):
            from drl.MARL.smores_runtime_binding import SMORESNativeDispatcher,SMORESNativeActuationBridge
            executor=lambda i:self.physical_expert_backends[i]['primitive_executor']
            router=lambda i:self.physical_expert_backends[i]['router']
            now=lambda i:self._sim_step_counter*self.physics_dt
            self.native_dispatcher=SMORESNativeDispatcher(executor_provider=executor,now_provider=now)
            self.native_actuation=SMORESNativeActuationBridge(dispatcher=self.native_dispatcher,executor_provider=executor,router_provider=router,now_provider=now)
        from drl.MARL.isaaclab_expert_runtime import InProcessAssemblyRuntime
        from drl.MARL.c1_runner import attached_connections
        from smores_ep.docking.model import DockingCommand
        from smores_ep.isaac.phase1_stairs_curriculum import place_phase1_stairs_spawn_layout
        from smores_ep.scenarios.parallel_self_assembly import place_button_spawn_layout
        ids=[int(i) for i in env_ids]
        self._prepared_cache.clear()
        for i in ids:
            backend=self.physical_expert_backends[i]
            # Cancel while the old physical state is still coherent. A cancel
            # may write a hold; all targets are overwritten before teleport.
            for goal in tuple(backend['primitive_executor'].active_goals):
                backend['primitive_executor'].cancel(goal.goal_id,self._sim_step_counter*self.physics_dt)
            self.native_dispatcher.clear(i)
            self.staged[i]=None
            docking=self.docking_managers[i]
            for c in tuple(docking.connections):
                docking.handle(DockingCommand('detach',c.first_face.module_id,c.second_face.module_id,c.first_face.face_name,c.second_face.face_name))
            if docking.connections:raise RuntimeError('Reset failed to detach selected docking constraints')
            self._resample_course(i)
            from drl.MARL.native_episode_reset import isolate_course_colliders
            isolation=isolate_course_colliders(self.sim.stage,f'/World/envs/env_{i}',
                tuple(f'/World/envs/env_{j}' for j in range(self.num_envs)))
            backend['router'].reset_free_modules(tuple(self.cfg.possible_agents))
            from drl.MARL.native_episode_reset import clear_authored_external_wrenches,flush_pending_physics
            clear_authored_external_wrenches(self.sim.stage,self.module_roots_by_env[i].values())
            flush_pending_physics()
            if i in self.plunger_paths:
                from drl.MARL.isaaclab_articulation import reset_rigid_body
                path = self.plunger_paths[i]
                position = tuple(float(self.courses[i].button_center_xyz_m[k] + self.scene.env_origins[i, k]) for k in range(3))
                reset_rigid_body(self.native_transforms.rigid_views[path], position, self.courses[i].button_plunger.yaw_deg)
            restored=None
            if mode is InitializationMode.ASSEMBLED_STATE:
                from drl.MARL.c1_snapshot_runtime import restore_snapshot_roots
                restored=restore_snapshot_roots(self,i)
            else:
                layout=self._spawn_layout
                if self.domain_by_env[i]=='stairs':layout=place_phase1_stairs_spawn_layout(layout,self.courses[i])
                if self.domain_by_env[i]=='manipulation':layout=place_button_spawn_layout(layout,self.courses[i])
                for owner,(x,y,z,yaw) in layout.items():
                    reader=self.state_readers_by_env[i][owner]
                    origin=self.scene.env_origins[i].cpu().tolist()
                    reader._articulation.reset_pose((x+origin[0],y+origin[1],z+origin[2]),yaw)
            self.native_transforms.invalidate()
            # Fresh per-row controllers retire all executor/group/posture caches.
            # The sibling keeps the very same backend objects and targets.
            self._ensure_native_actuation_backends(reset_env_ids=(i,))
            backend=self.physical_expert_backends[i]
            backend['router'].reset_free_modules(tuple(self.cfg.possible_agents))
            if restored is not None:
                self.docking_managers[i].restore_connections(restored['connections'])
                self.native_transforms.invalidate()
            self.handoffs[i].reset()
            if mode is InitializationMode.EXPERT_VALIDATION:
                fsm=self._start_expert_assembly(self.morphology_by_env,stamp_s=self._sim_step_counter*self.physics_dt,env_ids=(i,))[0]
                self.assembly_runtimes[i]=InProcessAssemblyRuntime(fsm,backend)
                self.handoffs[i].begin_expert(self.morphology_by_env[i])
            else:
                self.assembly_runtimes[i]=None
                self.handoffs[i].activate_policy()
            self.assembly_verified[i]=False;self.marl_steps[i]=0;self.generation[i]+=1
            self.assembly_started_wall[i]=time.monotonic()
            self._last_evaluations[i]=None
            self.intended_connections[i]=frozenset();self.end_effector_ids[i]=None
            self.native_dispatcher.clear(i)
            self.stagnation[i].reset();self.staged[i]=None
            snapshot_state={}
            if restored is not None:
                from drl.MARL.c1_snapshot_runtime import prime_snapshot_episode
                snapshot_state=prime_snapshot_episode(self,i,restored)
            if not hasattr(self,'reset_records'):self.reset_records=[]
            self.reset_records.append(dict(env_index=i,generation=self.generation[i],
                seed=self.seed_by_env[i],initialization_mode=mode.value,isolation=isolation,**snapshot_state))
        self._raw_current=None
        self.training_reset_wall_s = getattr(self,'training_reset_wall_s',0.) + time.monotonic()-reset_started

    def _local_snapshots(self):
        snapshots=self._state_graph_snapshots(stamp_s=self._sim_step_counter*self.physics_dt)
        result=[]
        for i,raw in enumerate(snapshots):
            raw=deepcopy(raw);origin=self.scene.env_origins[i].cpu().tolist()
            # Snapshot node attributes intentionally alias module payloads.
            # Update each shared module exactly once; never translate relative vectors.
            for module in raw['state']['modules']:
                for key in ('position',):module[key]=[module[key][k]-origin[k] for k in range(3)]
                module['pose']['position']=list(module['position'])
                for connector in module['connectors']:
                    connector['position_world']=[connector['position_world'][k]-origin[k] for k in range(3)]
            payload=self.courses[i].to_observation()
            if i in self.plunger_paths:
                xyz=self.native_transforms(self.plunger_paths[i]).ExtractTranslation()
                current=[float(xyz[k])-origin[k] for k in range(3)]
                button=payload['button'];normal=self.courses[i].press_direction_world_xy
                button['current_center_xyz_m']=current
                button['depression_m']=sum((current[k]-self.courses[i].button_center_xyz_m[k])*normal[k] for k in range(2))
                button['plunger_path']=self.plunger_paths[i]
            raw['state']['course']=payload;raw['graph']['global_attributes']['course']=payload
            result.append(raw)
        return tuple(result)

    def cached_raw_snapshot(self,env_index):
        if self._raw_current is None:self._raw_current=self._local_snapshots()
        return deepcopy(self._raw_current[env_index])

    def stage_decoded(self,env_index,decoded_actions,outcome):
        if not self.handoffs[env_index].marl_active:
            raise RuntimeError('MARL action staged without verified native handoff')
        self.staged[env_index]=(tuple(decoded_actions),outcome)

    def clear_staged(self,env_index=None):
        for i in range(self.num_envs) if env_index is None else (env_index,):self.staged[i]=None

    def _pre_physics_step(self,actions):
        self._actions=actions
        self._raw_current=self._local_snapshots();self._pre_raw=self._raw_current
        if mode_for(self) in (InitializationMode.LOOSE_MODULES,InitializationMode.ASSEMBLED_STATE):
            from drl.MARL.c1_native_observation import start_step_evidence
            start_step_evidence(self)
        self._marl_pre=[h.marl_active and hasattr(self,'runtime_context_provider') and not getattr(self,'_suspend_marl',False) for h in self.handoffs]
        self._outcomes=[SimpleNamespace(accepted=True,code='EMPTY',owner_ids=()) for _ in range(self.num_envs)]
        from drl.MARL.smores_runtime_binding import SMORESMARLActionRejected
        for i,active in enumerate(self._marl_pre):
            if not active:continue
            packet=self.staged[i];self.staged[i]=None
            self.native_dispatcher.clear(i)
            if packet is None:continue
            decoded,outcome=packet;self._outcomes[i]=outcome
            if not outcome.accepted:continue
            try:
                if outcome.projection is not None:
                    self.native_dispatcher.dispatch_projected(env_index=i,projection=outcome.projection)
                else:
                    self.native_dispatcher(env_index=i,decoded_actions=decoded,context=None)
                self._outcomes[i]=self.native_dispatcher.proposal_outcome(i)
            except SMORESMARLActionRejected as error:
                self._outcomes[i]=error.outcome

    def _apply_action(self):
        stamp=self._sim_step_counter*self.physics_dt
        from drl.MARL.c1_runner import attached_connections
        for i,backend in enumerate(self.physical_expert_backends):
            if not self.handoffs[i].marl_active:
                runtime=self.assembly_runtimes[i]
                if runtime is None:continue
                if self._sim_step_counter%4==0:
                    runtime.tick(stamp)
                    if runtime.decision is not None and runtime.decision.done:
                        current=self._current_attributed_graphs(stamp_s=stamp)[i]
                        target=self._load_target_graph(self.morphology_by_env[i])
                        match=self._ensure_topology_matchers()[i].configuration_assignment(current,target)
                        if match is None or len(self.docking_managers[i].connections)!=7:raise RuntimeError('Expert terminal without physical target topology')
                        self.assembly_verified[i]=True
                        self.assembly_records.append(dict(env_index=i, domain=self.domain_by_env[i],
                            seed=self.seed_by_env[i], generation=self.generation[i],
                            wall_s=time.monotonic()-self.assembly_started_wall[i],
                            physics_step=self._sim_step_counter, connections=7))
                        self.handoffs[i].observe_expert_terminal(True);self.handoffs[i].observe_topology(self.morphology_by_env[i])
                        raw=self._local_snapshots()[i];self.intended_connections[i]=attached_connections(raw)
                        from mssr_expert.planning.smores_ep.attributed_adapter import target_roles_from_graph
                        roles=target_roles_from_graph(target)
                        self.end_effector_ids[i]=next((mid for vertex,mid in match.target_to_module.items() if roles[vertex].get('target_role')=='end_effector'),None)
                runtime.apply()
            else:
                if hasattr(self,'native_actuation'):
                    result=self.native_actuation.step(i)
                    if result.proposal_outcome is not None:self._outcomes[i]=result.proposal_outcome
                    if mode_for(self) in (InitializationMode.LOOSE_MODULES,InitializationMode.ASSEMBLED_STATE):
                        from drl.MARL.c1_native_observation import record_physics_evidence
                        record_physics_evidence(self,i,result)
                else:
                    commands=backend['primitive_executor'].step(stamp).commands
                    backend['router'].apply(backend['primitive_executor'].compose_with_baseline({},commands))

    def _get_dones(self):
        self.native_transforms.invalidate();self._raw_current=self._local_snapshots();self._post_raw=deepcopy(self._raw_current)
        self._prepared_cache.clear();self._terminal.zero_();self._truncated.zero_();self.team_scalar_rewards.zero_()
        for values in self.owner_scalar_rewards.values():values.zero_()
        from drl.MARL.c1_runner import captured_observation,irrecoverable_fall,absorbing_failure_reward
        from drl.MARL.runtime_signals import collect_runtime_signals
        from drl.MARL.phase1_transition import Phase1TransitionEvaluator
        from drl.MARL.stairs_transition import StairsTransitionEvaluator
        from drl.MARL.gap_transition import GapTransitionEvaluator
        from drl.MARL.manipulation_transition import ManipulationTransitionEvaluator
        classes=dict(navigation=Phase1TransitionEvaluator,stairs=StairsTransitionEvaluator,gap=GapTransitionEvaluator,manipulation=ManipulationTransitionEvaluator)
        for i,active in enumerate(self._marl_pre):
            if not active:
                self.episode_length_buf[i]-=1
                telemetry=getattr(self,'training_telemetry',None)
                if telemetry is not None:
                    telemetry.observe_transition(env_index=i,domain=self.domain_by_env[i],seed=self.seed_by_env[i],
                        generation=self.generation[i],reward=0.,terminated=False,truncated=False,success=False,
                        active=False,delta_s=self.step_dt)
                continue
            if mode_for(self) is not InitializationMode.EXPERT_VALIDATION:
                from drl.MARL.c1_native_observation import frame_from_native
                from drl.MARL.c1_native_runtime import NativeC1RewardRuntime
                if not hasattr(self,'c1_runtime'):
                    self.c1_runtime=NativeC1RewardRuntime(settings_path=getattr(self.cfg,'c1_probe_config_path',None),
                        evidence_directory=getattr(self.cfg,'c1_evidence_directory',None))
                    self.c1_episode_records=self.c1_runtime.episode_records
                self.marl_steps[i]+=1
                evaluation=self.c1_runtime.evaluate(self,i,frame_from_native(self,i,self._post_raw[i]))
                self.team_scalar_rewards[i]=evaluation.team_reward
                for owner,value in evaluation.owner_rewards.items():self.owner_scalar_rewards[owner][i,0]=value
                self._terminal[i]=evaluation.terminated;self._truncated[i]=evaluation.truncated
                self._last_evaluations[i]=evaluation
                if self._terminal[i] or self._truncated[i]:self.handoffs[i].finish_episode()
                continue
            self.marl_steps[i]+=1;before_raw,after_raw=self._pre_raw[i],self._post_raw[i]
            before=captured_observation(self.domain_by_env[i],before_raw,self.intended_connections[i],end_effector_id=self.end_effector_ids[i])
            after=captured_observation(self.domain_by_env[i],after_raw,self.intended_connections[i],end_effector_id=self.end_effector_ids[i])
            positions=lambda raw:{m['module_id']:tuple(m['position']) for m in raw['state']['modules']}
            frame=collect_runtime_signals(owner_ids=self.cfg.possible_agents,delta_seconds=self.step_dt,proposal_outcome=self._outcomes[i],
                previous_module_positions_m=positions(before_raw),next_module_positions_m=positions(after_raw),stagnation_detector=self.stagnation[i])
            signal=lambda *a,**kw:frame.signals;domain=self.domain_by_env[i]
            if domain=='manipulation':evaluator=classes[domain](course=self.courses[i],signal_builder=signal,shaping_gamma=self.cfg.shaping_gamma)
            else:evaluator=classes[domain](courses=(self.courses[i],),max_episode_steps=self.max_episode_length,signals_provider=signal,shaping_gamma=self.cfg.shaping_gamma)
            evaluation=evaluator(env_index=0,episode_step=self.marl_steps[i],previous_state=before,next_state=after,actuation_result=self._outcomes[i],physics_result=None)
            raw=evaluation.raw_reward;terminated=bool(evaluation.terminated)
            if irrecoverable_fall(self.courses[i],after_raw):
                terminated=True;raw=absorbing_failure_reward(domain,self.courses[i],before,after,frame.signals,self.cfg.shaping_gamma)
            availability=dict(frame.availability)
            for term,value in asdict(raw.team).items():
                availability.setdefault(term,dict(status='measured_present' if value else 'measured_absent',source='native_captured_transition',units='raw_task_term'))
            scalar=self.reward_config.scalarize(domain,raw,availability=availability)
            self.team_scalar_rewards[i]=scalar.team
            for owner in self.cfg.possible_agents:self.owner_scalar_rewards[owner][i,0]=scalar.owners[owner]
            self._terminal[i]=terminated;self._truncated[i]=not terminated and self.marl_steps[i]>=self.max_episode_length
            self._last_evaluations[i]=evaluation
            telemetry=getattr(self,'training_telemetry',None)
            if telemetry is not None:
                telemetry.observe_transition(env_index=i,domain=domain,seed=self.seed_by_env[i],generation=self.generation[i],
                    reward=scalar.team,terminated=bool(self._terminal[i]),truncated=bool(self._truncated[i]),
                    success=bool(raw.team.success),active=True,delta_s=self.step_dt)
            if self._terminal[i] or self._truncated[i]:self.handoffs[i].finish_episode()
        return ({owner:self._terminal.clone() for owner in self.cfg.possible_agents},{owner:self._truncated.clone() for owner in self.cfg.possible_agents})

    def _get_rewards(self):
        return {owner:self.team_scalar_rewards.clone() for owner in self.cfg.possible_agents}

    def _get_observations(self):
        if not hasattr(self,'handoffs'):return {owner:torch.zeros(self.num_envs,1,device=self.device) for owner in self.cfg.possible_agents}
        self._raw_current=None
        rows=[self.cached_raw_snapshot(i) for i in range(self.num_envs)]
        return {owner:torch.tensor([[next(m['position'][0] for m in row['state']['modules'] if m['module_id']==owner)] for row in rows],device=self.device) for owner in self.cfg.possible_agents}

    def _get_states(self):
        return torch.stack(list(self._get_observations().values())).mean(0)

    def assemble_until_ready(self,max_steps=4000):
        if mode_for(self) is not InitializationMode.EXPERT_VALIDATION:
            raise RuntimeError('Expert assembly is restricted to expert_validation initialization')
        for step in range(max_steps):
            deadline = getattr(self,'_wall_deadline_s',None)
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError('campaign wall budget reached during initial Expert assembly')
            if all(h.marl_active for h in self.handoffs):
                print('NATIVE_C1_ASSEMBLY_VERIFIED:', {'steps': step, 'connections': [len(d.connections) for d in self.docking_managers], 'morphologies': self.morphology_by_env}, flush=True)
                return
            if step % 300 == 0:
                print('NATIVE_C1_ASSEMBLY_PROGRESS:', {'steps': step, 'connections': [len(d.connections) for d in self.docking_managers]}, flush=True)
            self.step({owner:torch.zeros(self.num_envs,1,device=self.device) for owner in self.cfg.possible_agents})
        raise RuntimeError('Native Expert assembly horizon exceeded')

    def initialize_policy_bridge(self,observation_bridge=None,*,actor_only=False,bulk_host_reads=True,
                                 structural_scheduler='fixed'):
        import json
        from pathlib import Path
        from drl.MARL.reward_config import RewardConfig
        from drl.MARL.c1_runner import build_observation_bridge
        from drl.MARL.native_policy_bridge import build_native_policy_bridge
        from drl.MARL.smores_runtime_binding import SMORESMARLRuntimeBinding
        from drl.IL.bc_v1.live_policy import smores_claims
        root=Path(__file__).resolve().parents[2]
        self.observation_bridge=observation_bridge or build_observation_bridge(root)
        self.task_context_config=json.loads((root/'configs/bc_v1/task_context.json').read_text())
        self.reward_config=RewardConfig.load(self.cfg.reward_config_path)
        class Source:
            num_envs=self.num_envs
            def prepare_env(source,i):return self.prepare_env(i)
            def prepare_env_with_candidates(source,i,*,include_action_candidates):
                return self.prepare_env(i,include_candidates_override=include_action_candidates)
            def prepare_env_privileged(source,i,*,include_action_candidates=False):
                return self.prepare_env(i,include_candidates_override=include_action_candidates,include_actor=False)
            def reset(source):self._prepared_cache.clear()
        from drl.MARL.c1_initialization import native_action_mask
        def mask(actor,route):return native_action_mask(route,mode_for(self),device=self.device)
        def neutral(i,context):return {owner:torch.zeros(1,device=self.device) for owner in self.cfg.possible_agents}
        bundle=build_native_policy_bridge(self,observation_bridge=self.observation_bridge,task_context_provider=None,
            binding=SMORESMARLRuntimeBinding(adapter=self.observation_bridge.adapter,claim_resolver=smores_claims),
            handoffs=self.handoffs,hard_mask_builder=mask,placeholder_action=neutral,observation_source=Source(),
            post_step_observation_provider=lambda i:self.prepare_env(
                i,post_step=True,include_candidates_override=False,include_actor=False),
            project_actions=mode_for(self) is not InitializationMode.EXPERT_VALIDATION,
            actor_only=actor_only,bulk_host_reads=bulk_host_reads,
            structural_scheduler=structural_scheduler)
        self.runtime_context_provider=bundle.provider
        return bundle

    def prepare_env(self,i,post_step=False,include_candidates_override=None,include_actor=True):
        from drl.MARL.c1_runner import physical_task_view
        raw=self._post_raw[i] if post_step else self.cached_raw_snapshot(i)
        # Terminal post-step rows retain the phase that produced the action,
        # even when native autoreset already returned this env to Expert.
        include_candidates = bool(self._marl_pre[i] if post_step else self.handoffs[i].marl_active)
        if include_candidates_override is not None:
            include_candidates = bool(include_candidates_override)
        privileged_prepare=getattr(self.observation_bridge,'prepare_privileged_snapshot',None)
        # Optional capability: injected bridges keep their original actor ABI.
        omit_actor=not include_actor and callable(privileged_prepare)
        key=(i,post_step,self.generation[i],raw['stamp'],include_candidates,not omit_actor)
        if key not in self._prepared_cache:
            context,obs=physical_task_view(self.domain_by_env[i],raw['state']['course'],self.task_context_config)
            cache_options = {}
            if getattr(self, '_c1_static_preparation_enabled', False):
                from drl.IL.bc_v1.runtime_static_cache import RuntimeStaticCache
                caches = self._c1_runtime_static_caches
                if i not in caches:
                    caches[i] = RuntimeStaticCache()
                caches[i].set_epoch(self.generation[i])
                # Candidate-free sleeping/bootstrap views have no reusable
                # table identity; keep them out of the bounded layout cache.
                if include_candidates:
                    cache_options['static_cache'] = caches[i]
            prepare=privileged_prepare if omit_actor else self.observation_bridge.prepare_snapshot
            self._prepared_cache[key]=prepare(raw,task_context=context,observation=obs,
                include_action_candidates=include_candidates,**cache_options)
        return self._prepared_cache[key]

    def current_altruistic_rollout_batch(self):
        from drl.MARL.altruistic_rollout import AltruisticRolloutBatchPayload
        rows=tuple(tuple(m['module_id'] for m in raw['state']['modules']) for raw in self._pre_raw)
        positions=tuple({m['module_id']:torch.tensor(m['position'],device=self.device) for m in raw['state']['modules']} for raw in self._pre_raw)
        mapping=tuple({owner:owner for owner in self.cfg.possible_agents} for _ in range(self.num_envs))
        return AltruisticRolloutBatchPayload(module_ids_by_row=rows,owner_to_module_id=mapping,module_positions_by_id=positions,
            owner_rewards={owner:value.clone() for owner,value in self.owner_scalar_rewards.items()},
            next_module_ids_by_row=rows,next_owner_to_module_id=mapping)
