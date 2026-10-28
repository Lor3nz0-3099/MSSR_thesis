"""Physical C1 TRAIN calibration and fixed-alpha altruistic PPO entry point.

Each episode owns its ROS/Isaac process group. Expert assembles once; verified
handoff closes Expert before lockstep MARL. No synthetic rows enter this runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
from types import SimpleNamespace
from uuid import uuid4

STOCHASTIC_LOG_STD_INIT = -2.
# Layout order only: independent of actor sampling and the frozen generators.
TRAIN_LAYOUT_POOL_SEED = 0
DOMAINS = ('navigation', 'stairs', 'gap', 'manipulation')
MORPHOLOGIES = dict(navigation='rc_car8', stairs='snake8', gap='snake8', manipulation='mobile_manipulator8')
BC_SHA256 = 'd90a73e9b5ba2ecb579e4fc8733066b1c55d9e60dafeb1d71c029c49a7c21c6d'
CHECKPOINT = 'artifacts/bc_v1/runs/20261022T232532_854802073Z_207309_bc_v4_actor_visible_r1m_scratch_final/best.pt'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


class TransitionMix:
    """Deficit scheduler whose accounting unit is one completed transition."""
    def __init__(self, weights=None):
        weights = weights or dict(navigation=.7, stairs=.1, gap=.1, manipulation=.1)
        self.weights = {d: float(weights[d]) for d in DOMAINS}
        if any(not math.isfinite(w) or w <= 0 for w in self.weights.values()):
            raise ValueError('transition mixture weights must be finite and positive')
        total = sum(self.weights.values())
        self.weights = {d: w/total for d,w in self.weights.items()}
        self.counts = Counter()

    def next_domain(self):
        total = sum(self.counts.values()) + 1
        return max(DOMAINS, key=lambda d: total*self.weights[d]-self.counts[d])

    def record(self, domain, transitions):
        self.counts[domain] += transitions


class TrainingLayoutPool:
    """Traverse each domain's reproducibly shuffled geometry pool in order."""
    def __init__(self,layouts):
        self.pools={domain:[row for row in layouts if row['domain']==domain] for domain in DOMAINS}
        self.next_index=Counter()

    def next_layout(self,domain):
        choices=self.pools.get(domain,())
        if not choices:
            raise ValueError('train requires all four domains for 70/10/10/10 transition mixture')
        layout=choices[self.next_index[domain]%len(choices)]
        self.next_index[domain]+=1
        return layout


def runtime_command(args, runtime_dir, domain, seed, split):
    command = ['ros2', 'launch', 'mssr_expert', 'smores_runtime.launch.py',
        f'repository_root:={args.repository_root}', f'runtime_dir:={runtime_dir}',
        'module_count:=8', 'clean_runtime:=true', f'headless:={str(not args.gui).lower()}',
        'performance:=true', 'simple_visuals:=true', 'physics_hz:=240', 'state_publish_hz:=30',
        'simulation_speed_factor:=1.0', 'actuator_effort_scale:=4.0', 'tilt_effort_scale:=8.0',
        'wheel_friction_scale:=1.50', 'simulation_steps:=300000',
        f'ros_domain_id:={os.environ.get("ROS_DOMAIN_ID", "0")}',
        f'rmw_implementation:={os.environ.get("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")}']
    prefix = dict(navigation='phase1_marl',stairs='phase1_stairs',gap='phase1_gap',
                  manipulation='phase1_manipulation')[domain]
    command.extend([f'{prefix}_course:=true',f'{prefix}_seed:={seed}',f'{prefix}_split:={split}'])
    return command


class LockstepTransport:
    """Environment-indexed native BC lockstep transport.

    A scalar directory/session remains supported for the historical
    single-environment C1 runtime. Sequences provide one independent
    native mailbox/session per vector environment.
    """

    def __init__(
        self,
        directory,
        session,
        synchronizer,
        *,
        step_ticks,
    ):
        if isinstance(
            directory,
            (
                str,
                Path,
            ),
        ):
            directories = (
                Path(directory),
            )
        else:
            directories = tuple(
                Path(item)
                for item
                in directory
            )

        if not directories:
            raise ValueError(
                "LockstepTransport requires at least one runtime directory"
            )

        if isinstance(
            session,
            str,
        ):
            sessions = (
                session,
            )
        else:
            sessions = tuple(
                str(item)
                for item
                in session
            )

        if len(sessions) != len(
            directories
        ):
            raise ValueError(
                "runtime directory/session batch size mismatch"
            )

        if any(
            not value
            for value
            in sessions
        ):
            raise ValueError(
                "native BC sessions must be nonempty"
            )

        self.directories = directories
        self.sessions = sessions
        self.num_envs = len(
            directories
        )

        # Historical single-env aliases.
        self.directory = directories[0]
        self.session = sessions[0]

        self.synchronizer = synchronizer
        self.step_ticks = step_ticks

        self._epochs = [
            None
            for _
            in range(
                self.num_envs
            )
        ]

        self._sequences = [
            0
            for _
            in range(
                self.num_envs
            )
        ]

        self._staged = [
            ()
            for _
            in range(
                self.num_envs
            )
        ]

        self._stamps = [
            None
            for _
            in range(
                self.num_envs
            )
        ]

        self._local_rejections = [
            None
            for _
            in range(
                self.num_envs
            )
        ]

        # Keep legacy attributes meaningful for env 0.
        self.epoch = None
        self.sequence = 0
        self.staged = ()
        self.stamp = None
        self.local_rejection = None

    def _validate_env_index(
        self,
        env_index,
    ):
        if type(env_index) is not int:
            raise TypeError(
                "env_index must be an int"
            )

        if not (
            0
            <= env_index
            < self.num_envs
        ):
            raise IndexError(
                env_index
            )

        return env_index

    def status(
        self,
        env_index=0,
    ):
        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        value = read_json(
            self.directories[
                env_index
            ]
            / "bc_status.json"
        )

        if (
            value.get(
                "session"
            )
            != self.sessions[
                env_index
            ]
        ):
            raise RuntimeError(
                "native BC session identity mismatch"
            )

        epoch = value.get(
            "epoch"
        )

        if (
            self._epochs[
                env_index
            ]
            is None
        ):
            self._epochs[
                env_index
            ] = epoch

        elif (
            epoch
            != self._epochs[
                env_index
            ]
        ):
            raise RuntimeError(
                "native BC epoch identity mismatch"
            )

        if env_index == 0:
            self.epoch = (
                self._epochs[
                    0
                ]
            )

        if value.get(
            "failure"
        ):
            raise RuntimeError(
                f'native BC fail closed: {value["failure"]}'
            )

        return value

    def request(
        self,
        *,
        env_index=0,
        **payload,
    ):
        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        write_json(
            self.directories[
                env_index
            ]
            / "bc_request.json",
            dict(
                payload,
                session=self.sessions[
                    env_index
                ],
                epoch=self._epochs[
                    env_index
                ],
            ),
        )

    def arm(
        self,
        timeout,
        env_index=None,
    ):
        if env_index is None:
            env_indices = tuple(
                range(
                    self.num_envs
                )
            )
        else:
            env_indices = (
                self._validate_env_index(
                    env_index
                ),
            )

        for index in env_indices:
            self.request(
                env_index=index,
                kind="arm",
            )

        pending = set(
            env_indices
        )

        deadline = (
            time.monotonic()
            + timeout
        )

        while (
            pending
            and time.monotonic()
            < deadline
        ):
            for index in tuple(
                sorted(
                    pending
                )
            ):
                status = self.status(
                    index
                )

                if (
                    status.get(
                        "armed"
                    )
                    and status.get(
                        "last_ack",
                        {},
                    ).get(
                        "kind"
                    )
                    == "arm"
                    and status.get(
                        "published_stamp"
                    )
                    == status[
                        "last_ack"
                    ][
                        "sim_stamp"
                    ]
                ):
                    pending.remove(
                        index
                    )

            if pending:
                time.sleep(
                    0.01
                )

        if pending:
            raise TimeoutError(
                "native lockstep arm acknowledgement timeout: "
                + ", ".join(
                    f"env {index}"
                    for index
                    in sorted(
                        pending
                    )
                )
            )

    def stage(
        self,
        decoded,
        *,
        state_stamp,
        local_rejection=None,
        env_index=0,
    ):
        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        staged = tuple(
            decoded
        )

        stamp = float(
            state_stamp
        )

        self._staged[
            env_index
        ] = staged

        self._stamps[
            env_index
        ] = stamp

        self._local_rejections[
            env_index
        ] = local_rejection

        if env_index == 0:
            self.staged = staged
            self.stamp = stamp
            self.local_rejection = (
                local_rejection
            )

    def step(
        self,
        env_index,
    ):
        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        self._sequences[
            env_index
        ] += 1

        if env_index == 0:
            self.sequence = (
                self._sequences[
                    0
                ]
            )

        command_id = (
            f"bc:"
            f"{self.sessions[env_index]}:"
            f"{self._sequences[env_index]}"
        )

        continuous = []
        goals = []
        owners = set()

        staged = self._staged[
            env_index
        ]

        stamp = self._stamps[
            env_index
        ]

        if stamp is None:
            raise RuntimeError(
                f"env {env_index} has no staged state stamp"
            )

        for decoded in staged:
            payload = deepcopy(
                dict(
                    decoded.command
                )
            )

            name = payload[
                "command"
            ]

            module_id = payload[
                "module_ids"
            ][0]

            owners.update(
                payload[
                    "module_ids"
                ]
            )

            resource = {
                "wheel_velocity":
                    "locomotion",
                "pan_velocity":
                    "internal_motion",
            }.get(
                name
            )

            key = [
                module_id,
                resource
                or name,
            ]

            row = dict(
                key=key,
                generation=command_id,
                source_stamp=stamp,
                payload=payload,
            )

            (
                continuous
                if resource
                else goals
            ).append(
                row
            )

        before = len(
            self.status(
                env_index
            ).get(
                "rejections",
                (),
            )
        )

        self.synchronizer.expect_action(
            env_index,
            command_id,
        )

        self.request(
            env_index=env_index,
            kind="update",
            command_id=command_id,
            state_stamp=stamp,
            state_deadline=(
                time.monotonic()
                + 1.0
            ),
            step_ticks=self.step_ticks,
            continuous=continuous,
            goals=goals,
        )

        return SimpleNamespace(
            env_index=env_index,
            command_id=command_id,
            rejection_start=before,
            owner_ids=tuple(
                sorted(
                    owners
                )
            ),
            decoded=staged,
            local_rejection=(
                self._local_rejections[
                    env_index
                ]
            ),
            proposal_diagnostics=getattr(
                self,
                "proposal_diagnostics",
                {},
            ),
        )

    def completed_outcome(
        self,
        action,
    ):
        env_index = getattr(
            action,
            "env_index",
            0,
        )

        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        status = self.status(
            env_index
        )

        completed = (
            status.get(
                "last_completed"
            )
            or {}
        )

        if (
            completed.get(
                "command_id"
            )
            != action.command_id
        ):
            raise RuntimeError(
                "completed proposal identity mismatch"
            )

        if (
            action.local_rejection
            is not None
        ):
            return (
                action.local_rejection
            )

        rows = status.get(
            "rejections",
            (),
        )[
            action.rejection_start:
        ]

        if (
            completed[
                "outcome"
            ]
            == "rejected"
        ):
            codes = {
                row.get(
                    "code"
                )
                for row
                in rows
            }

            if (
                not codes.issubset(
                    {
                        "RESOURCE_BUSY",
                        "INVALID_GOAL",
                    }
                )
                or not codes
            ):
                raise RuntimeError(
                    "native rejection lacks reliable "
                    f"policy code: {rows}"
                )

            code = (
                "INVALID_GOAL"
                if "INVALID_GOAL"
                in codes
                else "RESOURCE_BUSY"
            )

            owners = {
                owner
                for row
                in rows
                for owner
                in row.get(
                    "owner_ids",
                    (),
                )
            }

            if not owners:
                raise RuntimeError(
                    "native rejection lacks owner attribution"
                )

            return SimpleNamespace(
                accepted=False,
                code=code,
                owner_ids=tuple(
                    sorted(
                        owners
                    )
                ),
            )

        return SimpleNamespace(
            accepted=True,
            code=(
                "EMPTY"
                if completed[
                    "outcome"
                ]
                == "empty"
                else "ACCEPTED"
            ),
            owner_ids=(
                action.owner_ids
            ),
        )

    def close(
        self,
        env_index=None,
    ):
        if env_index is None:
            env_indices = tuple(
                range(
                    self.num_envs
                )
            )
        else:
            env_indices = (
                self._validate_env_index(
                    env_index
                ),
            )

        for index in env_indices:
            self.request(
                env_index=index,
                kind="revoke",
                reason="C1 episode completed",
            )


def _connection_key(edge):
    a = (str(edge['module_a_id']), str(edge['connector_a_id']))
    b = (str(edge['module_b_id']), str(edge['connector_b_id']))
    return tuple(sorted((a,b)))


def attached_connections(raw):
    edges = raw.get('graph', {}).get('edges', raw.get('edges', ()))
    return frozenset(_connection_key(edge) for edge in edges if edge.get('is_attached', False))


def captured_world_positions(raw):
    """Validate measured world geometry before it can become reward evidence."""
    positions = {}
    modules = raw['state']['modules']
    for module in modules:
        value = module.get('position')
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError('captured module position must be finite XYZ')
        xyz=tuple(float(x) for x in value)
        if not all(math.isfinite(x) and abs(x)<=100. for x in xyz):
            raise RuntimeError('invalid world XYZ: nonfinite or outside the 100 m physical envelope')
        owner=module['module_id']
        if not isinstance(owner,str) or not owner or owner in positions:
            raise ValueError('captured geometry requires unique module owners')
        positions[owner] = xyz
    if len(positions)!=8:
        raise ValueError('captured physical geometry requires exactly eight module owners')
    return positions


def minimum_support_height_m(course):
    heights=[]
    for box in course.support_boxes:
        if not box.collidable:
            continue
        pitch=math.radians(float(box.pitch_deg))
        # Lowest corner of the actual top face, including a pitched road tile.
        heights.append(float(box.center_xyz_m[2])+.5*float(box.size_xyz_m[2])*math.cos(pitch)
                       -.5*float(box.size_xyz_m[0])*abs(math.sin(pitch)))
    if not heights or not all(math.isfinite(height) for height in heights):
        raise ValueError('course requires finite physical support geometry')
    return min(heights)


def irrecoverable_fall(course,raw):
    threshold=minimum_support_height_m(course)-.15
    return all(xyz[2]<threshold for xyz in captured_world_positions(raw).values())


def absorbing_failure_reward(domain,course,before,after,signals,gamma):
    from dataclasses import replace
    from drl.MARL.phase1_reward import compute_phase1_raw_reward
    from drl.MARL.stairs_reward import compute_stairs_raw_reward
    from drl.MARL.gap_reward import compute_gap_raw_reward
    from drl.MARL.manipulation_reward import compute_manipulation_raw_reward
    compute=dict(navigation=compute_phase1_raw_reward,stairs=compute_stairs_raw_reward,
                 gap=compute_gap_raw_reward,manipulation=compute_manipulation_raw_reward)[domain]
    raw=compute(course,before,after,signals=signals,shaping_gamma=gamma,terminated=True)
    # Navigation's ordinary success test uses XY. A fallen body over its goal
    # cannot turn this physical failure into success; the same precedence holds
    # for a retained button depression measurement after a whole-body fall.
    return replace(raw,team=replace(raw.team,success=0.),
                   owners={owner:(replace(terms,success=0.) if hasattr(terms,'success') else terms)
                           for owner,terms in raw.owners.items()})


def captured_observation(domain, raw, intended_connections, *, end_effector_id=None):
    """Reward-only adapter over the exact cached frame; never actor input."""
    positions=captured_world_positions(raw)
    modules=raw['state']['modules']
    if domain == 'navigation':
        return {mid: xyz[:2] for mid,xyz in positions.items()}
    if domain != 'manipulation':
        return positions
    from drl.MARL.manipulation_reward import ManipulationObservation
    ee = next((m for m in modules if m['module_id'] == end_effector_id), None)
    if ee is None:
        raise ValueError('verified end_effector module missing')
    bottom = next((p for p in ee['connectors'] if p['connector_id'] == 'BOTTOM'), None)
    if bottom is None:
        raise ValueError('physical end-effector BOTTOM face missing')
    button = raw['state']['course']['button']
    if 'depression_m' not in button:
        raise ValueError('live physical button depression missing')
    # Compare actual connector pairs with the independently verified handoff.
    # Seven unrelated attachments cannot satisfy intended MM8 integrity.
    integrity = len(attached_connections(raw).intersection(intended_connections))
    return ManipulationObservation(module_xyz_m=positions,
        end_effector_bottom_center_xyz_m=tuple(bottom['position_world']),
        end_effector_bottom_normal_world=tuple(bottom['outward_normal_world']),
        button_depression_m=float(button['depression_m']), connection_count=integrity)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('calibrate', 'smoke', 'train', 'evaluate'))
    parser.add_argument('--repository-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-root',type=Path,default=Path('/dev/shm/mssr-c1'),
                        help='Fast native mailboxes; bounded evidence is copied under output/runtime on close.')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reward-config', type=Path)
    parser.add_argument('--appo-checkpoint', type=Path)
    parser.add_argument('--domains', nargs='+', choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--steps-per-layout', type=int, default=16)
    parser.add_argument('--layouts-per-domain', type=int,
                        help='Optional domain cap; calibration defaults to the 24-layout pilot, train to full frozen TRAIN pools.')
    parser.add_argument('--step-ticks', type=int, default=8)
    parser.add_argument('--rollout-transitions', type=int, default=64)
    parser.add_argument('--total-transitions', type=int, default=100000)
    parser.add_argument('--alpha', type=float, default=.5)
    parser.add_argument('--altruism-radius-m', type=float, default=.5)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--startup-timeout-s', type=float, default=240.)
    parser.add_argument('--assembly-timeout-s', type=float, default=600.)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=2)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    args = parser.parse_args(argv)
    if args.command != 'evaluate' and args.split != 'train':
        parser.error('calibration and learning accept TRAIN only')
    if args.command == 'evaluate' and args.split != 'validation':
        parser.error('configuration comparison requires --split validation')
    if args.command in ('smoke','train','evaluate') and args.reward_config is None:
        parser.error('--reward-config is required')
    if not 1 <= args.step_ticks <= 240 or args.steps_per_layout <= 0:
        parser.error('positive steps and step-ticks within [1,240] required')
    if args.layouts_per_domain is not None and args.layouts_per_domain < 1:
        parser.error('--layouts-per-domain must be positive')
    args.repository_root = args.repository_root.resolve()
    args.checkpoint = args.checkpoint or args.repository_root/CHECKPOINT
    return args


class InferenceDecisionBuilder:
    """Only the existing no-grad rollout decision method, without a learner."""
    def __init__(self, *, actor, critic, owners, device):
        from drl.MARL.mappo import MSSRMAPPO
        self.shared_actor, self.shared_critic = actor, critic
        self.possible_agents, self.device = tuple(owners), device
        self.forward_shared_actor = MSSRMAPPO.forward_shared_actor

    def build_online_rollout_decision(self, **kwargs):
        from drl.MARL.mappo import MSSRMAPPO
        return MSSRMAPPO.build_online_rollout_decision(self, **kwargs)


class C1RuntimeContext:
    """One synchronized BC frame per state, independent frozen graph snapshots."""
    num_envs = 1

    def __init__(self, episode, source, bridge):
        from drl.IL.bc_v1.live_policy import smores_claims
        from drl.MARL.smores_runtime_binding import SMORESMARLRuntimeBinding
        self.episode, self.source, self.bridge = episode, source, bridge
        self.binding = SMORESMARLRuntimeBinding(adapter=bridge.adapter, claim_resolver=smores_claims)
        self.context = None

    def current_action_contexts(self):
        import torch
        from drl.MARL.mappo import OnlineRolloutContext
        prepared = self.source.prepare_env(0)
        if prepared.privileged_graph is None:
            raise RuntimeError('privileged current critic tensorization missing')
        route = prepared.routing_snapshot
        # C1 fixes assembled morphology. C2/C3 structural reconfiguration is
        # outside this action space; resource candidates remain stochastic.
        hard_mask = torch.tensor([False]*len(route['resource_candidates']) +
                                 [True]*len(route['structural_bindings']), device=self.episode.args.device)
        self.context = OnlineRolloutContext(actor_graph=deepcopy(prepared.actor_graph),
            critic_graph=deepcopy(prepared.privileged_graph), routing_snapshot=deepcopy(route), hard_mask=hard_mask)
        return [self.context]

    def encode_env_actions(self, decisions):
        import torch
        decision = decisions[0]
        prepared = self.source.prepare_env(0)
        local_rejection = None
        try:
            decoded = self.binding.decode_marl_action(action=decision.action,
                prediction_template=decision.action_set_prediction_template,
                routing=self.context.routing_snapshot, batch=prepared.decoder_batch,
                normalizers=self.bridge.normalizers)
        except ValueError as error:
            # Invalid sampled Gaussian payloads and overlapping sampled claims
            # remain policy outcomes. Do not repair/resample the proposal.
            message = str(error)
            expected = ('Physical payload outside limits', 'Conflicting physical resource claims',
                        'Physical resource unavailable')
            if not any(token in message for token in expected):
                raise
            code = 'RESOURCE_BUSY' if 'claims' in message or 'unavailable' in message else 'INVALID_GOAL'
            if code=='RESOURCE_BUSY':
                owners=sampled_resource_conflict_owners(decision.action,self.context.routing_snapshot)
                if not owners:
                    raise RuntimeError('sampled conflict has no attributable competing resource candidates') from error
            else:
                # Locate invalid payloads by independently decoding each selected
                # resource for diagnostics. The original complete proposal stays
                # rejected; none of these diagnostic commands are dispatched.
                from dataclasses import replace
                owners=set()
                for i,candidate in enumerate(self.context.routing_snapshot['resource_candidates']):
                    if not decision.action.presence[i]:
                        continue
                    presence=torch.zeros_like(decision.action.presence)
                    presence[i]=1
                    try:
                        self.binding.decode_marl_action(action=replace(decision.action,presence=presence),
                            prediction_template=decision.action_set_prediction_template,
                            routing=self.context.routing_snapshot,batch=prepared.decoder_batch,
                            normalizers=self.bridge.normalizers)
                    except ValueError as candidate_error:
                        if 'Physical payload outside limits' not in str(candidate_error):
                            raise
                        owners.add(candidate.resource_ref.module.module_id)
                if not owners:
                    raise RuntimeError('sampled invalid payload has no attributable candidate') from error
                owners=tuple(sorted(owners))
            local_rejection = SimpleNamespace(accepted=False, code=code, owner_ids=owners, message=message)
            decoded = ()
        selected=[]
        for i,candidate in enumerate(self.context.routing_snapshot['resource_candidates']):
            if decision.action.presence[i]:
                selected.append(dict(candidate_index=i,owner=candidate.resource_ref.module.module_id,
                    resource=candidate.resource_ref.resource_id,action=candidate.action.action_id,
                    mode=candidate.mode.mode_id))
        self.episode.transport.proposal_diagnostics=dict(sampled_presence=decision.action.presence.detach().cpu().tolist(),
            presence_logits=decision.action_set_prediction_template['presence_logits'].detach().cpu().tolist(),
            sampled_resource_numeric={str(key):value.detach().cpu().tolist() for key,value in decision.action.resource_numeric.items()},
            sampled_resource_categorical={str(key):value.detach().cpu().tolist() for key,value in decision.action.resource_categorical.items()},
            selected_resources=selected,decoded_commands=[dict(item.command) for item in decoded],
            local_rejection=None if local_rejection is None else vars(local_rejection))
        self.episode.transport.stage(decoded, state_stamp=prepared.physical_state.stamp,
                                     local_rejection=local_rejection)
        return {owner:torch.zeros((1,1), device=self.episode.args.device) for owner in self.episode.owners}

    def current_critic_graphs(self):
        prepared = self.source.prepare_env(0)
        if prepared.privileged_graph is None:
            raise RuntimeError('privileged next critic tensorization missing')
        return [deepcopy(prepared.privileged_graph)]


def sampled_resource_conflict_owners(action,routing):
    claims=Counter()
    for i,candidate in enumerate(routing['resource_candidates']):
        if action.presence[i]:
            ref=candidate.resource_ref
            claims[(ref.module.module_id,ref.resource_id)]+=1
    return tuple(sorted({owner for (owner,resource),count in claims.items() if count>1}))


def build_observation_bridge(root):
    from drl.IL.bc_v1.source_frame_bridges import BridgeResolution, SourceFrameBridgeBatch
    from drl.MARL.bc_v4_runtime_observation import BCV4RuntimeObservationBridge
    from mssr_expert.graph.graph_builder import GraphBuilder
    bridges = SourceFrameBridgeBatch((BridgeResolution('world','world','current-live',None,True,
        'current publisher WORLD contract',None,()),),(('provider_id','mssr-c1-physical'),('revision','1')))
    return BCV4RuntimeObservationBridge.from_repository(repo_root=root, graph_builder=GraphBuilder(), bridges=bridges)


def build_models(args, bridge):
    from drl.MARL.bc_v4_actor import BCV4SharedActorWrapper
    from drl.MARL.global_critic import GlobalCritic
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != BC_SHA256:
        raise ValueError('BC-v4 SHA256 differs from frozen reference')
    actor = BCV4SharedActorWrapper.from_repository_checkpoint(spec=bridge.spec,
        feature_schema=bridge.feature_schema, repo_root=args.repository_root,
        checkpoint_path=args.checkpoint, expected_sha256=BC_SHA256,
        map_location=args.device, device=args.device,
        resource_log_std_init=STOCHASTIC_LOG_STD_INIT, structural_log_std_init=STOCHASTIC_LOG_STD_INIT)
    critic = GlobalCritic.from_repository_config(spec=bridge.spec, feature_schema=bridge.feature_schema,
        repo_root=args.repository_root, device=args.device)
    actor.to(args.device).eval()
    critic.to(args.device).eval()
    return actor, critic


def physical_task_view(domain, course, config):
    """Only recorded physical geometry enters the approved BC task parser."""
    from drl.IL.bc_v1.task_context import build_task_view
    course = deepcopy(course)
    if domain=='navigation' and isinstance(course.get('goal'),dict) and 'center_xyz_m' in course['goal']:
        course['current_goal'] = dict(position_xyz_m=course['goal']['center_xyz_m'])
    physical = course.get('physical_track',{})
    if domain=='navigation' and 'goal_xyyaw' in physical:
        course['current_goal'] = dict(xyyaw=physical['goal_xyyaw'])
    return build_task_view({'task_type':'button' if domain=='manipulation' else domain},
                           {'course':course},config)


def make_course(domain, seed, split):
    from smores_ep.isaac.phase1_marl_curriculum import phase1_marl_curriculum_course
    from smores_ep.isaac.phase1_stairs_curriculum import phase1_stairs_curriculum_course
    from smores_ep.isaac.phase1_gap_curriculum import phase1_gap_curriculum_course
    from smores_ep.isaac.phase1_manipulation_curriculum import phase1_manipulation_curriculum_course
    return dict(navigation=phase1_marl_curriculum_course, stairs=phase1_stairs_curriculum_course,
        gap=phase1_gap_curriculum_course, manipulation=phase1_manipulation_curriculum_course)[domain](seed, split=split)


def terminate_process_group(process, *, term_timeout_s=10., kill_timeout_s=5.):
    """Only this episode's newly created group; TERM precedes escalation."""
    def live_members():
        result = []
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path/'stat').read_text().split(') ',1)[1].split()
                if fields[0] != 'Z' and int(fields[2]) == process.pid:
                    result.append(int(path.name))
            except (OSError,ValueError,IndexError):
                continue
        return result

    # A ros2 launch leader can exit while its Isaac child remains alive.
    # Group membership, rather than leader.poll(), owns cleanup authority.
    for sig, timeout in ((signal.SIGTERM,term_timeout_s), (signal.SIGKILL,kill_timeout_s)):
        if not live_members():
            process.poll()
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic()+timeout
        while live_members() and time.monotonic() < deadline:
            process.poll()
            time.sleep(.02)
    survivors = live_members()
    if survivors:
        raise RuntimeError(f'owned process-group survivors after KILL: {survivors}')
    process.poll()


def copy_runtime_evidence(source,destination,*,max_log_bytes=2*1024*1024):
    source,destination=Path(source),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    if source.resolve()==destination.resolve():
        return
    files=[]
    for name in ('bc_status.json','bc_request.json','state_graph.json','robot_graph.json',
                 'module_states.json','self_assembly_state.json','actions.json','invalid_physical_frame.json'):
        path=source/name
        if path.is_file():
            shutil.copyfile(path,destination/name)
            files.append(name)
    truncated=False
    for name in ('runtime.log','expert.log','self_assembly.log','marl_assembly_expert.log'):
        path=source/name
        if path.is_file():
            size=path.stat().st_size
            with path.open('rb') as handle:
                handle.seek(max(0,size-max_log_bytes))
                (destination/name).write_bytes(handle.read(max_log_bytes))
            truncated|=size>max_log_bytes
            files.append(name)
    write_json(destination/'evidence.json',dict(source_runtime=str(source),files=files,
        max_log_bytes=max_log_bytes,log_tail_truncated=truncated))


class PhysicalEpisode:
    def __init__(self, args, domain, seed, episode_id, bridge):
        self.args, self.domain, self.seed, self.episode_id = args, domain, seed, episode_id
        self.bridge,self.directory=bridge,args.runtime_root.resolve()/episode_id
        self.evidence_directory=args.output/'runtime'/episode_id
        self._owned_directory=False
        self.process = self.binding = self.transport = self.log = None
        self.course = make_course(domain, seed, args.split)
        self.intended_connections, self.end_effector_id = frozenset(), None
        self.last_signals = None

    def wait(self, predicate, timeout, label):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f'physical runtime exited: {self.directory/"runtime.log"}')
            try:
                value = predicate()
                if value:
                    return value
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(.05)
        raise TimeoutError(label)

    def __enter__(self):
        try:
            return self.open()
        except BaseException:
            self.close()
            raise

    def open(self):
        from drl.MARL.direct_marl_env import DirectMARLEnv
        from drl.MARL.isaac_control_step import FileBackedIsaacControlStep
        from drl.MARL.phase1_bc_v4_observation import SynchronizedBCV4ObservationSource
        from drl.MARL.runtime_handoff import DirectMARLHandoff, RuntimePhase
        from drl.MARL.runtime_signals import StagnationDetector
        from drl.MARL.smores_expert_runtime_binding import (SMORESExpertRuntimeBinding,
            make_expert_state_file_source, make_robot_graph_file_source)
        from mssr_expert.graph.serialization import load_attributed_graph
        from mssr_expert.planning.smores_ep.self_reconfiguration_planner import SmoresSelfReconfigurationPlanner
        self.directory.mkdir(parents=True, exist_ok=False)
        self._owned_directory=True
        self.session = uuid4().hex
        envvars = dict(os.environ, MSSR_BC_SESSION=self.session, MSSR_C1_LOCKSTEP='1')
        # ROS uses the launch executable's system Python, not this learner venv.
        self.log = (self.directory/'runtime.log').open('w')
        command = runtime_command(self.args, self.directory, self.domain, self.seed, self.args.split)
        print('RUNTIME_LAUNCH: '+' '.join(command), flush=True)
        self.process = subprocess.Popen(command, cwd=self.args.repository_root, env=envvars,
            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.wait(lambda: (self.directory/'robot_graph.json').is_file() and (self.directory/'bc_status.json').is_file(),
                  self.args.startup_timeout_s, 'physical runtime startup timeout')
        config = self.args.repository_root/'mssr_ws/src/mssr_expert/config'
        morphology = MORPHOLOGIES[self.domain]
        target = load_attributed_graph(config/f'smores_{morphology}.json')
        graph_source = make_robot_graph_file_source(self.directory)
        self.matcher = SmoresSelfReconfigurationPlanner()
        self.binding = SMORESExpertRuntimeBinding(repository_root=self.args.repository_root,
            runtime_dirs=(self.directory,),expert_state_sources=(make_expert_state_file_source(self.directory),),
            current_graph_sources=(graph_source,),morphology_catalog={morphology:target},
            topology_matcher=self.matcher, episode_id_factory=lambda index:self.episode_id)
        self.synchronizer = FileBackedIsaacControlStep(runtime_dirs=(self.directory,), timeout_s=10.,
            application_source=lambda index:read_json(self.directory/'bc_status.json'))
        self.transport = LockstepTransport(self.directory,self.session,self.synchronizer,step_ticks=self.args.step_ticks)
        self.handoff = DirectMARLHandoff()
        self.env = DirectMARLEnv(handoffs=(self.handoff,), expert_launcher=self.binding.launch_expert,
            expert_status_reader=self.binding.expert_status, topology_observer=self.binding.observe_topology,
            actuation_bridge=self.transport,target_morphology=morphology,
            state_reader=self.state_reader,physics_step=self.synchronizer.physics_step,
            transition_evaluator=self.evaluate_transition)
        self.env.reset()
        def assembled():
            phase = self.env.advance_handoff(0)
            if phase is RuntimePhase.FAILED:
                raise RuntimeError('Expert assembly/topology verification failed')
            return self.handoff.marl_active
        self.wait(assembled,self.args.assembly_timeout_s,'Expert assembly/topology handoff timeout')
        # Closing the assembly process is mandatory before native MARL arm.
        self.binding.close()
        self.transport.arm(self.args.startup_timeout_s)
        self.synchronizer.state_reader(0)
        raw = self.synchronizer.cached_raw_snapshot(0)
        self.owners = tuple(sorted(m['module_id'] for m in raw['state']['modules']))
        self.intended_connections = attached_connections(raw)
        if len(self.intended_connections) != 7:
            raise RuntimeError('verified morphology must have exactly seven intended physical connections')
        assignment = self.matcher.configuration_assignment(graph_source(),target)
        if assignment is None:
            raise RuntimeError('physical topology changed at native handoff')
        from mssr_expert.planning.smores_ep.attributed_adapter import target_roles_from_graph
        roles = target_roles_from_graph(target)
        for vertex, module_id in assignment.target_to_module.items():
            if roles[vertex].get('target_role') == 'end_effector':
                self.end_effector_id = module_id
        self.stagnation = StagnationDetector()
        self.task_config = read_json(self.args.repository_root/'configs/bc_v1/task_context.json')
        self.source = SynchronizedBCV4ObservationSource(frame_synchronizer=self.synchronizer,
            observation_bridge=self.bridge, task_context_provider=lambda index:self.task_view()[0],
            observation_provider=lambda index:self.task_view()[1])
        self.context = C1RuntimeContext(self,self.source,self.bridge)
        print(f'MARL_HANDOFF_VERIFIED: {self.episode_id} {self.owners}',flush=True)
        return self

    def state_reader(self, env_index):
        self.synchronizer.state_reader(env_index)
        raw=self.synchronizer.cached_raw_snapshot(env_index)
        try:
            captured_world_positions(raw)
        except (RuntimeError,ValueError) as error:
            write_json(self.directory/'invalid_physical_frame.json',dict(reason=str(error),
                calibration_row=False,raw_snapshot_text=json.dumps(raw,allow_nan=True)))
            raise
        return raw

    def task_view(self):
        raw = self.synchronizer.cached_raw_snapshot(0)
        course = raw['state']['course']
        # Approved task parser whitelists physical features. Route/centerline,
        # reward-only geometry and future morphology are never actor inputs.
        return physical_task_view(self.domain,course,self.task_config)

    def evaluate_transition(self, **kwargs):
        from drl.MARL.runtime_signals import collect_runtime_signals
        from drl.MARL.phase1_transition import Phase1TransitionEvaluator
        from drl.MARL.stairs_transition import StairsTransitionEvaluator
        from drl.MARL.gap_transition import GapTransitionEvaluator
        from drl.MARL.manipulation_transition import ManipulationTransitionEvaluator
        pre, post = kwargs['previous_state'], kwargs['next_state']
        before = captured_observation(self.domain, pre,self.intended_connections,end_effector_id=self.end_effector_id)
        after = captured_observation(self.domain, post,self.intended_connections,end_effector_id=self.end_effector_id)
        outcome = self.transport.completed_outcome(kwargs['actuation_result'])
        positions = lambda raw:{m['module_id']:tuple(m['position']) for m in raw['state']['modules']}
        self.last_signals = collect_runtime_signals(owner_ids=self.owners,
            delta_seconds=post['stamp']-pre['stamp'],proposal_outcome=outcome,
            proposal_source=('sampled_action_set_executability_guard' if kwargs['actuation_result'].local_rejection
                             else 'native_bc_completed_application'),stagnation_detector=self.stagnation,
            previous_module_positions_m=positions(pre),next_module_positions_m=positions(post))
        signal_provider = lambda *a,**k:self.last_signals.signals
        if self.domain == 'manipulation':
            evaluator = ManipulationTransitionEvaluator(course=self.course,signal_builder=signal_provider,shaping_gamma=self.args.gamma)
        else:
            cls = dict(navigation=Phase1TransitionEvaluator,stairs=StairsTransitionEvaluator,gap=GapTransitionEvaluator)[self.domain]
            evaluator = cls(courses=(self.course,),max_episode_steps=self.args.steps_per_layout,
                            signals_provider=signal_provider,shaping_gamma=self.args.gamma)
        evaluation = evaluator(**dict(kwargs, previous_state=before,next_state=after))
        result = dict(vars(evaluation))
        if irrecoverable_fall(self.course,post):
            result.update(raw_reward=absorbing_failure_reward(self.domain,self.course,before,after,
                self.last_signals.signals,self.args.gamma),terminated=True,truncated=False,
                success=False,failure_reason='irrecoverable_fall')
        result['truncated'] = bool(result.get('truncated',False) or
                                  (not result['terminated'] and kwargs['episode_step']>=self.args.steps_per_layout))
        terms = asdict(result['raw_reward'].team)
        self.availability = dict(self.last_signals.availability)
        for term,value in terms.items():
            if term not in self.availability:
                self.availability[term] = dict(status='measured_present' if value else 'measured_absent',
                    units='raw_task_term',source='completed_transition_task_evaluator',attribution='owner_and_team')
        return SimpleNamespace(**result)

    def close(self):
        if self.transport is not None:
            try:
                self.transport.close()
            except (OSError,ValueError,RuntimeError):
                pass
        try:
            if self.binding is not None:
                self.binding.close()
        finally:
            if self.process is not None:
                terminate_process_group(self.process)
            if self.log is not None:
                self.log.close()
            if self._owned_directory:
                copy_runtime_evidence(self.directory,self.evidence_directory)
                if self.directory.resolve()!=self.evidence_directory.resolve():
                    shutil.rmtree(self.directory)

    def __exit__(self,*args):
        self.close()


def calibration_policy(index, count):
    if count >= 3 and index == count-1:
        return 'bc_v4_masked_perturbation'
    if count >= 3 and index == count-2:
        return 'bc_v4_deterministic_reference'
    return 'bc_v4_stochastic'


def calibration_collection_label(command):
    return 'bc_v4_stochastic_reference_perturbation' if command=='calibrate' else 'bc_v4_stochastic'


class CalibrationDecisionBuilder(InferenceDecisionBuilder):
    policy = 'bc_v4_stochastic'

    def build_online_rollout_decision(self, **kwargs):
        import torch
        from dataclasses import replace
        from drl.MARL.policy import MARLActionBundle, compute_owner_action_log_probs
        decision = super().build_online_rollout_decision(**kwargs)
        if self.policy == 'bc_v4_stochastic':
            return decision
        with torch.no_grad():
            output = self.forward_shared_actor(actor=self.shared_actor,graph=kwargs['actor_graph'],
                possible_agents=self.possible_agents,routing=kwargs['routing_snapshot'])
            records, payloads = output['records'], output['resource_payloads']
            presence = torch.stack([record.presence_logit >= 0 for record in records]).float()
            presence[kwargs['hard_mask']] = 0
            if self.policy == 'bc_v4_masked_perturbation':
                # Small valid/masked diagnostic perturbation of BC membership;
                # no geometry or expert command is substituted.
                eligible = torch.nonzero(~kwargs['hard_mask']).flatten()
                if len(eligible):
                    selected = eligible[torch.randint(len(eligible),(1,)).item()]
                    presence[selected] = 1-presence[selected]
            numeric, categorical = {}, {}
            for record in records:
                i = record.candidate_index
                if record.candidate_kind != 'resource' or not presence[i]:
                    continue
                payload = payloads.get(i)
                if payload is None:
                    continue
                if payload.numeric is not None:
                    numeric[i] = payload.numeric.detach().clone()
                if payload.categorical_logits is not None:
                    categorical[i] = payload.categorical_logits.argmax().detach()
            action = MARLActionBundle(presence,numeric,categorical,{},{},{},{},{},{})
            logs = compute_owner_action_log_probs(records=records,resource_payloads=payloads,
                structural_payloads=output['structural_payloads'],action=action,hard_mask=kwargs['hard_mask'],
                possible_agents=self.possible_agents,resource_log_std=output['resource_log_std'],
                structural_log_std=output['structural_log_std'])
        return replace(decision,action=action,owner_log_probs=logs,
            structured_sample=replace(decision.structured_sample,action=action))


def finite_tensors(value, label='runtime'):
    import torch
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise ValueError(f'nonfinite {label}')
    elif isinstance(value,dict):
        for key,item in value.items():
            finite_tensors(item,f'{label}.{key}')
    elif isinstance(value,(tuple,list)):
        for item in value:
            finite_tensors(item,label)


def append_jsonl(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value,allow_nan=False,default=str)+'\n')


def scalar_outcome_fields(evaluation):
    return dict(success=bool(evaluation.raw_reward.team.success),
                failure_reason=getattr(evaluation,'failure_reason',None))


def completed_episode_outcome(transition,*,episode_id,domain,seed,split,episode_start_stamp):
    """Describe an episode boundary using the already completed transition."""
    from drl.MARL.direct_marl_env import DirectMARLTransition
    if not isinstance(transition,DirectMARLTransition):
        raise TypeError('episode outcome requires a completed DirectMARLTransition')
    evaluation=transition.evaluation
    terminated,truncated=bool(evaluation.terminated),bool(evaluation.truncated)
    if not (terminated or truncated):
        raise ValueError('episode outcome requires termination or truncation')
    final_stamp=float(transition.next_state['stamp'])
    start_stamp=float(episode_start_stamp)
    if not math.isfinite(final_stamp) or not math.isfinite(start_stamp) or final_stamp<=start_stamp:
        raise ValueError('episode outcome requires finite increasing physical stamps')
    return dict(schema='mssr.c1_episode_outcome.v1',episode_id=episode_id,env_index=transition.env_index,
        domain=domain,seed=seed,split=split,length_steps=transition.episode_step,
        length_seconds=final_stamp-start_stamp,start_stamp=start_stamp,next_stamp=final_stamp,
        terminated=terminated,truncated=truncated,**scalar_outcome_fields(evaluation))


def aggregate_episode_outcomes(rows):
    result={domain:dict(episodes=0,success_count=0,failure_count=0,truncation_count=0,success_rate=None)
            for domain in DOMAINS}
    for row in rows:
        summary=result[row['domain']]
        summary['episodes']+=1
        summary['success_count']+=int(row['success'])
        summary['failure_count']+=int(row['terminated'] and not row['success'])
        summary['truncation_count']+=int(row['truncated'])
    for summary in result.values():
        if summary['episodes']:
            summary['success_rate']=summary['success_count']/summary['episodes']
    return result


def episode_manifest(args, run_id, episode, policy):
    from drl.MARL.reward_calibration import EpisodeManifest, curriculum_identity
    identity = curriculum_identity(episode.domain,episode.seed)
    base = subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repository_root,text=True).strip()
    dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=args.repository_root,text=True).strip())
    return EpisodeManifest(run_id=run_id,episode_id=episode.episode_id,domain=episode.domain,
        morphology=MORPHOLOGIES[episode.domain],seed=episode.seed,split=args.split,
        difficulty=identity.difficulty,family=getattr(identity,'family',None),collection_policy=policy,
        checkpoint_path=str(args.checkpoint.resolve()),checkpoint_sha256=BC_SHA256,
        base_commit=base,working_tree_dirty=dirty,physics_hz=240.,decision_hz=240./args.step_ticks)


def make_rollout_memory(**kwargs):
    from skrl.memories.torch import RandomMemory

    class PopulatedRandomMemory(RandomMemory):
        """Real skrl storage exposing only completed vector steps to GAE."""
        def get_tensor_by_name(self, name):
            if self.env_index:
                raise RuntimeError('GAE requires a completed vector memory step')
            return super().get_tensor_by_name(name)[:len(self)//self.num_envs]

    return PopulatedRandomMemory(**kwargs)


def create_learner(args, bridge, actor, critic, owners):
    import torch
    from drl.MARL.mappo import MSSRMAPPO
    from drl.MARL.owner_critic import build_repository_altruistic_critic_pair
    memory_size = max(args.rollout_transitions,args.steps_per_layout*len(args.domains))
    memories = {owner:make_rollout_memory(memory_size=memory_size,num_envs=1,device=args.device,replacement=False) for owner in owners}
    for memory in memories.values():
        for name in ('observations','states','actions','rewards','log_prob','values'):
            memory.create_tensor(name=name,size=1,dtype=torch.float32)
        for name in ('terminated','truncated'):
            memory.create_tensor(name=name,size=1,dtype=torch.bool)
    cfg = dict(learning_epochs=1,mini_batches=1,rollouts=memory_size,
        learning_rate=(1e-5,3e-4),discount_factor=args.gamma,gae_lambda=.95,entropy_loss_scale=.001,
        learning_rate_scheduler=None,
        **{name:{owner:{} for owner in owners} for name in ('learning_rate_scheduler_kwargs',
            'observation_preprocessor_kwargs','state_preprocessor_kwargs','value_preprocessor_kwargs')})
    learner = MSSRMAPPO(possible_agents=owners,models=MSSRMAPPO.build_shared_model_registry(
        possible_agents=owners,actor=actor,critic=critic),memories=memories,device=args.device,cfg=cfg)
    pair = build_repository_altruistic_critic_pair(spec=bridge.spec,feature_schema=bridge.feature_schema,
        repo_root=args.repository_root,device=args.device)
    learner.configure_fixed_alpha_appo(owner_critic=pair.owner,mean_field_critic=pair.mean_field,
                                      alpha=args.alpha)
    learner.shared_actor.eval()
    learner.shared_critic.eval()
    learner.altruistic_owner_critic.eval()
    learner.altruistic_mean_field_critic.eval()
    provider = SimpleNamespace(payload=None)
    provider.current_altruistic_rollout_batch = lambda:provider.payload
    learner.set_altruistic_rollout_provider(provider,radius_m=args.altruism_radius_m)
    return learner,provider



def record_learning_batch(
    args,
    learner,
    provider,
    step,
    scalar_rewards_by_env,
):
    """Record one active vector batch through the existing APPO/MAPPO path.

    MAPPO/skrl fixed-shape tensors remain the storage infrastructure.
    Owner rewards and per-environment geometry are preserved separately
    in AltruisticRolloutBatchPayload for fixed-alpha APPO.
    """
    import torch

    from drl.MARL.altruistic_rollout import (
        AltruisticRolloutBatchPayload,
    )

    active_env_ids = tuple(
        int(env_index)
        for env_index
        in step.active_env_indices
    )

    if not active_env_ids:
        raise ValueError(
            "record_learning_batch requires at least one active environment"
        )

    owners = tuple(
        learner.possible_agents
    )

    batch_size = len(
        active_env_ids
    )

    device = args.device

    decisions = [
        step.decisions[
            env_index
        ]
        for env_index
        in active_env_ids
    ]

    transitions = [
        step.transitions[
            env_index
        ]
        for env_index
        in active_env_ids
    ]

    contexts = [
        step.contexts[
            env_index
        ]
        for env_index
        in active_env_ids
    ]

    next_critic_graphs = [
        step.next_critic_graphs[
            env_index
        ]
        for env_index
        in active_env_ids
    ]

    rewards = []

    for env_index in active_env_ids:
        if env_index not in scalar_rewards_by_env:
            raise KeyError(
                f"missing scalar reward for active env {env_index}"
            )

        rewards.append(
            scalar_rewards_by_env[
                env_index
            ]
        )

    module_ids_by_row = []
    owner_to_module_id = []
    module_positions_by_id = []

    for context, transition in zip(
        contexts,
        transitions,
    ):
        route = (
            context.routing_snapshot
        )

        rows = tuple(
            ref.module_id
            for ref
            in route[
                "module_refs"
            ]
        )

        module_ids_by_row.append(
            rows
        )

        # Current C1 owner IDs are the physical module IDs.
        # Keep this explicit in the altruistic sidecar so each
        # vector environment retains its own mapping.
        owner_to_module_id.append(
            {
                owner: owner
                for owner
                in owners
            }
        )

        raw = (
            transition
            .previous_state
        )

        positions = {
            module["module_id"]:
                torch.tensor(
                    module[
                        "position"
                    ],
                    device=device,
                )
            for module
            in raw[
                "state"
            ][
                "modules"
            ]
        }

        module_positions_by_id.append(
            positions
        )

    owner_rewards = {
        owner:
            torch.tensor(
                [
                    [
                        float(
                            reward
                            .owners[
                                owner
                            ]
                        )
                    ]
                    for reward
                    in rewards
                ],
                device=device,
            )
        for owner
        in owners
    }

    provider.payload = (
        AltruisticRolloutBatchPayload(
            module_ids_by_row=tuple(
                module_ids_by_row
            ),
            owner_to_module_id=tuple(
                owner_to_module_id
            ),
            module_positions_by_id=tuple(
                module_positions_by_id
            ),
            owner_rewards=(
                owner_rewards
            ),
            next_module_ids_by_row=tuple(
                module_ids_by_row
            ),
            next_owner_to_module_id=tuple(
                owner_to_module_id
            ),
        )
    )

    dummy = {
        owner:
            torch.zeros(
                (
                    batch_size,
                    1,
                ),
                device=device,
            )
        for owner
        in owners
    }

    team_rewards = torch.tensor(
        [
            [
                float(
                    reward.team
                )
            ]
            for reward
            in rewards
        ],
        device=device,
    )

    terminated = torch.tensor(
        [
            [
                bool(
                    transition
                    .evaluation
                    .terminated
                )
            ]
            for transition
            in transitions
        ],
        device=device,
        dtype=torch.bool,
    )

    truncated = torch.tensor(
        [
            [
                bool(
                    transition
                    .evaluation
                    .truncated
                )
            ]
            for transition
            in transitions
        ],
        device=device,
        dtype=torch.bool,
    )

    vector_env_actions = {}

    for owner in owners:
        if owner not in step.env_actions:
            raise KeyError(
                f"env_actions is missing owner {owner}"
            )

        values = (
            step.env_actions[
                owner
            ]
        )

        if not torch.is_tensor(
            values
        ):
            raise TypeError(
                f"env_actions[{owner}] must be a torch.Tensor"
            )

        max_env_index = max(
            active_env_ids
        )

        if (
            values.ndim >= 1
            and values.shape[0]
            > max_env_index
        ):
            indexes = torch.tensor(
                active_env_ids,
                device=values.device,
                dtype=torch.long,
            )

            selected = (
                values.index_select(
                    0,
                    indexes,
                )
            )

        elif (
            values.ndim >= 1
            and values.shape[0]
            == batch_size
        ):
            selected = values

        else:
            raise ValueError(
                f"env_actions[{owner}] cannot represent "
                f"active env batch {active_env_ids}: "
                f"shape={tuple(values.shape)}"
            )

        vector_env_actions[
            owner
        ] = selected

    learner.record_online_rollout_batch(
        decisions=decisions,
        observations=dummy,
        states=dummy,
        env_actions=(
            vector_env_actions
        ),
        team_rewards=(
            team_rewards
        ),
        next_observations=dummy,
        next_states=dummy,
        terminated=terminated,
        truncated=truncated,
        next_critic_graphs=(
            next_critic_graphs
        ),
    )


def record_learning_step(args, learner, provider, step, episode, reward):
    import torch
    from drl.MARL.altruistic_rollout import AltruisticRolloutBatchPayload
    owners = learner.possible_agents
    device = args.device
    dummy = {owner:torch.zeros((1,1),device=device) for owner in owners}
    raw = step.transitions[0].previous_state
    positions = {m['module_id']:torch.tensor(m['position'],device=device) for m in raw['state']['modules']}
    route = step.contexts[0].routing_snapshot
    rows = tuple(ref.module_id for ref in route['module_refs'])
    provider.payload = AltruisticRolloutBatchPayload(module_ids_by_row=(rows,),
        owner_to_module_id=({owner:owner for owner in owners},),module_positions_by_id=(positions,),
        owner_rewards={owner:torch.tensor([[reward.owners[owner]]],device=device) for owner in owners},
        next_module_ids_by_row=(rows,),next_owner_to_module_id=({owner:owner for owner in owners},))
    evaluation = step.transitions[0].evaluation
    learner.record_online_rollout_batch(decisions=step.decisions,observations=dummy,states=dummy,
        env_actions=step.env_actions,team_rewards=torch.tensor([[reward.team]],device=device),
        next_observations=dummy,next_states=dummy,
        terminated=torch.tensor([[evaluation.terminated]],device=device,dtype=torch.bool),
        truncated=torch.tensor([[evaluation.truncated]],device=device,dtype=torch.bool),
        next_critic_graphs=step.next_critic_graphs)


def update_learner(args,learner,transitions,updates):
    import torch
    before = {key: [p.detach().clone() for p in model.parameters()] for key,model in
        [('actor',learner.shared_actor),('owner',learner.altruistic_owner_critic),('mean_field',learner.altruistic_mean_field_critic)]}
    learner.shared_actor.eval()
    learner.coordinated_update(timestep=transitions,timesteps=transitions)
    metrics = dict(learner.last_update_metrics)
    for key,model in [('actor',learner.shared_actor),('owner',learner.altruistic_owner_critic),('mean_field',learner.altruistic_mean_field_critic)]:
        finite_tensors(dict(model.named_parameters()),f'updated {key} parameters')
        metrics[f'{key}_parameters_changed'] = any(not torch.equal(old,p.detach()) for old,p in zip(before[key],model.parameters()))
    if not metrics or any(not math.isfinite(float(v)) for v in metrics.values() if isinstance(v,(float,int))):
        raise RuntimeError(f'nonfinite or missing APPO update diagnostics: {metrics}')
    for key in ('actor','owner','mean_field'):
        if not metrics[f'{key}_parameters_changed']:
            raise RuntimeError(f'{key} optimizer produced no parameter change')
    metrics.update(update=updates,transitions=transitions)
    append_jsonl(args.output/'updates.jsonl',metrics)
    artifact = args.output/'APPO_SMOKE.pt' if args.command=='smoke' else args.output/'C1_A_latest.pt'
    learner.save(str(artifact))
    write_json(artifact.with_suffix('.provenance.json'),dict(label='APPO_SMOKE' if args.command=='smoke' else 'C1_A',
        bc_checkpoint=str(args.checkpoint),bc_sha256=BC_SHA256,reward_config=str(args.reward_config),
        reward_config_sha256=hashlib.sha256(args.reward_config.read_bytes()).hexdigest(),
        fixed_alpha=args.alpha,altruism_radius_m=args.altruism_radius_m,
        actor_visibility_radius_m=1.,stochastic_log_std_init=STOCHASTIC_LOG_STD_INIT,
        physics_hz=240,decision_hz=240/args.step_ticks,metrics=metrics))
    print('APPO_UPDATE: '+json.dumps(metrics),flush=True)
    # All sidecars and tensors belong to the completed rollout generation.
    for memory in learner.memories.values():
        memory.reset()
    learner._structured_rollout_sidecar.clear()
    learner._altruistic_rollout_sidecar.clear()
    return metrics


def selected_layouts(args):
    from drl.MARL.reward_calibration import pilot_layouts,curriculum_identity
    if args.seeds:
        if len(args.domains) != 1:
            raise ValueError('--seeds requires exactly one --domains value')
        return [dict(domain=args.domains[0],seed=seed,split=args.split) for seed in args.seeds]
    if args.split=='validation':
        starts=dict(navigation=8000,stairs=13000,gap=17000,manipulation=21000)
        return [dict(domain=d,seed=starts[d],split='validation') for d in args.domains]
    if args.command=='train':
        import random
        from smores_ep.isaac.phase1_marl_curriculum import curriculum_seeds
        from smores_ep.isaac.phase1_stairs_curriculum import stair_curriculum_seeds
        from smores_ep.isaac.phase1_gap_curriculum import gap_curriculum_seeds
        from smores_ep.isaac.phase1_manipulation_curriculum import manipulation_curriculum_seeds
        allocations=dict(navigation=curriculum_seeds,stairs=stair_curriculum_seeds,
                         gap=gap_curriculum_seeds,manipulation=manipulation_curriculum_seeds)
        rows=[]
        for domain in args.domains:
            seeds=list(allocations[domain]('train'))
            random.Random(TRAIN_LAYOUT_POOL_SEED).shuffle(seeds)
            rows.extend(dict(domain=domain,seed=seed,split='train',
                difficulty=curriculum_identity(domain,seed).difficulty) for seed in seeds)
    else:
        rows = [layout for layout in pilot_layouts() if layout['domain'] in args.domains]
    if args.layouts_per_domain is not None:
        counts=Counter()
        selected=[]
        for row in rows:
            if counts[row['domain']]<args.layouts_per_domain:
                selected.append(row)
                counts[row['domain']]+=1
        rows=selected
    return rows if args.command in ('calibrate','train') else [next(row for row in rows if row['domain']==domain) for domain in args.domains]


def run_result(command,transitions,updates,failures,counts,accepted_nonempty):
    covered={domain for domain,count in counts.items() if count}
    complete=covered==set(DOMAINS)
    collection_ok=transitions>0 and not failures
    if not collection_ok:
        status='INCOMPLETE'
    elif command=='calibrate' and not complete:
        status='BOOTSTRAP_ONLY'
    elif not accepted_nonempty:
        status='INCOMPLETE_ACTUATION'
    elif command in ('smoke','train') and not updates:
        status='INCOMPLETE_UPDATE'
    else:
        status='GREEN'
    return dict(command=command,physical_transitions=transitions,updates=updates,failures=len(failures),
        status=status,physical_collection_ok=collection_ok,transition_counts=dict(counts),
        complete_four_domain_evidence=complete,
        calibration_fit_complete=command=='calibrate' and complete and collection_ok and accepted_nonempty>0,
        accepted_nonempty_native_proposals=accepted_nonempty)


def shutdown_on_signal(signum,frame):
    """SystemExit unwinds the active PhysicalEpisode and its owned process group."""
    raise SystemExit(128+signum)


def main(argv=None):
    import torch
    from dataclasses import replace
    from drl.MARL.phase1_vertical_slice import Phase1VerticalSliceRunner
    from drl.MARL.reward_calibration import CalibrationDataset,TransitionRecorder
    from drl.MARL.reward_config import RewardConfig,derive_initial_config
    args = parse_args(argv)
    signal.signal(signal.SIGTERM,shutdown_on_signal)
    torch.set_num_threads(args.torch_threads)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    run_id = time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid4().hex[:8]
    bridge = build_observation_bridge(args.repository_root)
    actor,critic = build_models(args,bridge)
    reward_config = RewardConfig.load(args.reward_config) if args.reward_config else None
    layouts = selected_layouts(args)
    failures,transitions,updates,accepted_nonempty = [],0,0,0
    episode_outcomes=[]
    learner = provider = None
    scheduler = TransitionMix()
    training_layout_pool=TrainingLayoutPool(layouts)
    all_rows = args.output/'transitions.jsonl'
    write_json(args.output/'run.json',dict(run_id=run_id,command=args.command,bc_sha256=BC_SHA256,
        layouts=layouts,physics_hz=240,step_ticks=args.step_ticks,fixed_alpha=args.alpha,
        altruism_radius_m=args.altruism_radius_m,structural_candidates='masked: fixed C1 morphology',
        stochastic_log_std_init=STOCHASTIC_LOG_STD_INIT,
        train_layout_pool_seed=TRAIN_LAYOUT_POOL_SEED if args.command=='train' and not args.seeds else None))
    episode_number = 0
    while True:
        if args.command=='train':
            if transitions >= args.total_transitions:
                break
            domain=scheduler.next_domain()
            layout=training_layout_pool.next_layout(domain)
        else:
            if episode_number>=len(layouts):
                break
            layout=layouts[episode_number]
        episode_id=f'{run_id}_{episode_number:04d}_{layout["domain"]}_{layout["seed"]}'
        episode_number+=1
        try:
            with PhysicalEpisode(args,layout['domain'],layout['seed'],episode_id,bridge) as episode:
                if learner is None and args.command in ('smoke','train'):
                    learner,provider=create_learner(args,bridge,actor,critic,episode.owners)
                    if args.appo_checkpoint:
                        learner.load(str(args.appo_checkpoint))
                if args.command=='evaluate' and args.appo_checkpoint:
                    # Restore actor via the canonical shared registry; evaluation
                    # never records rollout or calls an optimizer.
                    if learner is None:
                        learner,provider=create_learner(args,bridge,actor,critic,episode.owners)
                        learner.load(str(args.appo_checkpoint))
                builder = learner if args.command in ('smoke','train') else CalibrationDecisionBuilder(
                    actor=actor,critic=critic,owners=episode.owners,device=args.device)
                runner=Phase1VerticalSliceRunner(env=episode.env,runtime_context=episode.context,
                    decision_builder=builder,frame_synchronizer=episode.synchronizer)
                recorder=None
                if args.split=='train':
                    recorder=TransitionRecorder(all_rows,episode_manifest(args,run_id,episode,calibration_collection_label(args.command)))
                # Physical episode clocks are environment-local.
                episode_start_stamps={}

                for index in range(args.steps_per_layout):
                    policy=calibration_policy(index,args.steps_per_layout) if args.command=='calibrate' else 'bc_v4_stochastic'

                    if isinstance(builder,CalibrationDecisionBuilder):
                        builder.policy=policy

                    step=runner.step()

                    if step is None:
                        break

                    active_env_indices=tuple(
                        int(env_index)
                        for env_index
                        in step.active_env_indices
                    )

                    if not active_env_indices:
                        raise RuntimeError(
                            'vector MARL step contains no active environments'
                        )

                    scalar_rewards_by_env={}

                    for env_index in active_env_indices:
                        transition=step.transitions[env_index]

                        if transition is None:
                            raise RuntimeError(
                                f'active env {env_index} produced no completed physical transition'
                            )

                        if env_index not in episode_start_stamps:
                            episode_start_stamps[env_index]=float(
                                transition.previous_state['stamp']
                            )

                        finite_tensors(
                            step.contexts[env_index].actor_graph,
                            f'actor observations env {env_index}'
                        )

                        finite_tensors(
                            step.contexts[env_index].critic_graph,
                            f'critic observations env {env_index}'
                        )

                        finite_tensors(
                            step.decisions[env_index].owner_log_probs,
                            f'log probabilities env {env_index}'
                        )

                        finite_tensors(
                            step.decisions[env_index].global_value,
                            f'global value env {env_index}'
                        )

                        # Task 2 vectorizes C1 accounting/learning.
                        # Task 3 makes runtime transport itself env-aware.
                        status=episode.transport.status()

                        provenance=dict(
                            command_id=transition.actuation_result.command_id,
                            completed=status['last_completed'],
                            last_applied=status.get('last_applied'),
                            collection_policy=policy,
                            proposal=transition.actuation_result.proposal_diagnostics
                        )

                        native_accepted=(
                            status['last_completed']['outcome']=='accepted'
                            and bool(transition.actuation_result.decoded)
                        )

                        accepted_nonempty+=int(
                            native_accepted
                        )

                        transition_episode_id=(
                            episode_id
                            if (
                                len(active_env_indices)==1
                                and env_index==0
                            )
                            else f'{episode_id}_env{env_index:02d}'
                        )

                        append_jsonl(
                            args.output/'proposals.jsonl',
                            dict(
                                episode_id=transition_episode_id,
                                env_index=env_index,
                                domain=episode.domain,
                                episode_step=index+1,
                                native_accepted_nonempty=native_accepted,
                                **provenance
                            )
                        )

                        if recorder is not None:
                            recorder.observe(
                                transition,
                                availability=episode.availability,
                                application_provenance=provenance,
                                collector_truncated=(
                                    index==args.steps_per_layout-1
                                    and not transition.evaluation.terminated
                                )
                            )

                        if (
                            transition.evaluation.terminated
                            or transition.evaluation.truncated
                        ):
                            outcome=completed_episode_outcome(
                                transition,
                                episode_id=transition_episode_id,
                                domain=episode.domain,
                                seed=episode.seed,
                                split=args.split,
                                episode_start_stamp=episode_start_stamps[
                                    env_index
                                ]
                            )

                            episode_outcomes.append(
                                outcome
                            )

                            append_jsonl(
                                args.output/'episode_outcomes.jsonl',
                                outcome
                            )

                            episode_start_stamps.pop(
                                env_index,
                                None
                            )

                        if reward_config is not None:
                            scalar=reward_config.scalarize(
                                episode.domain,
                                transition.evaluation.raw_reward,
                                availability=episode.availability
                            )

                            scalar_rewards_by_env[
                                env_index
                            ]=scalar

                            append_jsonl(
                                args.output/'scalar_rewards.jsonl',
                                dict(
                                    episode_id=transition_episode_id,
                                    env_index=env_index,
                                    domain=episode.domain,
                                    step=index+1,
                                    team=scalar.team,
                                    owners=scalar.owners,
                                    excluded_terms=getattr(
                                        scalar,
                                        'excluded_terms',
                                        ()
                                    ),
                                    broadcast_terms=getattr(
                                        scalar,
                                        'broadcast_terms',
                                        ()
                                    ),
                                    terminated=transition.evaluation.terminated,
                                    truncated=transition.evaluation.truncated,
                                    **scalar_outcome_fields(
                                        transition.evaluation
                                    )
                                )
                            )

                        # Count genuine environment transitions, not
                        # vector step() calls.
                        transitions+=1
                        scheduler.record(
                            episode.domain,
                            1
                        )

                        print(
                            f'PHYSICAL_TRANSITION: {episode.domain} '
                            f'{index+1} env={env_index} '
                            f'total={transitions}',
                            flush=True
                        )

                    # Exactly one APPO/MAPPO vector storage operation per
                    # shared physical vector step.
                    if (
                        reward_config is not None
                        and args.command in ('smoke','train')
                    ):
                        if (
                            set(scalar_rewards_by_env)
                            != set(active_env_indices)
                        ):
                            raise RuntimeError(
                                'missing scalar reward for active vector environment'
                            )

                        record_learning_batch(
                            args,
                            learner,
                            provider,
                            step,
                            scalar_rewards_by_env
                        )

                    if (
                        learner is not None
                        and args.command in ('smoke','train')
                        and len(
                            learner.memories[
                                episode.owners[0]
                            ]
                        ) >= args.rollout_transitions
                    ):
                        updates+=1

                        update_learner(
                            args,
                            learner,
                            transitions,
                            updates
                        )
        except Exception as error:
            failure=dict(episode_id=episode_id,layout=layout,error_type=type(error).__name__,failure_reason=str(error),
                         bootstrap_or_runtime_failure=True,calibration_row=False)
            failures.append(failure)
            append_jsonl(args.output/'episode_failures.jsonl',failure)
            print('EPISODE_FAILURE: '+json.dumps(failure),flush=True)
            # An infrastructure failure has no valid terminal transition.
            # Never append the next reset/domain to this learner rollout: that
            # would connect GAE across unrelated physical episodes.
            if args.fail_fast or args.command in ('smoke','train'):
                raise
    if args.command=='calibrate' and all_rows.is_file():
        dataset=CalibrationDataset.from_jsonl(all_rows)
        write_json(args.output/'calibration_summary.json',dataset.report(gamma=args.gamma))
        if not failures and accepted_nonempty and {row['manifest']['domain'] for row in dataset.rows}==set(DOMAINS):
            config=derive_initial_config(dataset)
            config.save(args.output/'reward_config.initial.json')
            write_json(args.output/'mixing_summary.json',config.mixing_summary(dataset,gamma=args.gamma))
    if args.command in ('smoke','train') and learner is not None and len(learner.memories[learner.possible_agents[0]]):
        updates+=1
        update_learner(args,learner,transitions,updates)
    result=run_result(args.command,transitions,updates,failures,scheduler.counts,accepted_nonempty)
    result.update(episodes=len(episode_outcomes),episodes_by_domain=aggregate_episode_outcomes(episode_outcomes))
    write_json(args.output/'result.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['status'] in ('GREEN','BOOTSTRAP_ONLY') else 1


if __name__=='__main__':
    raise SystemExit(main())
