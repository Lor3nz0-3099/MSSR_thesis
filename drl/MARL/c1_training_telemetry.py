"""Passive C1 training evidence, independent of Isaac and trainer control flow.

Call observe_transition once per native row decision, using the pre-reset course
identity. Call record_update only when the learner actually updates. Resource
sampling is explicit and should run outside the transition hot path.
"""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import copy
import math
import os
import resource
import subprocess
import time

import torch


def _finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach()).all().item())
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(item) for item in value)
    if is_dataclass(value):
        return all(_finite_tree(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, (float, int)):
        return math.isfinite(value)
    return True


class TrainingTelemetry:
    def __init__(self):
        self.started = time.monotonic()
        self.domains = defaultdict(lambda: dict(row_transitions=0, active_marl_transitions=0,
            completed_episodes=0, censored_episodes=0, successes=0, failures=0))
        self.open_episodes = {}
        self.episodes = []
        self.updates = []
        self.timings = defaultdict(lambda: dict(calls=0, seconds=0.))
        self.finite_checks = {name: dict(checked=0, nonfinite=0) for name in ('reward', 'actions', 'graphs')}
        self.resources = {}

    def observe_transition(self, *, domain, seed, generation, reward, terminated, truncated,
                           success, active, delta_s=None, actions=None, graphs=None,
                           failure=None, env_index=0, reward_breakdown=None, c1_state=None):
        """Record one row; inactive assembly time stays visible in episode length.

        Timeout and still-open episodes are censored, never counted as failures.
        A terminal success takes precedence over a simultaneous time limit.
        """
        domain, seed, generation = str(domain), int(seed), int(generation)
        key = (int(env_index), domain, seed, generation)
        row = self.domains[domain]
        row['row_transitions'] += 1
        row['active_marl_transitions'] += int(bool(active))
        episode = self.open_episodes.setdefault(key, dict(env_index=int(env_index), domain=domain,
            seed=seed, generation=generation, return_=0., active_return=0., length=0,
            active_length=0, elapsed_s=0., nonfinite_rewards=0))
        reward = float(reward)
        for name, value in (('reward', reward), ('actions', actions), ('graphs', graphs)):
            if value is not None:
                self.finite_checks[name]['checked'] += 1
                self.finite_checks[name]['nonfinite'] += int(not _finite_tree(value))
        if math.isfinite(reward):
            episode['return_'] += reward
            episode['active_return'] += reward if active else 0.
        else:
            episode['nonfinite_rewards'] += 1
        episode['length'] += 1
        episode['active_length'] += int(bool(active))
        if reward_breakdown is not None:
            breakdown=episode.setdefault('reward_breakdown',{})
            for name,value in reward_breakdown.items():
                value=float(value)
                if not math.isfinite(value):raise ValueError('nonfinite reward breakdown')
                breakdown[name]=breakdown.get(name,0.)+value
        if c1_state is not None:episode['c1_state']=copy.deepcopy(c1_state)
        if delta_s is not None:
            if not math.isfinite(float(delta_s)) or delta_s < 0:
                raise ValueError('delta_s must be finite and nonnegative')
            episode['elapsed_s'] += float(delta_s)
        if terminated or truncated:
            censored = bool(truncated and not terminated)
            episode.update(terminated=bool(terminated), truncated=bool(truncated),
                success=bool(success), failure=bool(terminated and not success) if failure is None else bool(failure),
                censored=censored, censor_reason='time_limit' if censored else None)
            row['censored_episodes' if censored else 'completed_episodes'] += 1
            if not censored:
                row['successes'] += int(bool(success))
                row['failures'] += int(episode['failure'])
            episode['return'] = episode.pop('return_')
            self.episodes.append(episode)
            del self.open_episodes[key]

    def record_update(self, metrics):
        numeric = {name: float(value) for name, value in metrics.items()}
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError('nonfinite learner update metric')
        self.updates.append(numeric)

    def record_timing(self, name, seconds):
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('timing must be finite and nonnegative')
        self.timings[str(name)]['calls'] += 1
        self.timings[str(name)]['seconds'] += seconds

    def sample_resources(self, device=None):
        """Sample process resources without resetting allocator counters or syncing CUDA.

        nvidia-smi is best effort and reports a sampled process peak, not an
        allocator peak. RSS peak uses Linux ru_maxrss (KiB). Missing GPU data is
        explicit rather than silently reported as zero.
        """
        self.resources['rss_peak_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        try:
            with open('/proc/self/statm', encoding='ascii') as stream:
                self.resources['rss_bytes'] = int(stream.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            pass
        cuda_device = device is None or torch.device(device).type == 'cuda'
        if cuda_device and torch.cuda.is_available():
            self.resources['torch_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated(device))
            self.resources['torch_peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved(device))
        try:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
            used = sum(int(line.split(',')[1].strip()) * 1024**2 for line in result.stdout.splitlines()
                if line.split(',')[0].strip() == str(os.getpid()))
            self.resources['process_gpu_sampled_peak_bytes'] = max(
                self.resources.get('process_gpu_sampled_peak_bytes', 0), used)
            self.resources['process_gpu_measurement'] = 'nvidia-smi sampled process memory'
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            self.resources.setdefault('process_gpu_measurement', 'unavailable')
        return dict(self.resources)

    def snapshot(self):
        domains = copy.deepcopy(dict(self.domains))
        episodes = copy.deepcopy(self.episodes)
        for episode in self.open_episodes.values():
            episode = dict(episode)
            episode['return'] = episode.pop('return_')
            episode.update(terminated=False, truncated=False, success=False, failure=False,
                censored=True, censor_reason='observation_end')
            episodes.append(episode)
            domains[episode['domain']]['censored_episodes'] += 1
        for row in domains.values():
            completed = row['completed_episodes']
            row['episode_success_rate'] = row['successes'] / completed if completed else None
        return dict(schema='mssr.c1_training_telemetry.v1', wall_s=time.monotonic()-self.started,
            row_transitions=sum(row['row_transitions'] for row in domains.values()),
            active_marl_transitions=sum(row['active_marl_transitions'] for row in domains.values()),
            completed_episodes=sum(row['completed_episodes'] for row in domains.values()),
            censored_episodes=sum(row['censored_episodes'] for row in domains.values()),
            domains=domains, episodes=episodes, updates=copy.deepcopy(self.updates),
            timings=copy.deepcopy(dict(self.timings)), finite_checks=copy.deepcopy(self.finite_checks),
            resources=dict(self.resources))
