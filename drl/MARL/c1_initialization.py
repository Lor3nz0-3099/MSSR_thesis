"""Episode initialization and authority, independent of desired morphology."""
from enum import Enum

import torch

from drl.MARL.runtime_handoff import DirectMARLHandoff, RuntimeAuthority, RuntimePhase


class InitializationMode(str, Enum):
    LOOSE_MODULES = 'loose_modules'
    ASSEMBLED_STATE = 'assembled_state'
    EXPERT_VALIDATION = 'expert_validation'


def resolve_initialization(curriculum, requested=None):
    if curriculum not in ('c1', 'catalog', 'expert_validation'):
        raise ValueError(f'Unsupported curriculum {curriculum!r}')
    mode = InitializationMode(requested or (
        'loose_modules' if curriculum == 'c1' else 'expert_validation'))
    if curriculum == 'c1' and mode is InitializationMode.EXPERT_VALIDATION:
        raise ValueError('--curriculum c1 requires policy-controlled initialization')
    return mode


def mode_for(env):
    return InitializationMode(getattr(getattr(env, 'cfg', None),
                                      'initialization_mode', 'expert_validation'))


def require_expert_validation(env):
    if mode_for(env) is not InitializationMode.EXPERT_VALIDATION:
        raise RuntimeError('Expert planning is forbidden outside expert_validation')


class PolicyControlledHandoff(DirectMARLHandoff):
    """Provider-compatible policy authority with no Expert or topology gate."""
    def activate_policy(self):
        if self.phase is not RuntimePhase.IDLE:
            raise RuntimeError('Policy activation requires a reset episode')
        self._authority = RuntimeAuthority.MARL
        self._phase = RuntimePhase.MARL_CONTROL

    def begin_expert(self, *_args, **_kwargs):
        raise RuntimeError('Expert authority is forbidden in policy-controlled episodes')

    def observe_expert_terminal(self, *_args, **_kwargs):
        raise RuntimeError('Expert completion is forbidden in policy-controlled episodes')

    def observe_topology(self, *_args, **_kwargs):
        raise RuntimeError('A target topology cannot grant policy authority')


def native_action_mask(route, mode, *, device):
    """False permits an enumerated candidate; runtime still admits/rejects it."""
    structural_masked = InitializationMode(mode) is InitializationMode.EXPERT_VALIDATION
    return torch.tensor([False] * len(route['resource_candidates']) +
                        [structural_masked] * len(route['structural_bindings']),
                        dtype=torch.bool, device=device)
