"""BC-v4 public physical action set. Absence is not an action class.

Instance IDs are routing/output only. Ordered connectors retain endpoint roles;
participants and reference sets are unordered. No simulator/framework dependency.
"""
from dataclasses import dataclass
from .bc_v1_contract import ActionChoice,ModuleRef,ConnectorRef

VERSION='mssr.bc_v1.action_set.v1'
ActionInstance=ActionChoice


def action_identity(action):
    return (action.action_id,action.initiator,action.resource,action.connectors)


def _value(value):
    if isinstance(value,tuple) and value and all(isinstance(v,(ModuleRef,ConnectorRef)) for v in value):
        return frozenset(value)
    return value


def _semantic(action):
    return (action_identity(action),frozenset(action.participants),
            frozenset((p.name,_value(p.value)) for p in action.parameters),action.payload)


@dataclass(frozen=True,eq=False)
class ActionBundle:
    actions: tuple[ActionInstance,...] = ()

    def __post_init__(self):
        if type(self.actions) is not tuple or any(not isinstance(a,ActionChoice) for a in self.actions):
            raise ValueError('ActionBundle requires physical ActionChoice instances')
        if len({action_identity(a) for a in self.actions})!=len(self.actions):
            raise ValueError('Duplicate action candidate identity')

    def __eq__(self,other):
        if not isinstance(other,ActionBundle):return NotImplemented
        return frozenset(map(_semantic,self.actions))==frozenset(map(_semantic,other.actions))

    def __hash__(self):return hash(frozenset(map(_semantic,self.actions)))
