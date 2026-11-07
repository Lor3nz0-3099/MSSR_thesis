"""In-process scheduling adapter, reusing both existing Expert state machines."""
from smores_ep.primitives.model import PrimitiveGoal, PrimitiveName, PrimitiveState


def reset_free_assembly_modules(backend, module_ids, stamp, execution_id):
    """Apply the existing acknowledged reset barrier before a fresh Expert FSM.

    Robot poses are reset and old goals canceled by the native environment.
    The primitive owns clearing retained posture references and group barriers;
    a router-only drive reset cannot retire that execution state.
    """
    executor = backend['primitive_executor']
    goal = PrimitiveGoal(goal_id=f'{execution_id}-reset-free',
        primitive=PrimitiveName.RESET_FREE_MODULES, module_ids=tuple(module_ids))
    admitted = executor.submit(goal, stamp)
    if admitted.state is not PrimitiveState.ACCEPTED:
        raise RuntimeError(f'Native free-module reset rejected: {admitted.code}: {admitted.message}')
    result = executor.step(stamp)
    status = next((item for item in result.statuses if item.goal_id == goal.goal_id), None)
    if status is None or status.state is not PrimitiveState.SUCCEEDED:
        reason = 'missing completion acknowledgement' if status is None else f'{status.code}: {status.message}'
        raise RuntimeError(f'Native free-module reset failed: {reason}')
    return status


class InProcessAssemblyRuntime:
    def __init__(self, fsm, backend):
        self.fsm, self.backend = fsm, backend
        self.statuses = {}
        self.held_commands = {}
        self.decision = None
        self.submitted = []

    def tick(self, stamp):
        # Admission barriers are consumed synchronously; physical completion
        # remains exclusively in the existing primitive executor.
        for _ in range(32):
            payload = {'statuses':[s.to_dict() for s in self.statuses.values()]}
            self.decision = self.fsm.step(payload)
            if self.decision.done:
                if not self.decision.success:
                    raise RuntimeError(f'Expert assembly failed: {self.decision.message}')
                break
            goal = self.decision.primitive_goal_payload
            if goal is None:
                break
            admitted = self.backend['primitive_executor'].submit(PrimitiveGoal.from_dict(goal),stamp)
            self.statuses[admitted.goal_id] = admitted
            self.submitted.append(goal)
        else:
            raise RuntimeError('Expert admission loop exceeded bounded wave size')
        result = self.backend['primitive_executor'].step(stamp)
        for status in result.statuses:
            self.statuses[status.goal_id] = status
        self.held_commands = dict(result.commands)

    def apply(self):
        commands = self.backend['primitive_executor'].compose_with_baseline({},self.held_commands)
        return self.backend['router'].apply(commands)
