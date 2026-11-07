"""Read-only, port-labelled partial target embedding for viewer explanations.

No complete physical-module assignment is assumed. Only the declared root is
anchored; unconstrained roles are explored, never provided to control code.
"""
from itertools import permutations


class TargetRelationDiagnostic:
    def __init__(self, target, modules, *, physical_root='smores_01'):
        self.target = target
        self.modules = frozenset(modules)
        self.physical_root = physical_root
        roots = [n['node_id'] for n in target['nodes'] if n.get('attributes', {}).get('is_target_root')]
        if len(roots) != 1 or physical_root not in self.modules:
            raise ValueError('Viewer target requires one declared root and its physical anchor')
        self.root = roots[0]
        self.slots = tuple(n['node_id'] for n in target['nodes'])
        if len(set(self.slots)) != len(self.slots) or len(self.slots) != len(self.modules):
            raise ValueError('Target slots and physical module count differ')
        self.edges = set()
        for edge in target['edges']:
            a, b = edge['module_a_id'], edge['module_b_id']
            fa, fb = edge['attributes']['face_a'], edge['attributes']['face_b']
            self.edges.add((a, fa, b, fb))
            self.edges.add((b, fb, a, fa))
        self.history = {}

    @staticmethod
    def key(pair):
        return tuple(sorted(tuple(p) for p in pair))

    def record_attempt(self, pair, generation, failure_code, stamp):
        self.history.setdefault((generation, self.key(pair)), []).append(
            dict(timestamp_s=float(stamp), failure_code=failure_code))

    def classify(self, pair, connections, generation):
        pair = tuple(tuple(p) for p in pair)
        current = [dict(joint_id=c.joint_id, endpoints=c.endpoints, joint_valid=bool(c.joint_valid))
                   for c in connections]
        result = dict(classification='UNKNOWN', proposed_relation=pair,
            target_topology=self.target, current_topology=current,
            anchor={self.physical_root: self.root},
            assignment_convention='partial port-labelled embedding; other module roles unassigned',
            prior_attempts=list(self.history.get((generation, self.key(pair)), ())),
            ambiguous=False, compatible_target_relations=[])
        if len(pair) != 2 or any(len(p) != 2 or p[0] not in self.modules for p in pair):
            return result
        if pair[0][0] == pair[1][0]:
            result['classification'] = 'NON_TARGET_OR_INCOMPATIBLE'
            return result
        valid = [tuple(tuple(p) for p in c.endpoints) for c in connections if c.joint_valid]
        involved = {self.physical_root, *(p[0] for p in pair), *(p[0] for e in valid for p in e)}
        if not involved <= self.modules:
            return result
        unknown = sorted(involved - {self.physical_root})
        slots = tuple(s for s in self.slots if s != self.root)
        compatible, base_count = set(), 0
        for role_tuple in permutations(slots, len(unknown)):
            roles = dict(zip(unknown, role_tuple))
            roles[self.physical_root] = self.root
            def mapped(edge):
                (a, fa), (b, fb) = edge
                return roles[a], fa, roles[b], fb
            if not all(mapped(e) in self.edges for e in valid):
                continue
            base_count += 1
            relation = mapped(pair)
            if relation in self.edges:
                compatible.add(relation)
        result['compatible_target_relations'] = sorted(compatible)
        if not base_count:
            result['diagnostic_reason'] = 'current valid topology cannot be embedded under root anchor'
        elif not compatible:
            result['classification'] = 'NON_TARGET_OR_INCOMPATIBLE'
        elif len(compatible) > 1:
            result.update(ambiguous=True, diagnostic_reason='multiple target relations fit unassigned roles')
        elif any(self.key(e) == self.key(pair) for e in valid):
            result['classification'] = 'TARGET_RELATION_ALREADY_SATISFIED'
        else:
            result['classification'] = ('RETRY_SAME_RELATION' if result['prior_attempts']
                                        else 'TARGET_RELATION_UNSATISFIED')
        return result
