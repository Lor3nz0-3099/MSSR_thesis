from itertools import permutations
import pytest

from drl.IL.bc_v1.bc_v1_contract import ActionChoice,ModuleRef,ResourceRef
from drl.IL.bc_v1.action_set_decoder import assert_executable
from drl.MARL.action_projection import ProjectionCandidate,project_executable_action_set


def candidate(name,score,resources,invalid=None):
    claims=tuple(ResourceRef(ModuleRef(m),'locomotion') for m in resources)
    choice=ActionChoice(name,ModuleRef(name))
    return ProjectionCandidate(candidate_id=name,candidate_index=ord(name),raw_presence_logit=score,
        owner_ids=(name,),choice=choice,claims=claims,invalid_reason=invalid)


def project(rows,**kwargs):
    result=project_executable_action_set(tuple(rows),**kwargs)
    assert result.original_bundle.actions==tuple(c.choice for c in rows if c.choice is not None)
    by_action={c.choice:c.claims for c in rows}
    assert_executable(result.projected_bundle,lambda a:by_action[a])
    return result


@pytest.mark.parametrize('scores,expected', [((8,6),('A',)),((6,8),('B',))])
def test_conflict_higher_policy_confidence_wins(scores,expected):
    result=project([candidate('A',scores[0],('x',)),candidate('B',scores[1],('x',))])
    assert result.accepted_candidate_ids==expected
    assert result.dropped[0].reason=='Conflicting physical resource claims'
    assert result.dropped[0].code=='RESOURCE_BUSY'
    assert len(result.conflicts)==1


def test_independent_survives_and_result_invariant_to_input_order():
    rows=[candidate('A',8,('x',)),candidate('B',6,('x',)),candidate('C',5,('y',))]
    for order in permutations(rows):
        result=project(order)
        assert result.accepted_candidate_ids==('A','C')
        assert tuple(d.candidate_id for d in result.dropped)==('B',)


@pytest.mark.parametrize('delta',[0.,.5e-6])
def test_conflicting_ties_drop_all_independent_survives(delta):
    for order in permutations([candidate('A',6,('x',)),candidate('B',6+delta,('x',)),candidate('C',5,('y',))]):
        result=project(order)
        assert result.accepted_candidate_ids==('C',)
        assert {d.candidate_id for d in result.dropped}=={'A','B'}
        assert all(d.reason=='Conflicting confidence tie' for d in result.dropped)


def test_multiple_conflicts_and_tied_component_removal_releases_lower_action():
    rows=[candidate('A',8,('x','y')),candidate('B',8,('x',)),
          candidate('C',7,('y',)),candidate('D',5,('z',))]
    for order in permutations(rows):
        result=project(order)
        assert result.accepted_candidate_ids==('C','D')
        assert {d.candidate_id for d in result.dropped}=={'A','B'}


def test_preblocked_tie_peer_cannot_veto_independent_survivor():
    result=project([candidate('A',9,('x',)),candidate('B',6,('x','y')),candidate('C',6,('y',))])
    assert result.accepted_candidate_ids==('A','C')


def test_invalid_or_unavailable_high_confidence_does_not_reserve_claim():
    rows=[candidate('A',9,('x',),invalid='Invalid participant scope'),candidate('B',6,('x',))]
    result=project(rows)
    assert result.accepted_candidate_ids==('B',)
    assert result.dropped[0].code=='INVALID_GOAL'
    assert result.dropped[0].reason=='Invalid participant scope'
    result=project([candidate('A',9,('x',)),candidate('B',6,('x',))],
        feasibility=lambda c:('RESOURCE_BUSY','owned by active goal') if c.candidate_id=='A' else None)
    assert result.accepted_candidate_ids==('B',)
    assert result.dropped[0].reason=='owned by active goal'


def test_invalid_claims_drop_individually():
    result=project([candidate('A',9,()),candidate('B',6,('x',))])
    assert result.accepted_candidate_ids==('B',)
    assert result.dropped[0].reason=='Invalid physical claims'


def test_true_ties_use_no_action_type_or_candidate_id_preference():
    rows=[candidate('A',6,('x',)),candidate('B',6,('x',)),candidate('C',6,('z',))]
    result=project(rows)
    assert result.accepted_candidate_ids==('C',)
    assert {d.candidate_id for d in result.dropped}=={'A','B'}
