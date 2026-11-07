"""CPU-only checks for C1 Phase D stable physical-connection signatures."""

from drl.MARL.c1_phase_d_capture import stable_connection_signature
from drl.MARL.c1_structure import C1Connection


def test_signature_matches_native_c1_connection_fields():
    first = C1Connection(
        '/World/envs/env_0/DockingJoints/dock_0001',
        'smores_07', 'BOTTOM', 'smores_01', 'TOP', True)
    assert stable_connection_signature((first,)) == (
        (('smores_01', 'TOP'), ('smores_07', 'BOTTOM')),
    )


def test_signature_is_order_and_joint_path_invariant():
    first = C1Connection('joint-a','smores_07','BOTTOM','smores_01','TOP',True)
    reverse = C1Connection('joint-b','smores_01','TOP','smores_07','BOTTOM',True)
    other = C1Connection('joint-c','smores_02','LEFT','smores_03','RIGHT',True)
    assert stable_connection_signature((first,other)) == stable_connection_signature((other,reverse))
    assert stable_connection_signature(()) == ()
