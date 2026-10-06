"""Regression tests for the compact, command-level IL contract."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import preprocessing

from preprocessing import (
    ActionDecisionTracker, CanonicalTransitionBuilder, PreprocessingError,
    adapt_action,
)


def graph(stamp):
    return {
        'stamp': stamp,
        'nodes': [{'module_id': 'smores_01', 'node_id': 'smores_01', 'attributes': {
            'module_id': 'smores_01', 'position': [0., 0., 0.],
            'orientation': [0., 0., 0., 1.], 'pose': {'position': [0., 0., 0.]},
            'linear_velocity': [0., 0., 0.], 'angular_velocity': [0., 0., 0.],
            'actuators': {'tilt': {'position_rad': .2, 'velocity_rad_s': .1,
                                  'max_effort_nm': 4.8}},
            'connectors': [{'connector_id': 'TOP', 'face_name': 'TOP',
                            'status': 'available', 'is_enabled': False}],
            'capabilities': ['static'], 'current_role': 'unassigned',
        }}], 'edges': [], 'global_attributes': {'course': {'gap': {
            'near_edge_x_m': 1., 'far_edge_x_m': 1.2, 'width_m': .2},
            'course_profile': 'gap_test'}},
    }


def record(stamp=1., primitive='set_tilt', goal_id='g1'):
    return {'stamp': stamp, 'timestep': int(stamp), 'task_type': 'self_assembly',
            '_loader_phase': 'assembly', 'graph_t': graph(stamp),
            'graph_t_plus_1': graph(stamp + .1), 'action_valid': True,
            'supervision': {'valid_for_behavior_cloning': True},
            'expert_action': {'locomotion': {}, 'primitive_goal': {
                'primitive': primitive, 'goal_id': goal_id,
                'module_ids': ['smores_01'], 'parameters': {'angle_rad': .4}}}}


class CommandContractTests(unittest.TestCase):
    @staticmethod
    def alignment(stamp, phase, mobile='smores_01', action_index=0, retry=0):
        row = record(stamp, primitive='align_faces',
                     goal_id=f'expert-w0-a{action_index}-{phase}' + (f'-r{retry}' if retry else ''))
        row['_loader_source_path'] = 'assembly.jsonl'
        goal = row['expert_action']['primitive_goal']
        goal['module_ids'] = [mobile, 'anchor']
        goal['parameters'] = {'face_a': 'BOTTOM', 'face_b': 'LEFT',
                              'execution_phase': phase, 'clocking_quarter_turns': 0}
        return row

    def test_alignment_phases_are_one_policy_operation_and_keep_execution_events(self):
        builder = CanonicalTransitionBuilder()
        reach = builder.build_expert_step(self.alignment(1., 'reach'))
        align = builder.build_expert_step(self.alignment(2., 'align'))
        retry = builder.build_expert_step(self.alignment(3., 'retreat', retry=1))
        approach = builder.build_expert_step(self.alignment(4., 'approach', retry=1))
        self.assertEqual(reach['action_t']['commands'][0]['command'], 'align_faces')
        self.assertNotIn('execution_phase', reach['action_t']['commands'][0]['parameters'])
        self.assertEqual(reach['graph_t']['control_state']['active_alignments'], [])
        self.assertEqual(len(reach['graph_t_plus_1']['control_state']['active_alignments']), 1)
        for transition in [align, retry, approach]:
            self.assertEqual(transition['action_t']['commands'], [])
            self.assertFalse(transition['supervision']['valid_for_behavior_cloning'])
            self.assertEqual(len(transition['graph_t']['control_state']['active_alignments']), 1)
        self.assertEqual(retry['provenance']['alignment_events'][0]['phase'], 'retreat')

    def test_dock_closes_alignment_without_automatically_becoming_part_of_it(self):
        builder = CanonicalTransitionBuilder()
        builder.build_expert_step(self.alignment(1., 'reach'))
        row = self.alignment(2., 'dock')
        row['expert_action']['primitive_goal']['primitive'] = 'dock'
        result = builder.build_expert_step(row)
        self.assertEqual(result['action_t']['commands'][0]['command'], 'dock')
        self.assertEqual(result['graph_t_plus_1']['control_state']['active_alignments'], [])

    def test_parallel_alignments_share_an_anchor_but_not_the_same_face(self):
        from preprocessing import coalesce_parallel_transitions
        def transition(commands):
            result = CanonicalTransitionBuilder().build_expert_step(record())
            result['action_t']['commands'] = commands
            result['supervision']['command_mask'] = [True]*len(commands)
            result['provenance']['source_repeat_count'] = 1
            return result
        a = {'command': 'align_faces', 'module_ids': ['A', 'anchor'],
             'parameters': {'face_a': 'BOTTOM', 'face_b': 'LEFT'}}
        b = {'command': 'align_faces', 'module_ids': ['B', 'anchor'],
             'parameters': {'face_a': 'BOTTOM', 'face_b': 'RIGHT'}}
        rows = list(coalesce_parallel_transitions([transition([a]), transition([b])]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]['action_t']['commands']), 2)
        self.assertEqual(rows[0]['provenance']['source_repeat_count'], 2)
        b['parameters']['face_b'] = 'LEFT'
        self.assertEqual(len(list(coalesce_parallel_transitions([transition([a]), transition([b])]))), 2)

    def test_parallel_merge_keeps_different_states_and_conflicting_revisions_separate(self):
        from preprocessing import coalesce_parallel_transitions
        builder = CanonicalTransitionBuilder()
        a = builder.build_expert_step(record(1., goal_id='one'))
        b = builder.build_expert_step(record(2., goal_id='two'))
        self.assertEqual(len(list(coalesce_parallel_transitions([a,b]))), 2)
        b['graph_t'] = copy.deepcopy(a['graph_t'])
        b['provenance']['decision_stamp'] = a['provenance']['decision_stamp']
        b['action_t']['commands'][0]['parameters']['angle_rad'] = .9
        self.assertEqual(len(list(coalesce_parallel_transitions([a,b]))), 2)

    def test_parallel_merge_preserves_distinct_recorded_next_states(self):
        from preprocessing import coalesce_parallel_transitions
        a = CanonicalTransitionBuilder().build_expert_step(record())
        b = copy.deepcopy(a)
        b['graph_t_plus_1']['nodes'][0]['position'][0] = .5
        rows = list(coalesce_parallel_transitions([a, b]))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['graph_t_plus_1']['nodes'][0]['position'][0], 0.)
        self.assertEqual(rows[1]['graph_t_plus_1']['nodes'][0]['position'][0], .5)

    def test_no_decision_repeats_do_not_inflate_merged_bc_weight(self):
        from preprocessing import coalesce_parallel_transitions
        builder = CanonicalTransitionBuilder()
        a = builder.build_expert_step(record())
        idle = record()
        idle['_source_repeat_count'] = 100
        b = builder.build_expert_step(idle)
        rows = list(coalesce_parallel_transitions([a, b]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['provenance']['source_repeat_count'], 101)
        self.assertEqual(rows[0]['supervision']['sample_weight'], 1)

    def test_continuous_revision_masks_earlier_target_without_changing_endpoints(self):
        from preprocessing import resolve_continuous_revisions
        a = CanonicalTransitionBuilder().build_expert_step(record())
        a['action_t']['commands'] = [{'command': 'wheel_velocity', 'module_ids': ['smores_01'],
                                    'parameters': {'left_rad_s': .2, 'right_rad_s': .2}}]
        b = copy.deepcopy(a)
        a['graph_t_plus_1'] = copy.deepcopy(a['graph_t'])
        b['action_t']['commands'][0]['parameters'] = {'left_rad_s': .4, 'right_rad_s': .4}
        endpoints = [copy.deepcopy(row['graph_t_plus_1']) for row in (a,b)]
        rows = list(resolve_continuous_revisions([a,b]))
        self.assertEqual([row['graph_t_plus_1'] for row in rows], endpoints)
        self.assertEqual(rows[0]['action_t']['commands'][0]['parameters']['left_rad_s'], .2)
        self.assertEqual(rows[0]['supervision']['command_mask'], [False])
        self.assertEqual(rows[0]['supervision']['sample_weight'], 0)
        self.assertFalse(rows[0]['supervision']['valid_for_behavior_cloning'])
        self.assertEqual(rows[1]['supervision']['command_mask'], [True])

    def test_continuous_revision_preserves_joint_dispatch_and_its_source_weight(self):
        from preprocessing import resolve_continuous_revisions, coalesce_parallel_transitions
        builder = CanonicalTransitionBuilder()
        raw = record()
        raw['expert_action']['primitive_goal'] = None
        raw['expert_action']['locomotion'] = {'smores_01': {'vx': .01}}
        raw['_source_repeat_count'] = 7
        a = builder.build_expert_step(raw)
        joint = record()
        joint['expert_action']['locomotion'] = {'smores_01': {'vx': .01}}
        joint['_source_repeat_count'] = 3
        b = builder.build_expert_step(joint)
        revised = copy.deepcopy(raw)
        revised['expert_action']['locomotion']['smores_01']['vx'] = .02
        revised['_source_repeat_count'] = 1
        c = builder.build_expert_step(revised)
        rows = list(coalesce_parallel_transitions(resolve_continuous_revisions([a,b,c])))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['supervision']['command_mask'], [False, True])
        self.assertEqual(rows[0]['supervision']['sample_weight'], 3)
        self.assertEqual(rows[0]['provenance']['source_repeat_count'], 10)
        self.assertTrue(rows[0]['supervision']['valid_for_behavior_cloning'])

    def test_continuous_revision_does_not_cross_inputs_sources_or_distinct_modules(self):
        from preprocessing import resolve_continuous_revisions
        a = CanonicalTransitionBuilder().build_expert_step(record())
        a['action_t']['commands'] = [{'command': 'wheel_velocity', 'module_ids': ['smores_01'],
                                    'parameters': {'left_rad_s': .2, 'right_rad_s': .2}}]
        for difference in ('input', 'source', 'module', 'stamp'):
            first = copy.deepcopy(a)
            other = copy.deepcopy(a)
            other['action_t']['commands'][0]['parameters']['left_rad_s'] = .4
            if difference == 'input': other['graph_t']['nodes'][0]['position'][0] = .5
            elif difference == 'source': other['provenance']['loader_source_path'] = 'other.jsonl'
            elif difference == 'module': other['action_t']['commands'][0]['module_ids'] = ['smores_02']
            else: other['provenance']['decision_stamp'] = 2.
            rows = list(resolve_continuous_revisions([first,other]))
            self.assertTrue(all(row['supervision']['valid_for_behavior_cloning'] for row in rows))

    def test_invalid_continuous_command_cannot_replace_a_valid_bc_target(self):
        from preprocessing import resolve_continuous_revisions
        a = CanonicalTransitionBuilder().build_expert_step(record())
        a['action_t']['commands'] = [{'command': 'wheel_velocity', 'module_ids': ['smores_01'],
                                    'parameters': {'left_rad_s': .2, 'right_rad_s': .2}}]
        invalid = copy.deepcopy(a)
        invalid['action_t']['commands'][0]['parameters']['left_rad_s'] = .4
        invalid['supervision'].update(command_mask=[False], valid_for_behavior_cloning=False,
                                      sample_weight=0, exclusion_reason='source_action_invalid')
        rows = list(resolve_continuous_revisions([a,invalid]))
        self.assertEqual(rows[0]['supervision']['command_mask'], [True])
        self.assertEqual(rows[1]['supervision']['exclusion_reason'], 'source_action_invalid')

    def test_internal_velocity_revision_preserves_wheels_and_discrete_joint_goals(self):
        from preprocessing import resolve_continuous_revisions
        a = CanonicalTransitionBuilder().build_expert_step(record())
        discrete = copy.deepcopy(a['action_t']['commands'][0])
        discrete['module_ids'] = ['smores_02']
        a['action_t']['commands'] = [
            {'command': 'pan_velocity', 'module_ids': ['smores_01'],
             'parameters': {'velocity_rad_s': .2}},
            {'command': 'wheel_velocity', 'module_ids': ['smores_01'],
             'parameters': {'left_rad_s': .3, 'right_rad_s': .3}},
            discrete,
        ]
        a['supervision']['command_mask'] = [True, True, True]
        b = copy.deepcopy(a)
        b['action_t']['commands'] = [
            {'command': 'tilt_velocity', 'module_ids': ['smores_01'],
             'parameters': {'velocity_rad_s': .4}},
        ]
        b['supervision']['command_mask'] = [True]
        rows = list(resolve_continuous_revisions([a, b]))
        self.assertEqual(rows[0]['supervision']['command_mask'], [False, True, True])
        self.assertEqual(rows[0]['supervision']['sample_weight'], 1)
        self.assertEqual(rows[0]['action_t']['commands'][2], discrete)
        self.assertEqual(rows[1]['supervision']['command_mask'], [True])

    def test_primitive_is_emitted_only_at_its_own_dispatch(self):
        tracker = ActionDecisionTracker()
        first = adapt_action(record(), decision_tracker=tracker)
        second = adapt_action(record(2.), decision_tracker=tracker)
        self.assertEqual(first['commands'][0]['command'], 'set_tilt')
        self.assertEqual(first['commands'][0]['parameters']['angle_rad'], .4)
        self.assertEqual(second['commands'], [])

    def test_direct_wheels_remain_during_running_primitive(self):
        tracker = ActionDecisionTracker()
        adapt_action(record(), decision_tracker=tracker)
        row = record(2.)
        row['expert_action']['locomotion'] = {'smores_01': {'vx': .02, 'yaw_rate': .1}}
        result = adapt_action(row, decision_tracker=tracker)
        self.assertEqual([c['command'] for c in result['commands']], ['wheel_velocity'])
        params = result['commands'][0]['parameters']
        self.assertNotEqual(params['left_rad_s'], params['right_rad_s'])
        self.assertNotIn('source_action', result)

    def test_align_faces_keeps_both_modules_and_faces(self):
        row = record(primitive='align_faces')
        goal = row['expert_action']['primitive_goal']
        goal['module_ids'] = ['smores_01', 'smores_02']
        goal['parameters'] = {'face_a': 'TOP', 'face_b': 'BOTTOM', 'clocking_quarter_turns': 2}
        command = adapt_action(row)['commands'][0]
        self.assertEqual(command['module_ids'], goal['module_ids'])
        self.assertEqual(command['parameters'], goal['parameters'])

    def test_macro_name_is_rejected_as_a_primitive(self):
        with self.assertRaises(PreprocessingError):
            adapt_action(record(primitive='restore_drive'))

    def test_recorded_ik_sequence_has_commands_but_no_instantaneous_bc(self):
        row = record()
        goal = row['expert_action'].pop('primitive_goal')
        row['expert_action']['joint_configuration'] = {'target_rad': {'arm.tilt': .4}}
        row['expert_action']['primitive_sequence'] = [goal, {
            'goal_id': 'g2', 'primitive': 'rotate_pan_by', 'module_ids': ['smores_01'],
            'parameters': {'delta_rad': .2}}]
        transition = CanonicalTransitionBuilder().build_expert_step(row)
        self.assertEqual(transition['action_t']['schedule'], 'recorded_sequence')
        self.assertEqual([c['command'] for c in transition['action_t']['commands']],
                         ['set_tilt', 'rotate_pan_by'])
        self.assertTrue(transition['supervision']['valid_for_behavior_cloning'])
        self.assertTrue(transition['supervision']['configuration_mask'])
        self.assertFalse(any(transition['supervision']['command_mask']))
        self.assertEqual(transition['action_t']['configuration_target_rad'], {'arm.tilt': .4})
        self.assertEqual(transition['supervision']['command_exclusion_reason'],
                         'intermediate_command_states_unavailable')

    def test_different_dispatch_times_keep_different_states(self):
        builder = CanonicalTransitionBuilder()
        first = builder.push(record(1., goal_id='g1'))
        second = builder.push(record(2., goal_id='g2'))
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0]['graph_t']['stamp'], 1.)
        self.assertEqual(second[0]['graph_t']['stamp'], 2.)

    def test_graph_drops_aliases_but_keeps_dynamic_joint_and_face_state(self):
        source = graph(1.)
        untouched = copy.deepcopy(source)
        self.assertTrue(hasattr(preprocessing, 'compact_robot_graph'))
        result = preprocessing.compact_robot_graph(source)
        node = result['nodes'][0]
        self.assertEqual(node['faces']['TOP']['state'], 'available')
        self.assertEqual(node['actuators']['tilt'], {'position_rad': .2, 'velocity_rad_s': .1})
        self.assertNotIn('attributes', node)
        self.assertNotIn('pose', node)
        self.assertNotIn('capabilities', node)
        self.assertNotIn('global_attributes', result)
        self.assertEqual(source, untouched)

    def test_attached_edges_make_their_faces_unavailable_even_with_a_stale_node_snapshot(self):
        source = graph(1.)
        other = copy.deepcopy(source['nodes'][0])
        other['module_id'] = other['node_id'] = 'smores_02'
        source['nodes'].append(other)
        source['edges'] = [{'module_a_id': 'smores_01', 'module_b_id': 'smores_02',
                            'attributes': {'face_a': 'TOP', 'face_b': 'TOP', 'is_attached': True}}]
        result = preprocessing.compact_robot_graph(source)
        self.assertEqual([n['faces']['TOP']['state'] for n in result['nodes']], ['connected','connected'])

    def test_edge_relative_pose_uses_source_module_frame(self):
        source = graph(1.)
        source['nodes'][0]['attributes']['orientation'] = [0., 0., math.sqrt(.5), math.sqrt(.5)]
        other = copy.deepcopy(source['nodes'][0])
        other['module_id'] = other['node_id'] = 'smores_02'
        other['attributes']['position'] = [1., 0., 0.]
        other['attributes']['connectors'].append({'connector_id': 'BOTTOM', 'status': 'available'})
        source['nodes'].append(other)
        source['edges'] = [{'module_a_id': 'smores_01', 'module_b_id': 'smores_02',
                            'attributes': {'face_a': 'TOP', 'face_b': 'BOTTOM',
                                           'is_attached': True}}]
        self.assertTrue(hasattr(preprocessing, 'compact_robot_graph'))
        edge = preprocessing.compact_robot_graph(source)['edges'][0]
        self.assertAlmostEqual(edge['relative_pose']['position'][0], 0.)
        self.assertAlmostEqual(edge['relative_pose']['position'][1], -1.)
        self.assertEqual(edge['face_b'], 'BOTTOM')

    def test_transient_observation_does_not_copy_world_geometry(self):
        transition = CanonicalTransitionBuilder().build_expert_step(record())
        observation = transition['observation_t']
        self.assertEqual(observation['task_type'], 'gap')
        self.assertIsNone(observation['relative_geometry'])
        self.assertNotIn('geometry_world', str(observation))

    def test_restore_drive_is_context_and_its_joint_command_is_the_target(self):
        row = record()
        row['debug'] = {'behavior': 'restore_drive'}
        row['_loader_source_path'] = 'assembly.jsonl'
        row['_loader_source_row'] = 17
        result = CanonicalTransitionBuilder().build_expert_step(row)
        self.assertEqual([c['command'] for c in result['action_t']['commands']], ['set_tilt'])
        self.assertTrue(result['supervision']['valid_for_behavior_cloning'])
        self.assertEqual(result['provenance']['behavior_context'], 'restore_drive')
        self.assertEqual(result['provenance']['source_row'], 17)

    def test_absent_wheels_are_not_zero_commands_but_explicit_stop_is(self):
        row = record()
        self.assertEqual(len(adapt_action(row)['commands']), 1)
        row['expert_action'] = {'locomotion': {'smores_01': {'vx': 0., 'yaw_rate': 0.}}}
        command = adapt_action(row)['commands'][0]
        self.assertEqual(command['command'], 'wheel_velocity')
        self.assertEqual(command['parameters'], {'left_rad_s': 0., 'right_rad_s': 0.})

    def test_conflicting_internal_commands_are_rejected(self):
        row = record()
        row['expert_action']['locomotion'] = {'smores_01': {'pan_rate_rad_s': .1}}
        with self.assertRaises(PreprocessingError):
            adapt_action(row)

    def test_unknown_commands_fail_instead_of_silently_losing_a_target(self):
        row = record()
        row['expert_action']['locomotion'] = {'smores_01': {'unmapped_joint': .1}}
        with self.assertRaises(PreprocessingError):
            adapt_action(row)

    def test_cone_relative_position_changes_and_ground_distance_is_pitch_invariant(self):
        def relative(root, orientation):
            return preprocessing._navigation_cones_relative(
                [[3., 4.]], floor_z=0., cone_height_m=.4,
                root_xyz=root, root_orientation=orientation)[0]
        initial = relative([0., 0., 0.], [0., 0., 0., 1.])
        tilted = relative([0., 0., 0.], [0., math.sqrt(.5), 0., math.sqrt(.5)])
        moved = relative([1., 0., 0.], [0., 0., 0., 1.])
        self.assertEqual(initial['distance_xy_m'], 5.)
        self.assertEqual(tilted['distance_xy_m'], 5.)
        self.assertNotEqual(initial['center_relative_xyz_m'], moved['center_relative_xyz_m'])
        observation = preprocessing.compact_observation({'active_task': {
            'type': 'flat_navigation', 'geometry': {'cones_world_xy_m': [[3., 4.]],
            'cone_radius_m': .1, 'cones_relative': [moved]}}})
        self.assertEqual(list(observation['relative_geometry']), ['cones_relative'])

    def test_materializer_rejects_mismatched_module_resource_masks(self):
        from materialize_canonical_dataset import _transition_stats
        row = preprocessing.CanonicalTimestepBuilder().build_bucket([record()])
        row['provenance']['canonical_transition_index'] = 0
        self.assertTrue(_transition_stats(row, expected_index=0)[0])
        del row['supervision']['module_resource_masks']['smores_01']['internal']
        with self.assertRaises(ValueError):
            _transition_stats(row, expected_index=0)

    def test_materializer_preserves_compact_repeat_coverage_for_one_physical_timestep(self):
        from dataset_loader import EpisodeSpec
        from materialize_canonical_dataset import _materialize_episode
        source = record()
        source['_source_repeat_count'] = 100
        row = preprocessing.CanonicalTimestepBuilder().build_bucket([source])
        row['provenance']['canonical_transition_index'] = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = EpisodeSpec('expert:test', 'expert', root)
            with patch('materialize_canonical_dataset.iter_canonical_episode', return_value=iter([row])):
                entry = _materialize_episode(spec, repo_root=root, episodes_dir=root)
            self.assertEqual(entry['canonical_transitions'], 1)
            self.assertEqual(entry['effective_weight'], 100)
            self.assertEqual(entry['bc_effective_weight'], 100)

    def test_source_phase_completion_does_not_end_logical_episode(self):
        from canonical_dataset import iter_canonical_episode
        from dataset_loader import EpisodeSpec
        assembly = record(1.)
        assembly['done'] = True
        behavior = record(2., goal_id='behavior-goal')
        behavior['_loader_phase'] = 'behavior'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = EpisodeSpec('expert:test', 'expert', root)
            with patch('canonical_dataset.iter_episode_records', return_value=iter([assembly, behavior])):
                rows = list(iter_canonical_episode(spec, repo_root=root))
        self.assertTrue(rows[0]['episode_state']['phase_done_observed'])
        self.assertFalse(rows[0]['episode_state']['done'])
        self.assertTrue(rows[1]['episode_state']['done'])

    def test_overwrite_restores_previous_dataset_if_publish_is_interrupted(self):
        from materialize_canonical_dataset import _publish_materialized_dataset
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'canonical'
            staging = Path(directory) / 'staging'
            output.mkdir()
            staging.mkdir()
            (output / 'manifest.json').write_text('previous')
            (staging / 'manifest.json').write_text('new')
            replace = Path.replace
            def interrupted(path, target):
                if path == staging:
                    raise KeyboardInterrupt()
                return replace(path, target)
            with patch.object(Path, 'replace', interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    _publish_materialized_dataset(staging, output, overwrite=True)
            self.assertEqual((output / 'manifest.json').read_text(), 'previous')
            _publish_materialized_dataset(staging, output, overwrite=True)
            self.assertEqual((output / 'manifest.json').read_text(), 'new')


if __name__ == '__main__':
    unittest.main()
