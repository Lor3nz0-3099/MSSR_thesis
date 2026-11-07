"""CPU-only viewer; invoked once or by a human's watch command. Never launches Isaac."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

CAUSES = ('masked_busy', 'dropped_intra_step_conflict', 'executor_rejected_busy')


def rows(path):
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # A final append may still be in progress.


def show_progress(run, target=256, updates=2):
    path = run / 'c1_discovery/action_summary.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    completed = max((int(row.get('update', 0)) for row in rows(run / 'updates.jsonl')), default=0)
    print(f"transitions: {data.get('row_transitions', 0)} / {target}; PPO updates: {completed} / {updates}")
    for label, key in (
        ('structural sampled', 'structural_sampled_count'), ('BUSY masked', 'masked_busy_count'),
        ('unavailable masked', 'masked_unavailable_count'), ('optional membership masked', 'masked_optional_membership_count'),
        ('intra-step conflict drops', 'intra_step_conflict_drop_count'),
        ('structural projected', 'structural_projected_count'), ('structural executed', 'structural_executed_count'),
        ('primitive started', 'primitive_started_count'), ('primitive completed', 'primitive_completed_count'),
        ('primitive failed/canceled', 'primitive_failed_count'), ('executor BUSY fallback', 'executor_busy_rejection_count'),
        ('valid connection max', 'valid_connection_max'), ('C1 success', 'C1_success_count'), ('mean reward', 'mean_reward')):
        print(f'{label}: {data.get(key, 0 if key != "mean_reward" else None)}')


def show_details(run, limit=8):
    examples = defaultdict(list)
    pairs, busy, tasks, counts = Counter(), Counter(), Counter(), Counter()
    for row in rows(run / 'c1_discovery/action_events.jsonl'):
        outcome = row.get('outcome')
        counts[outcome] += 1
        if outcome in CAUSES and len(examples[outcome]) < limit:
            examples[outcome].append(row)
        if outcome == 'masked_busy':
            busy[row.get('primitive')] += 1
        if outcome == 'dropped_intra_step_conflict':
            tasks[row.get('task')] += 1
            for peer in row.get('conflicts_with', ()):
                pairs[tuple(sorted((row.get('candidate_id', '?'), peer.get('candidate_id', '?'))))] += 1
    for cause in CAUSES:
        print(f'\n{cause}:')
        for row in examples[cause]:
            print(json.dumps({k: row.get(k) for k in ('task', 'env_index', 'generation', 'timestamp_s',
                'candidate_id', 'primitive', 'module_ids', 'faces', 'connectors', 'resource_claims',
                'raw_presence_logit', 'calibrated_presence_probability', 'drop_reason', 'conflicts_with')}, ensure_ascii=False))
    print('\nTop intra-step conflict pairs:', json.dumps(pairs.most_common(limit)))
    print('Detailed BUSY mask events by primitive (bounded sample):', json.dumps(busy))
    print('Intra-step conflict drops by task:', json.dumps(tasks))
    print('Detailed lifecycle/outcome counts:', json.dumps(counts))
    path = run / 'c1_discovery/action_summary.json'
    print('Unbounded aggregate counts:', path.read_text() if path.exists() else 'not available yet')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--progress', action='store_true')
    mode.add_argument('--details', action='store_true')
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--target-transitions', type=int, default=256)
    parser.add_argument('--target-updates', type=int, default=2)
    args = parser.parse_args(argv)
    if args.progress:
        show_progress(args.run, args.target_transitions, args.target_updates)
    else:
        show_details(args.run, args.limit)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
