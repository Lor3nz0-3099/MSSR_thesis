# Expert dataset pipeline

`build_expert_v1.py` discovers completed successful expert runs, imports one
canonical run per `(task, seed)` into `datasets/expert_v1`, and derives
`datasets/expert_v1_compact`.

Run it from the repository root after a campaign finishes:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/dataset_tools/build_expert_v1.py
```

The default raw copy mode uses hard links because `logs/` and `datasets/` are
on the same filesystem. This avoids duplicating tens of GB while preserving the
dataset if the original log pathname is later deleted. Completed source JSONL
files must therefore be treated as immutable. Use `--copy-mode physical` when
independent file inodes are required.

An episode is imported only when its success artifact is present, all required
phase files are non-empty and older than that artifact, and every row passes
the transition/alias checks. Stair recovery episodes may contain only behavior:
Snake8 self-assembly is invariant and already represented by the canonical
assembly corpus. Button episodes require assembly, behavior, and manipulation
plus a certified physical press and successful return to RC-Car8.

The compact dataset retains `graph_t`, `expert_action`, `graph_t_plus_1`, all
node/edge attributes, graph features, task/FSM context, supervision, and module
roles/assignments. Exact aliases and graph-derivable observation views are
removed only after record-by-record validation. Constants move to phase
metadata only when exact across the whole stream. Consecutive semantic repeats
are represented with source spans and `_source_repeat_count`.

For behavior cloning, train only on rows with `action_valid == true` and
`supervision.valid_for_behavior_cloning == true`. Terminal rows without a valid
expert action are retained to preserve episode boundaries, not as action
targets.

See `datasets/expert_v1/import_audit.json` for selections, rejected attempts,
same-seed runs, validation counts, and pending campaigns.
