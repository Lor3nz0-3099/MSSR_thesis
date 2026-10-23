# Conservative pre-MARL plan

1.Inventory and classify every relevant file and tracked delta.2.Preserve local scientific history; untrack only explicit classified historical paths.3.Keep official best/training/Test artifacts byte-identical.4.Apply readable generated-artifact ignore policy.5.Run exact previous sealed CPU suite, with audit-local wrapper/TEST guard.6.Stage explicit classified paths and inspect content/size.7.One local commit; never push.8.Verify HEAD checkpoint blob and preserved history.

No deletion planned: temporary files will be retained/ignored unless removal is demonstrably necessary. Ambiguous/unrelated changes remain untouched and unstaged.

Ruling: use this user-requested audit as the task ledger; no new worktree or separate skill workspace. This is repository packaging, not feature implementation. No production fixes or new test semantics are authorized.
