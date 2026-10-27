# Runtime gate recovery

Incident 1: the original gate failed during collection with two `ModuleNotFoundError: smores_ep` errors, before runtime assertions. Original runtime_test_result.txt and runtime_test_results.json are preserved.

The package declares its source directory in scripts/smores_ep/pyproject.toml. The documented working invocation in docs/teleoperation_v1_runbook.md uses scripts/smores_ep/src and mssr_ws/src/mssr_expert; scripts/smores_ep/run_bc_live_smoke.py also inserts those roots. The audit-local wrapper inserted only the repository root.

Minimal fix: insert those verified roots in the audit wrapper. Use fresh runtime_gate_recovery_work and recovery result/command/log paths to retain the original failure evidence. No production, test or scientific configuration changed.

Retry: same three supplemental test modules, CPU-only, existing sealing guard, original skip/environment policy. Full 2,077-test suite is not rerun because source/test/config hashes remain unchanged. Result: 41 passed in 0.60 seconds, exit 0; no failures, errors or skips; real runtime assertions executed. TEST payload access 0; denied attempts 0. One harness incident, resolved on the first fix/retry.


Incident 2: initial staged whitespace gate reported only five audit files: two generated TSV inventories (default CSV CRLF) and three raw Git transcripts (two frozen TEST reports plus this audit's original patch). Every one of the 612 raw-transcript findings is a literal single-space blank diff context line, not source whitespace. Original raw check log remains local-only.

Minimal audit-only remedy: generate inventories with LF. For exactly the three raw transcripts, use a per-command core.attributesFile pointing to this audit's whitespace_policy.gitattributes, disabling only blank-at-eol checking for those evidence files. No repository config, root .gitattributes, frozen report or source is modified. Production/tests/configs keep strict default whitespace checks. This is a byte-preserving evidence-format exception, not a production fix. The inventory also includes staged names when regenerated after staging, so newly staged additions remain in the selection.

Incident 2 retry: staged whitespace check passed with the documented exact-transcript policy; unmodified source default check passed. No runtime test rerun was needed for audit-only formatting/policy changes. Two distinct incidents resolved; no failed fix retries.
