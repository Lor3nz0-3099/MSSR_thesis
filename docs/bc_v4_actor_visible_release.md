# Frozen actor-visible BC-v4 release for MAPPO

The official from-scratch run is
`artifacts/bc_v1/runs/20261022T232532_854802073Z_207309_bc_v4_actor_visible_r1m_scratch_final`.
Its selected checkpoint is `best.pt`, epoch 5, validation objective
`0.06868529843473761`, SHA256
`d90a73e9b5ba2ecb579e4fc8733066b1c55d9e60dafeb1d71c029c49a7c21c6d`.
The existing strict absolute improvement rule uses `min_delta=0.002` and
`patience=8`; training completed at epoch 13. The checkpoint selection was
frozen before TEST. The final `last.pt` stays on the originating machine and
is intentionally excluded from Git.

The official one-shot TEST report is
`artifacts/bc_v1/audits/20261023T073725_980736Z_bc_v4_official_one_shot_test_evaluation`.
It records 18,536 samples, 12 episodes, 290 batches, loss
`0.07753449015633553`, F1 `0.8230016687845431`, exact-set
`0.48225075528700906`, and verdict `SUITABLE_FOR_MAPPO_WARM_START`.
TEST is consumed and frozen: do not rerun it for selection, thresholds,
checkpoint comparisons or BC retraining. This Git packaging task does not
perform evaluation, training or rollout.

The verdict concerns actor initialization. Excess false-positive actions,
stairs performance and rare structural actions remain limitations documented
in the frozen assessment; these offline metrics do not establish autonomous
task success. Exact-set concerns action membership; payload errors are
reported separately.

The policy uses the existing hardware-independent ActionSet vocabulary and
frozen actor-visible geometry contract: R=1.0 m, `team_union_wireless`, current
task/environment observations, and the approved physical feature pipeline.
Stage/FSM, future topology, configuration goals and expert labels are not new
actor inputs. The previously certified tensor target reconstruction and
immutable-tuple reader optimizations are preserved without modification.

## Versioning and reproduction

`artifacts/bc_v1/` is ignored by default. Only explicit release exceptions are
force-added: the selected checkpoint and its lightweight run metadata, final
training log, full official TEST audit, final training helper, frozen
actor-visible F2 statistics and source/split identity metadata, and the new
pre-MARL Git checkpoint audit. The latter contains the exact file classification
and commit manifest. Already tracked exceptions remain tracked normally.

All previous runs, checkpoints, profiling/debugging audits and manual logs are
superseded diagnostic history and intentionally remain local-only. They are
not deleted. Upstream source metadata included for provenance is not an
additional selected model or a substitute for the official TEST result.
No Git LFS is introduced; the approximately 17 MiB official `best.pt` is a
normal Git blob. Generated datasets, episode caches, SQLite sidecars, `last.pt`,
temporary test outputs and older binary checkpoints are not release files.

The frozen helper is
`artifacts/bc_v1/manual_helpers/20261022T194614_079486Z_bc_v4_reader_final_full_training/manual_train_bc_v4_actor_visible_final.sh`.
Its existing hash pins, missing-data checks and exclusive fresh-output policy
are unchanged. It is preserved as training provenance, not executed by this
packaging task. Its original branch/HEAD admission gate identifies the
pre-release training workspace; the resulting release commit has a different
HEAD. Do not silently relax that gate or interpret its rejection on a later
HEAD as permission to train again.

The checkpoint source manifest fingerprint is
`d80b0774ed1793dedfa101cbaa82e45b7cb460384bfdd36a4291a592e9d97a3f`;
actor visibility fingerprint
`e6b819c2e23e1713f70649d83b547d7701d78ef8c0ec7a21668c803c7c8f6234`;
F2 fingerprint
`bdb5304cd620faa7461c448794044af4757b85e77e672ca7022a1083da9f37cd`.
The training configuration is `configs/bc_v1/training_action_set_v1.json`;
H1/H2 configurations and RobotSpec are retained in `configs/bc_v1/` and in
the official run/evaluation identity records. Full reproduction requires the
externally retained immutable Canonical/G1/G2 data and the pinned upstream
artifacts. Committed manifests authenticate those assets but do not contain
their payloads. Do not rebuild or refit them implicitly when an asset is absent.

The relevant CPU regression command, all original exclusions, TEST access
guard, verification results and preservation records are stored in the
pre-MARL audit. The user performs the eventual push; this milestone creates
one local commit only.
