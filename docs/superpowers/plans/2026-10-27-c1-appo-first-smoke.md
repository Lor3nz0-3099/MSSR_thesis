# C1 physical calibration and fixed-alpha APPO smoke

User-authorized implementation plan, 2026-10-27. No commit/push and no long
training. The pasted specification in this session is authoritative.

## Frozen contracts

- Work on `snake8-global-path-ik-recovery`; preserve existing dirty files.
- Physics 240 Hz. TRAIN-only calibration; validation/test/challenge excluded
  from scale fitting. Challenge geometry may be unit-tested only.
- Expert performs assembly only, then independent topology verification.
- MM8 success is physical depression >=0.0035 m with intended integrity;
  no deterministic approach and no return-to-spawn requirement.
- BC-v4 SHA256:
  `d90a73e9b5ba2ecb579e4fc8733066b1c55d9e60dafeb1d71c029c49a7c21c6d`.
- RL Bernoulli sampling does not use the deterministic BC threshold 4.5.
- APPO means altruistic PPO, fixed alpha; reuse MSSRMAPPO and both altruistic
  critics. Scalar weights remain separate from raw evaluators.

## Macro tasks and verification

1. Button: RED physical contract/stroke/arbitrary yaw/runtime displacement;
   generalize the existing joint installer, then scoped legacy regression.
   Root performs an actual oblique TRAIN physical press smoke.
2. Rewards: RED impossible depression, manipulation closed loops, gap eight
   owners/signed entry, RC/stairs adversarial cases; implement endpoint
   potentials where necessary and validate terminal/discount semantics.
3. Causality/lifecycle: RED pre-application frames, accepted/rejected/empty
   proposals, causal post-state, carry continuity, timeout and terminal gate;
   use existing command/goal IDs and a native application acknowledgement.
4. Signals: explicit measured/absent/unavailable/invalid metadata, sourced
   native rejection and commanded effort; collision unavailable unless a
   reliable contact source is connected; physical-time stagnation detector.
5. Calibration: observer over completed transitions, TRAIN-only manifest,
   signed/nonzero/missing statistics, episode returns and correlations.
   Run a small physical smoke, then two TRAIN seeds per difficulty/domain.
6. APPO: derive an initial versioned config from physical calibration only;
   connect owner rewards, fixed-alpha sidecars, GAE and existing update;
   verify optimizer changes and finite diagnostics; save APPO_SMOKE artifact.

## Integration boundaries

Independent work: button USD/publisher; raw reward contracts; calibration
statistics/recorder. Root owns native stepping/causality, physical collection,
signal wiring and APPO runner. No competing edits to shared scenario/learner.
All runtime launches use scoped verified-process cleanup first. Unit fixtures
and synthetic diagnostics are never labelled empirical calibration.

## Progress / decisions

- Initial audit: four tracked dirty files and 25 untracked files preserved in
  a separate temporary baseline snapshot. Git diff --check is clean.
- All six macro tasks are complete. Physical results, calibration/config paths,
  alpha, diagnostics and final readiness are recorded below and in the results report.

### Verified progress

- Macro 1: 152 scoped USD/legacy tests passed. Real Isaac contact smoke on
  TRAIN manipulation seed 20000 at 240 Hz measured maximum depression
  0.00399988035205207 m, exceeding 0.0035 m while respecting the 4 mm stroke.
  Evidence: `logs/appo_smoke/button_train20000_physics.json`.
- Macro 2: 120 reward/transition tests passed; endpoint potentials accept
  explicit shaping_gamma and true terminated flag. Raw defaults remain gamma1.
- Macro 3: 143 causal/lifecycle/legacy tests passed; production collection uses
  opt-in native BC lockstep and per-command physical application completion.
- Macro 4: 176 scoped tests passed including 29 signal availability/stagnation
  tests. Unmeasured collision/redundancy/effort are not fitted as measured zeros.
- Calibration/config core and learner diagnostics/checkpoint scoped suite:244
  passed. No config is fitted until physical TRAIN collection completes.


### Continuation 2026-10-28

- Prior physical navigation TRAIN7000 completed: Expert terminal success,
  independent rc_car8 topology, eight 8-tick causal transitions at 240 Hz,
  and collector truncation. Preserved evidence in
  `logs/appo_smoke/navigation_bootstrap_train7000`. All sampled proposals
  conflicted and native packets were empty: this is bootstrap evidence,
  **not** proof of MARL actuation nor sufficient APPO readiness.
- ROS launch now forwards exact C1 course/split/seed for all four domains.
  Three new RED checks became GREEN; six launch wiring checks pass.
- Continuation scoped regression: 180 button/reward/signal/causality tests
  passed. No reimplementation of prior GREEN components.
- Review found time-limit bias in altruistic GAE. Three new RED tests require
  terminal-only bootstrap masking and terminated|truncated trace masking.
  Implemented captured per-transition next critic graphs and next owner-row
  mapping; 116 focused native/GAE/learner tests pass. Truncations bootstrap
  their own final physical state, never a reset/other task state.
- Review found new goals could be acknowledged before the 4-tick primitive
  executor refresh for 1..3-tick intervals. First application now forces an
  executor refresh before drive and physics, preserving ordinary cadence.
- Gap scalarization broadcasts only explicitly global absent owner fields
  (success/time/stagnation); unavailable terms retain exclusion metadata,
  and invalid signal status fails closed. 69 focused tests pass.
- Current host disk stalls (`rq_qos_wait`, IO pressure ~79%) made runtime
  file fsyncs prohibitively slow. First stairs attempt was interrupted and
  scoped processes reaped; retry uses `/dev/shm` for communication files.
  Durable completed evidence is copied into repository logs.

- Physical stairs TRAIN12000 assembled and matched snake8 independently,
  then exposed a runner adapter defect before any MARL transition: navigation
  goal features were injected into non-navigation task contexts and rejected
  by the frozen strict BC whitelist. Fixed only runner projection; all four
  generated C1 task views and saved physical stairs frame pass BC preprocessing.
- First calibration will deliberately use four TRAIN layouts (7000,12000,
  16000,20000), up to256decisions each. Latest user continuation explicitly
  permits a preliminary subset to prioritize a real learner update. The full
  two-seeds-per-difficulty 24-layout pilot remains available by default.
- Unconstrained native-valid Bernoulli proposal probability on the saved RC
  physical frame is about0.0148. Preserve stochastic warm start and reject
  whole conflicting proposals without repair. Budget256steps increases chance
  of native nonempty action without introducing BC4.5 threshold or resampling.

- Final preliminary fit corpus:1024 valid physical TRAIN transitions,256
  each on navigation7000,stairs12000,gap16000,manipulation20000. All four
  episodes end by physical-horizon truncation; no task success claimed.
  Nonempty accepted native proposals by domain:194/96/168/51 respectively.
  The entire old divergent stairs episode is excluded, not clipped/relabelled.
  Reproduction wrapper calls existing CalibrationDataset/derive_initial_config:
  `logs/appo_smoke/calibration_validated_train/fit_preliminary.py`.
- RewardConfig v2 identifies continuous normalized safety distances separately
  from event/count/fraction costs. A dense signal below1e-5rawunits retains
  measured statistics and falls back to natural scale1: resting button noise
  (~1.7e-7depression_progress) cannot create a million-fold reward amplifier.
- Initial scales+weights saved in
  `logs/appo_smoke/calibration_validated_train/reward_config.initial.json`;
  balanced/stronger_safety/stronger_sparse_success presets exist. TRAIN-only
  schema/provenance validates all rows; no VAL/TEST/CHALLENGE fitting.
- 70/10/10/10 mixing magnitude proxy shares:76.12/5.36/13.28/5.25percent.
  Scalar reward density1 includes time/shaping/costs and is not success density.
  This is a reward-magnitude proxy, not a claim to measure actual gradient share.
- Root changed owned Expert shutdown order to TERM then KILL (3RED→17GREEN),
  matching requested scoped cleanup. No unrelated process termination.
- APPO physical smoke now running: up to64steps/domain, rollout256,
  fixedalpha0.5,gamma0.99,radius0.5m, actorBCvisibility1m, initialGaussian
  logstd−2, no deterministic threshold4.5. No longC1-A run started.


### Final physical acceptance

- APPO_SMOKE_fixed_alpha_v1:256physical transitions (64/domain),73native
  nonempty accepted proposals,ONE actual actor/owner/MF optimizer update.
  Losses actor10.0534982681,owner247.700012207,MF219.843078613;
  gradient norms26.7991450591/318.656662190/337.229880379, all nonzero.
  Entropy−0.8511800766,ratio1,KL0,clipfraction0; all parameters finite/changed.
- Checkpoint independently loaded:2641finite tensors,585Adam step tensors
  all exactly1. Causal identity/durations/continuity audited across all256rows.
  BC hash unchanged; separate APPO_SMOKE.pt saved; no long training started.
- Final scoped suite606passed; diffcheckclean. Owned simulator/Expert/ROS
  processes absent after scoped cleanup. Detailed report:
  `docs/superpowers/plans/2026-10-28-c1-appo-smoke-results.md`.
- READY FOR FIRST C1-A TRAINING: YES. Preliminary4easy-layout calibration
  remains explicit; the24-layout pilot/final reward tuning are not claimed.

- Future C1-A seed selection corrected: all1000 frozen TRAIN seeds/domain,
  reproducibly shuffled with seed0 and independent per-domain episode cursors.
  Calibration/smoke pools unchanged. New outcome logging distinguishes task
  success, failure and horizon truncation for held-out VALIDATION assessment.
  Final verification:606tests passed in9.38s; runner50tests passed.
  Prepared TRAIN/VALIDATION commands passed argument/shell syntax preflight;
  neither was executed. Full commands in logs/appo_smoke/first_c1_a_commands.sh.
