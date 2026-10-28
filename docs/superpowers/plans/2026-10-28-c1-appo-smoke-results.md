# C1 physical calibration and APPO smoke — 2026-10-28

This report accompanies the unchanged dirty working tree; no commit or push. APPO means Altruistic PPO. The physical learner smoke completed successfully; audit and checkpoint are verified below.

## Physical calibration provenance

1024 real TRAIN transitions, four independent Expert assembly → topology verification → MARL episodes; 256 decisions/episode, 240 Hz physics, 8 ticks/decision (30 Hz). Seeds: navigation7000, stairs12000, gap16000, manipulation20000; all easy. Actor frozen BC-v4 eval mode, no optimizer during calibration. 254 stochastic, one deterministic mean reference and one masked diagnostic perturbation per episode. Owner channels are not independent episodes.

The entire first stairs episode from run20261028T125915Z_ff9e5b10 is excluded: the pre-fix lifecycle continued after a whole-body fall and the physical state later diverged. Its auto-fitted config is quarantined as `.REJECTED_STAIR_FALL_TAIL`. The replacement stairs episode is stable and ends by horizon truncation. All four retained episodes truncate; none achieved task success.

Reproduce fitting with `PYTHONPATH=$PWD:$PWD/scripts/smores_ep/src:$PWD/mssr_ws/src/mssr_expert /home/lorenzo/isaaclab-env-beta2/bin/python logs/appo_smoke/calibration_validated_train/fit_preliminary.py`. This wrapper uses the common CalibrationDataset/RewardConfig APIs and preserves original selected JSONL rows; source SHA256 hashes and exclusions are embedded in config provenance. No VAL/TEST/CHALLENGE data fit scales.

## Normalization and initial preferences

RewardConfig version2 is explicitly a **calibration-derived initial APPO smoke config**. Dense scales are nonzero absolute p95 with at least three measured samples. Amplitude below1e-5rawunits uses natural-unit scale1 and an insufficient-signal status: this conservative numerical-resolution assumption is recorded, not a learned optimum. Event/count/fraction channels retain identity without inverse-frequency weighting. Continuous safety distances remain continuous. Uninstrumented terms have no scale and zero weight. Scalarization preserves excluded availability metadata and only explicitly shared absent owner terms may inherit team values.

The weights express an initial balanced preference: total progress-channel share+1, success+5, goal entry+1, safety−1, time−0.01. They are not fitted optimal preferences. Stronger safety and stronger sparse success presets are saved for future comparison on VALIDATION. The full24-layout, three-difficulty pilot remains deferred under the continuation request.

### navigation

| Raw term | Scale | Weight | Normalization | Evidence status |
|---|---:|---:|---|---|
| collision | unavailable | 0 | excluded | excluded_unavailable |
| conflict | 1 | -1 | event_identity | measured_event |
| control_effort | unavailable | 0 | excluded | excluded_unavailable |
| invalid | 1 | -1 | event_identity | insufficient_event_not_observed |
| outside_road | 1 | -1 | unit_fallback | insufficient_nonzero_support |
| progress | 0.000578998217 | 1 | nonzero_abs_p95 | empirical |
| redundant_action | unavailable | 0 | excluded | excluded_unavailable |
| stagnation | 1 | -1 | event_identity | measured_event |
| success | 1 | 5 | event_identity | insufficient_event_not_observed |
| target_entry | 1 | 1 | event_identity | insufficient_event_not_observed |
| time | 1 | -0.01 | decision_identity | measured_event |

### stairs

| Raw term | Scale | Weight | Normalization | Evidence status |
|---|---:|---:|---|---|
| below_support | 1 | -1 | unit_fallback | insufficient_nonzero_support |
| collision | unavailable | 0 | excluded | excluded_unavailable |
| conflict | 1 | -1 | event_identity | measured_event |
| control_effort | unavailable | 0 | excluded | excluded_unavailable |
| goal_entry | 1 | 1 | event_identity | insufficient_event_not_observed |
| invalid | 1 | -1 | event_identity | insufficient_event_not_observed |
| level_progress | 1 | 0.333333 | unit_fallback | insufficient_nonzero_support |
| outside_width | 1 | -1 | unit_fallback | insufficient_nonzero_support |
| profile_progress | 0.00617483235 | 0.333333 | nonzero_abs_p95 | empirical |
| redundant_action | unavailable | 0 | excluded | excluded_unavailable |
| stagnation | 1 | -1 | event_identity | insufficient_event_not_observed |
| success | 1 | 5 | event_identity | insufficient_event_not_observed |
| time | 1 | -0.01 | decision_identity | measured_event |
| vertical_progress | 0.164574687 | 0.333333 | nonzero_abs_p95 | empirical |

### gap

| Raw term | Scale | Weight | Normalization | Evidence status |
|---|---:|---:|---|---|
| approach_progress | 0.0205571693 | 0.333333 | nonzero_abs_p95 | empirical |
| collision | unavailable | 0 | excluded | excluded_unavailable |
| conflict | 1 | -1 | event_identity | measured_event |
| control_effort | unavailable | 0 | excluded | excluded_unavailable |
| crossing_progress | 1 | 0.333333 | unit_fallback | insufficient_nonzero_support |
| fall | 1 | -1 | event_identity | insufficient_event_not_observed |
| goal_entry | 1 | 1 | event_identity | insufficient_event_not_observed |
| invalid | 1 | -1 | event_identity | insufficient_event_not_observed |
| landing_progress | 1 | 0.333333 | unit_fallback | insufficient_nonzero_support |
| redundant_action | unavailable | 0 | excluded | excluded_unavailable |
| stagnation | 1 | -1 | event_identity | measured_event |
| success | 1 | 5 | event_identity | insufficient_event_not_observed |
| time | 1 | -0.01 | decision_identity | measured_event |

### manipulation

| Raw term | Scale | Weight | Normalization | Evidence status |
|---|---:|---:|---|---|
| approach_progress | 0.00843359466 | 0.25 | nonzero_abs_p95 | empirical |
| collision | unavailable | 0 | excluded | excluded_unavailable |
| conflict | 1 | -1 | event_identity | measured_event |
| connection_loss | 1 | -1 | event_identity | insufficient_event_not_observed |
| control_effort | unavailable | 0 | excluded | excluded_unavailable |
| depression_progress | 1 | 0.25 | unit_fallback | insufficient_signal_above_numerical_floor |
| ee_surface_progress | 1 | 0.25 | unit_fallback | insufficient_nonzero_support |
| invalid | 1 | -1 | event_identity | insufficient_event_not_observed |
| orientation_progress | 1 | 0.25 | unit_fallback | insufficient_nonzero_support |
| redundant_action | unavailable | 0 | excluded | excluded_unavailable |
| stagnation | 1 | -1 | event_identity | insufficient_event_not_observed |
| success | 1 | 5 | event_identity | insufficient_event_not_observed |
| time | 1 | -0.01 | decision_identity | measured_event |

## Common signal availability

| Signal | Navigation | Stairs | Gap | Manipulation | Source / units / attribution |
|---|---|---|---|---|---|
| collision | unavailable | unavailable | unavailable | unavailable | No connected reliable collision sensor; excluded, never measured zero |
| conflict | measured | measured | measured | measured | Native admission or sampled-set executability guard; event/fraction; actual competing owners |
| invalid | measured absent | measured absent | measured absent | measured absent | Native INVALID_GOAL or bounded-payload guard; event/fraction; offending owners only |
| redundant_action | unavailable | unavailable | unavailable | unavailable | No connected redundancy classifier; excluded |
| control_effort | unavailable | unavailable | unavailable | unavailable | Incomplete actual interval actuator-rate channel; excluded; no energy claim |
| stagnation | measured after warmup | measured after warmup | measured after warmup | measured after warmup | Centroid displacement over1s physical time, threshold5mm, event; shared task signal. Warmup unavailable |
| time | measured | measured | measured | measured | One fixed8tick decision; physical dt also logged |

## Mixing 70/10/10/10

| Task | Mean absolute scalar reward | abs p95 | Discounted episode return | Weighted magnitude share |
|---|---:|---:|---:|---:|
| navigation | 0.618955 | 1.67991 | -51.887 | 76.12% |
| stairs | 0.30518 | 0.704893 | -24.7722 | 5.36% |
| gap | 0.755742 | 3.10088 | -54.5297 | 13.28% |
| manipulation | 0.298604 | 0.702133 | -26.3699 | 5.25% |

Scalar density1 includes time, potential shaping and costs; it is not success/progress-event density. Navigation contributes76.12% of this reward-magnitude proxy at70% of transitions: a real domination risk to monitor in validation. This is not a measured gradient share. Sparse success/goal events, lateral safety failures and actual button progress are not represented sufficiently for final tuning; scale1 fallbacks are explicit. None of these preliminary limits is presented as thesis-optimal tuning.

## Physical and lifecycle contracts

- Button: shared legacy/C1 prismatic installer; arbitrary yaw, stroke4mm, actual displacement-projected depression. TRAIN20000 contact-pusher smoke at240Hz measured0.00399988035205207m; no plunger pose assignment. Success remains physical>=3.5mm plus intended MM8 connections; no return-to-spawn.
- Expert owns assembly only. It is stopped after terminal success and independent morphology verification. MM8 approach/press belong to MARL immediately.
- Existing BC mailbox command IDs carry exact pre-state proof; first drive application precedes an exact bounded physics interval; completed stamp binds the observed post-state. No physics advances while policy/learner computes. Goals refresh before their first acknowledged tick.
- True terminal and truncation close action authority until reset. Infrastructure failures stop learning, preventing GAE across an unmarked reset. Whole-body fall below minimum actual support−0.15m terminates failure; partial hanging remains permitted. Invalid/nonfinite/out-of-envelope world states fail closed.
- GAE bootstraps time limits from each captured next-state critic graph/owner mapping; terminal masks suppress bootstrap, terminal|truncation masks stop traces across reset/domain boundaries.
- Shared BC actor in eval mode is warm-started from pinned best.pt; stochastic Bernoulli/Normal RL sampling, initial new logstd−2, no threshold4.5. C1 structural actions masked; original conflicting proposals rejected as whole, never repaired/resampled. Privileged critics receive current full tensors; actor retains its certified1m visibility boundary.

## Verification

Final scoped regression606passed in9.38s (MARL, native BC, physical button contract and C1 launch); runner50passed including actual BC model/real memory/optimizer unit updates, four strict task contexts, native fall failure, partial rollouts and cross-task truncation. Shared Expert TERM-first cleanup17passed. Prior button/legacy regression152passed; current causal/native tests and launch checks passed. All changes remain uncommitted.

## APPO physical update

**GREEN: one actual fixed-alpha Altruistic PPO update after256 physical transitions**,64 per domain.73 native nonempty proposals accepted: navigation0,stairs14,gap50,manipulation9. Navigation actuation was independently demonstrated by194 accepted proposals in calibration; its64 smoke proposals were rejected conflicts and retained as real policy outcomes. No task success claimed; all four smoke episodes end at their64step physical horizon and reset/reassemble for the next domain.

BC-v4 hash verified before load and unchanged after smoke:
`d90a73e9b5ba2ecb579e4fc8733066b1c55d9e60dafeb1d71c029c49a7c21c6d`.

Fixedalpha0.5; radius0.5m; gamma0.99; GAE lambda0.95. Micro-smoke transition mixture25/25/25/25; futureC1-A70/10/10/10. Actor lr1e-5; critics lr3e-4; one epoch/minibatch; stochastic warm-start with new logstd−2. No threshold4.5 in RL.

| Model | Loss | Gradient norm | Optimizer steps | Parameters changed |
|---|---:|---:|---:|---|
| actor | 10.0534983 | 26.7991451 | 1 | yes |
| owner critic | 247.700012 | 318.656662 | 1 | yes |
| mean-field critic | 219.843079 | 337.22988 | 1 | yes |

Entropy-0.851180077 (differential Gaussian entropy can be negative); ratio1.0, approximateKL0.0, clipfraction0.0. Ratio/KL are reevaluated before the first optimizer change. Global critic optimizer steps0 as expected on the owner/MF APPO branch.

Finite observations, logits, log probabilities, values and scalar rewards were checked on the live path. The saved46MiB checkpoint was loaded independently:2641 tensors finite,585 Adam step tensors all exactly1; all raw/scalar/proposal JSONL numbers finite. Command IDs, exact8tick durations and pre/post carry continuity verified for every recorded transition. No NaN or Inf found.

Artifacts:
- `logs/appo_smoke/APPO_SMOKE_fixed_alpha_v1/APPO_SMOKE.pt`
- `logs/appo_smoke/APPO_SMOKE_fixed_alpha_v1/APPO_SMOKE.provenance.json`
- `logs/appo_smoke/APPO_SMOKE_fixed_alpha_v1/updates.jsonl`
- `logs/appo_smoke/APPO_SMOKE_fixed_alpha_v1/validation_audit.json`
- `logs/appo_smoke/APPO_SMOKE_fixed_alpha_v1/verification_tests_final.log`

Checkpoint SHA256:`9f9831d883ae5a1e8309a2787d2a0f468c7fb17b44cecf8e2b8cd5286fbddbd1`. RewardConfig SHA256:`d9c74751a479630421f80600b37409944db3c393d2eef8963d0624914c7c2020`.

Owned runtime/Expert processes were reaped; final scoped process check found no survivors; ROS daemon stop reported not running. No commit/push; HEAD remains32ee2d4b79b07f54d93ebdc850caf17093d034b9. No longC1-A training was started.

## First C1-A commands — prepared, never executed

Full commands are in `logs/appo_smoke/first_c1_a_commands.sh` (bash syntax checked only). TRAIN learning and held-out VALIDATION evaluation are separate commands. Future training uses the full frozen1000seed allocation per domain, reproducibly shuffled with seed0 and independent per-domain cursors; calibration/smoke pools remain unchanged. Completed outcome logging supports VALIDATION success/failure/truncation summaries. The command parser and Bash syntax were checked; neither command was executed.

```bash
# Prepared commands only. Not executed by the agent.
cd /home/lorenzo/MSSR_thesis
source mssr_ws/install/setup.bash
export PYTHONPATH="$PWD:$PWD/scripts/smores_ep/src:$PWD/mssr_ws/src/mssr_expert${PYTHONPATH:+:$PYTHONPATH}"
MSSR_PYTHON=/home/lorenzo/isaaclab-env-beta2/bin/python
MSSR_REWARD=logs/appo_smoke/calibration_validated_train/reward_config.initial.json
MSSR_BC=artifacts/bc_v1/runs/20261022T232532_854802073Z_207309_bc_v4_actor_visible_r1m_scratch_final/best.pt

# First C1-A learning run: full TRAIN seed pools, 70/10/10/10 by transitions.
"$MSSR_PYTHON" -m drl.MARL.c1_runner train \
  --output logs/c1_a_first --checkpoint "$MSSR_BC" --reward-config "$MSSR_REWARD" \
  --domains navigation stairs gap manipulation --split train \
  --total-transitions 100000 --steps-per-layout 256 --rollout-transitions 256 \
  --step-ticks 8 --alpha 0.5 --gamma 0.99 --fail-fast

# Separate held-out VALIDATION evaluation after the learning run.
"$MSSR_PYTHON" -m drl.MARL.c1_runner evaluate \
  --output logs/c1_a_first_validation --reward-config "$MSSR_REWARD" \
  --appo-checkpoint logs/c1_a_first/C1_A_latest.pt \
  --domains navigation stairs gap manipulation --split validation \
  --steps-per-layout 256 --step-ticks 8 --alpha 0.5 --gamma 0.99 --fail-fast

```

READY FOR FIRST C1-A TRAINING: YES
