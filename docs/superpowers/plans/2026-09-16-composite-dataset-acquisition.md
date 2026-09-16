# Composite Dataset Acquisition Implementation Plan

Spec: docs/superpowers/specs/2026-09-16-composite-dataset-acquisition-design.md

1. Add stage-owned dataset stream manifest.
2. Make morphology behavior dataset path switchable per stage.
3. Route assembly/reconfiguration/behavior/Nav2 to distinct raw JSONL files.
4. Disable duplicate Nav2 BC writer in composite mode.
5. Split button acquisition into behavior/reconfiguration/manipulation streams.
6. Finalize success/failure manifests without monolithic concatenation.
7. Run full smores_ep and mssr_expert tests.
8. Validate composite-c05 and measure per-stream size/rate.

Deferred: extend build_expert_v1.py to import composite manifests into expert_v1_compact.
