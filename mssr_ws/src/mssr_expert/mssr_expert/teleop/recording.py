"""Non-blocking recording backend for human teleoperation demonstrations."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Mapping
import math

from mssr_expert.dataset.dataset_logger import DatasetLogger
from mssr_expert.experts.expert_output import ExpertOutput
from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph
from mssr_expert.utils.json_io import dumps_json


_STOP = object()


class RecordingManager:
    """Own one teleoperation recording episode and its manifest."""

    def __init__(
        self,
        *,
        root: Path,
        git_commit: str,
        dataset_rate_hz: float,
        queue_capacity: int,
    ) -> None:
        if dataset_rate_hz <= 0.0:
            raise ValueError("dataset_rate_hz must be positive")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")

        self.root = Path(root)
        self.git_commit = str(git_commit)
        self.dataset_rate_hz = float(dataset_rate_hz)
        self.queue_capacity = int(queue_capacity)

        self.recording = False
        self.episode_dir: Path | None = None
        self.human_path: Path | None = None
        self.manifest_path: Path | None = None

        self._queue: queue.Queue[Any] | None = None
        self._thread: threading.Thread | None = None
        self._finalizer_thread: threading.Thread | None = None
        self._writer_error: str | None = None
        self._unwritten_records = 0
        self._state_lock = threading.RLock()
        self._manifest: dict[str, Any] = {}
        self._structural_streams: list[dict[str, str]] = []

    def start(
        self,
        *,
        episode_id: str,
        started_at: float,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if self.recording:
            raise RuntimeError("recording already active")

        self.root.mkdir(parents=True, exist_ok=True)

        episode_dir = self.root / str(episode_id)
        episode_dir.mkdir(parents=False, exist_ok=False)

        self.episode_dir = episode_dir
        self.human_path = episode_dir / "human_behavior.jsonl"
        self.manifest_path = episode_dir / "manifest.json"

        # Create the authoritative human stream even if it stays empty.
        self.human_path.touch()

        self._queue = queue.Queue(maxsize=self.queue_capacity)
        self._writer_error = None
        self._unwritten_records = 0
        self._structural_streams = []

        self._manifest = {
            "schema_version": "mssr.teleop_recording_manifest.v1",
            "episode_id": str(episode_id),
            "git_commit": self.git_commit,
            "status": "running",
            "started_at": float(started_at),
            "ended_at": None,
            "task_success": None,
            "dataset_rate_hz_configured": self.dataset_rate_hz,
            "dataset_rate_hz_measured": None,
            "config": dict(config or {}),
            "human_stream": {
                "path": str(self.human_path),
                "producer": "human_expert",
                "records": 0,
                "bytes": 0,
            },
            "structural_streams": self._structural_streams,
            "events": [],
            "error": None,
            "eligible_for_import": False,
            "unwritten_records": 0,
        }

        self._write_manifest()
        self.recording = True

        self._thread = threading.Thread(
            target=self._writer_loop,
            name=f"teleop-recorder-{episode_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def failed(self) -> bool:
        with self._state_lock:
            return self._writer_error is not None

    @property
    def error(self) -> str | None:
        with self._state_lock:
            return self._writer_error

    def _latch_error(self, reason: str, *, unwritten: int = 0) -> None:
        with self._state_lock:
            if self._writer_error is None:
                self._writer_error = str(reason)
            self._unwritten_records += int(unwritten)
            self._manifest["status"] = "failed"
            self._manifest["error"] = self._writer_error
            self._manifest["eligible_for_import"] = False
            self._manifest["unwritten_records"] = self._unwritten_records

    def enqueue_record(self, record: Mapping[str, Any]) -> None:
        if not self.recording or self._queue is None:
            raise RuntimeError("recording is not active")

        with self._state_lock:
            if self._writer_error is not None:
                raise RuntimeError(self._writer_error)

            try:
                self._queue.put_nowait(dict(record))
            except queue.Full as error:
                self._writer_error = "writer_queue_saturated"
                self._unwritten_records += 1
                self._manifest["status"] = "failed"
                self._manifest["error"] = self._writer_error
                self._manifest["eligible_for_import"] = False
                self._manifest["unwritten_records"] = self._unwritten_records
                raise RuntimeError(self._writer_error) from error

    def record_event(self, event: Mapping[str, Any]) -> None:
        """Retain one important session/control event in the manifest."""
        if not self.recording:
            return
        self._manifest["events"].append(dict(event))
        self._write_manifest()

    def register_structural_stream(
        self,
        *,
        stream_id: str,
        phase: str,
        path: Path,
        producer: str,
    ) -> None:
        if not self.recording:
            raise RuntimeError("recording is not active")

        self._structural_streams.append(
            {
                "stream_id": str(stream_id),
                "phase": str(phase),
                "producer": str(producer),
                "path": str(path),
            }
        )
        self._write_manifest()

    @property
    def finalizing(self) -> bool:
        thread = self._finalizer_thread
        return thread is not None and thread.is_alive()

    def request_stop(
        self,
        *,
        ended_at: float,
        task_success: bool | None,
    ) -> None:
        """Finalize on a worker so the 50 Hz control loop never waits."""

        if not self.recording:
            raise RuntimeError("recording is not active")
        if self.finalizing:
            raise RuntimeError("recording finalization already active")

        self._finalizer_thread = threading.Thread(
            target=self.stop,
            kwargs={
                "ended_at": float(ended_at),
                "task_success": task_success,
            },
            name="teleop-recording-finalizer",
            daemon=True,
        )
        self._finalizer_thread.start()

    def wait_finalized(self, timeout: float | None = None) -> None:
        thread = self._finalizer_thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError("recording finalization timed out")

    def stop(
        self,
        *,
        ended_at: float,
        task_success: bool | None,
    ) -> None:
        if not self.recording:
            raise RuntimeError("recording is not active")

        assert self._queue is not None
        assert self._thread is not None
        assert self.human_path is not None

        # Blocking here is intentional: STOP is finalization, not the 50 Hz loop.
        self._queue.put(_STOP)
        self._queue.join()
        self._thread.join()

        records = sum(
            1
            for line in self.human_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
        byte_count = self.human_path.stat().st_size

        started_at = float(self._manifest["started_at"])
        duration = max(0.0, float(ended_at) - started_at)
        measured_rate = (
            records / duration
            if duration > 0.0
            else None
        )

        self.recording = False
        self._manifest["ended_at"] = float(ended_at)
        self._manifest["task_success"] = task_success
        self._manifest["status"] = (
            "failed"
            if self.failed
            else "completed"
        )
        self._manifest["error"] = self.error
        self._manifest["eligible_for_import"] = not self.failed
        self._manifest["unwritten_records"] = self._unwritten_records
        self._manifest["dataset_rate_hz_measured"] = measured_rate
        self._manifest["human_stream"]["records"] = records
        self._manifest["human_stream"]["bytes"] = byte_count

        self._write_manifest()

    def _writer_loop(self) -> None:
        assert self._queue is not None
        assert self.human_path is not None

        with self.human_path.open("a", encoding="utf-8") as stream:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return

                    if self.failed:
                        with self._state_lock:
                            self._unwritten_records += 1
                        continue

                    stream.write(dumps_json(item) + "\n")
                    stream.flush()
                except Exception as error:
                    self._latch_error(
                        f"writer_error:{type(error).__name__}:{error}",
                        unwritten=1,
                    )
                finally:
                    self._queue.task_done()

    def _write_manifest(self) -> None:
        assert self.manifest_path is not None

        temporary = self.manifest_path.with_suffix(
            self.manifest_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(
                self._manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)


class TransitionSampler:
    """Sample human state/action transitions independently of control rate."""

    def __init__(
        self,
        *,
        dataset_rate_hz: float,
        episode_id: str = "teleop",
    ) -> None:
        rate = float(dataset_rate_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("dataset_rate_hz must be finite and positive")

        self.dataset_rate_hz = rate
        self.episode_id = str(episode_id)
        self._period_s = 1.0 / rate
        self._next_sample_at: float | None = None
        self._pending: dict[str, Any] | None = None
        self._timestep = 0

        # build_record() performs no file write.  This object only supplies
        # the canonical mssr.expert_transition.v3 construction logic.
        self._record_builder = DatasetLogger(
            Path("human_behavior.jsonl"),
            label_source="human_expert",
            executed_action_source="human_expert",
        )

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def step(
        self,
        *,
        now: float,
        graph: AttributedRobotGraph,
        controller_input: Mapping[str, Any],
        intent: Mapping[str, Any],
        effective_actions: Mapping[str, Mapping[str, Any]],
        morphology: str,
    ) -> dict[str, Any] | None:
        """Advance synchronization and return at most one complete transition."""

        now = float(now)
        produced: dict[str, Any] | None = None

        # A pending action belongs to graph_t.  It becomes a valid transition
        # only after observing a strictly newer physical graph.
        if (
            self._pending is not None
            and float(graph.stamp)
            > float(self._pending["graph"].stamp)
        ):
            previous = self._pending

            observation = {
                "schema_version": "mssr.teleop_observation.v1",
                "controller_input": dict(previous["controller_input"]),
                "intent": dict(previous["intent"]),
                "morphology": str(previous["morphology"]),
                "sampled_at_wall": float(previous["sampled_at_wall"]),
            }

            next_observation = {
                "schema_version": "mssr.teleop_observation.v1",
                "morphology": str(morphology),
            }

            output = ExpertOutput(
                locomotion={
                    str(module_id): dict(command)
                    for module_id, command
                    in previous["effective_actions"].items()
                },
                fsm_state="TELEOP",
                active_primitive="human_teleop",
                debug={
                    "action_source": "effective_module_actions",
                },
            )

            produced = self._record_builder.build_record(
                episode_id=self.episode_id,
                timestep=self._timestep,
                observation=observation,
                graph=previous["graph"],
                expert_output=output,
                stage_name="human_behavior",
                stage_id=self._timestep,
                task_type=f"{previous['morphology']}_teleop",
                difficulty=0.0,
                next_graph=graph,
                next_observation=next_observation,
            )

            self._timestep += 1
            self._pending = None

        # Sampling cadence is wall-clock based and independent from the
        # control loop.  Never replace an unfinished pending transition.
        if self._pending is None and self._sample_due(now):
            self._pending = {
                "graph": graph,
                "controller_input": dict(controller_input),
                "intent": dict(intent),
                "effective_actions": {
                    str(module_id): dict(command)
                    for module_id, command
                    in effective_actions.items()
                },
                "morphology": str(morphology),
                "sampled_at_wall": now,
            }
            self._advance_sample_deadline(now)

        return produced

    def close(self) -> int:
        """Discard an incomplete final transition instead of inventing s_(t+1)."""

        dropped = 1 if self._pending is not None else 0
        self._pending = None
        return dropped

    def _sample_due(self, now: float) -> bool:
        return (
            self._next_sample_at is None
            or now + 1.0e-12 >= self._next_sample_at
        )

    def _advance_sample_deadline(self, now: float) -> None:
        if self._next_sample_at is None:
            self._next_sample_at = now + self._period_s
            return

        # Keep the configured cadence rather than drifting by
        # "current time + period" after a delayed control tick.
        while self._next_sample_at <= now + 1.0e-12:
            self._next_sample_at += self._period_s


class TeleopRecordingController:
    """Coordinate recording lifecycle without owning robot authority."""

    def __init__(
        self,
        *,
        root: Path,
        git_commit: str,
        dataset_rate_hz: float,
        queue_capacity: int = 1024,
        episode_id_factory=None,
    ) -> None:
        from uuid import uuid4

        self.root = Path(root)
        self.git_commit = str(git_commit)
        self.dataset_rate_hz = float(dataset_rate_hz)
        self.queue_capacity = int(queue_capacity)
        self._episode_id_factory = (
            episode_id_factory
            if episode_id_factory is not None
            else lambda: "teleop-" + uuid4().hex
        )

        self.manager: RecordingManager | None = None
        self.sampler: TransitionSampler | None = None

    @property
    def backend_ready(self) -> bool:
        if self.manager is None:
            return True
        return not self.manager.failed

    @property
    def error(self) -> str | None:
        return self.manager.error if self.manager is not None else None

    @property
    def episode_id(self) -> str | None:
        if self.manager is None or self.manager.episode_dir is None:
            return None
        return self.manager.episode_dir.name

    def update(
        self,
        *,
        recording_requested: bool,
        authority: str,
        graph: AttributedRobotGraph | None,
        controller_input: Mapping[str, Any],
        intent: Mapping[str, Any],
        effective_actions: Mapping[str, Mapping[str, Any]],
        morphology: str | None,
        now: float,
        wall_time: float,
        events: tuple[Mapping[str, Any], ...] = (),
    ) -> None:
        if recording_requested:
            self._ensure_started(wall_time)

            manager = self.manager
            sampler = self.sampler

            if manager is not None:
                for event in events:
                    manager.record_event(event)

            if (
                manager is None
                or sampler is None
                or manager.failed
                or manager.finalizing
            ):
                return

            # Never let a human-labelled transition span E-STOP,
            # structural-macro authority, disconnect/no-authority, or
            # another interval in which TELEOP does not own the robot.
            if authority != "TELEOP":
                sampler.close()
                return

            if graph is None or morphology is None:
                sampler.close()
                return

            record = sampler.step(
                now=now,
                graph=graph,
                controller_input=controller_input,
                intent=intent,
                effective_actions=effective_actions,
                morphology=morphology,
            )

            if record is not None:
                try:
                    manager.enqueue_record(record)
                except RuntimeError:
                    # Recording failure must never remove robot control.
                    pass
            return

        manager = self.manager

        if manager is not None and manager.recording:
            for event in events:
                manager.record_event(event)

        if (
            manager is not None
            and manager.recording
            and not manager.finalizing
        ):
            if self.sampler is not None:
                self.sampler.close()

            manager.request_stop(
                ended_at=wall_time,
                task_success=None,
            )

    def wait_finalized(self, timeout: float | None = None) -> None:
        if self.manager is not None:
            self.manager.wait_finalized(timeout)

    def register_structural_stream(
        self,
        *,
        stream_id: str,
        phase: str,
        path: Path,
        producer: str = "deterministic_expert",
    ) -> None:
        if self.manager is None or not self.manager.recording:
            raise RuntimeError("recording is not active")

        self.manager.register_structural_stream(
            stream_id=stream_id,
            phase=phase,
            path=path,
            producer=producer,
        )

    def _ensure_started(self, wall_time: float) -> None:
        if self.manager is not None:
            if self.manager.recording or self.manager.finalizing:
                return

        episode_id = str(self._episode_id_factory())

        manager = RecordingManager(
            root=self.root,
            git_commit=self.git_commit,
            dataset_rate_hz=self.dataset_rate_hz,
            queue_capacity=self.queue_capacity,
        )
        manager.start(
            episode_id=episode_id,
            started_at=wall_time,
            config={
                "dataset_rate_hz": self.dataset_rate_hz,
            },
        )

        self.manager = manager
        self.sampler = TransitionSampler(
            dataset_rate_hz=self.dataset_rate_hz,
            episode_id=episode_id,
        )
