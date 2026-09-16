from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetStream:
    stream_id: str
    stage_id: int
    task_id: str
    phase: str
    producer: str
    path: Path
    action_space: str
    source_morphology: str | None
    target_morphology: str | None
    intended_for_behavior_cloning: bool


class CompositeDatasetManifest:
    def __init__(
        self,
        runtime_dir: Path,
        episode_id: str,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.parts_dir = self.runtime_dir / "dataset_parts"
        self.parts_dir.mkdir(parents=True, exist_ok=True)

        self.path = self.runtime_dir / "dataset_manifest.json"
        self._payload: dict[str, Any] = {
            "schema_version": "mssr.composite_dataset_manifest.v1",
            "episode_id": str(episode_id),
            "status": "running",
            "streams": [],
        }
        self._write()

    @staticmethod
    def _slug(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in value
        )

    def _write(self) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                self._payload,
                stream,
                indent=2,
                sort_keys=False,
            )
            stream.write("\n")
            stream.flush()
        temporary.replace(self.path)

    @staticmethod
    def _measure(path: Path) -> tuple[int, int]:
        records = 0
        byte_count = 0

        if not path.is_file():
            return records, byte_count

        with path.open("rb") as stream:
            for line in stream:
                byte_count += len(line)
                if line.strip():
                    records += 1

        return records, byte_count

    def register(
        self,
        *,
        stage_id: int,
        task_id: str,
        phase: str,
        producer: str,
        action_space: str,
        source_morphology: str | None,
        target_morphology: str | None,
        intended_for_behavior_cloning: bool,
        filename: str | None = None,
    ) -> DatasetStream:
        stream_id = (
            f"{int(stage_id):02d}-"
            f"{self._slug(task_id)}-"
            f"{self._slug(phase)}"
        )

        if any(
            item["stream_id"] == stream_id
            for item in self._payload["streams"]
        ):
            raise ValueError(
                f"Duplicate composite dataset stream {stream_id!r}"
            )

        if filename is None:
            filename = stream_id + ".jsonl"

        path = self.parts_dir / filename

        if any(
            item["path"] == path.relative_to(self.runtime_dir).as_posix()
            for item in self._payload["streams"]
        ):
            raise ValueError(
                f"Duplicate composite dataset path {path}"
            )

        result = DatasetStream(
            stream_id=stream_id,
            stage_id=int(stage_id),
            task_id=str(task_id),
            phase=str(phase),
            producer=str(producer),
            path=path,
            action_space=str(action_space),
            source_morphology=source_morphology,
            target_morphology=target_morphology,
            intended_for_behavior_cloning=bool(
                intended_for_behavior_cloning
            ),
        )

        self._payload["streams"].append(
            {
                "stream_id": result.stream_id,
                "stage_id": result.stage_id,
                "task_id": result.task_id,
                "phase": result.phase,
                "producer": result.producer,
                "path": result.path.relative_to(
                    self.runtime_dir
                ).as_posix(),
                "action_space": result.action_space,
                "source_morphology": result.source_morphology,
                "target_morphology": result.target_morphology,
                "intended_for_behavior_cloning": (
                    result.intended_for_behavior_cloning
                ),
                "status": "registered",
                "records": 0,
                "bytes": 0,
            }
        )
        self._write()
        return result

    def finalize_stage(
        self,
        stage_id: int,
        success: bool,
    ) -> None:
        for item in self._payload["streams"]:
            if int(item["stage_id"]) != int(stage_id):
                continue

            path = self.runtime_dir / item["path"]
            records, byte_count = self._measure(path)

            item["records"] = records
            item["bytes"] = byte_count
            item["status"] = (
                "completed" if success else "partial"
            )

        self._write()

    def finalize_episode(
        self,
        success: bool,
    ) -> None:
        # Re-measure every stream at episode close.  A behavior stream
        # may receive its final transition after finalize_stage(), when
        # the next dataset boundary flushes a pending (s_t, a_t, s_t+1).
        for item in self._payload["streams"]:
            path = self.runtime_dir / item["path"]
            records, byte_count = self._measure(path)

            item["records"] = records
            item["bytes"] = byte_count

            if item["status"] == "registered":
                item["status"] = (
                    "partial"
                    if records > 0
                    else "not_started"
                )

        self._payload["status"] = (
            "succeeded" if success else "failed"
        )
        self._write()
