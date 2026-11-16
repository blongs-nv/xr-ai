# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The JSONL result record from ``evals/EVAL_SCHEMA.md``."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class SchemaError(ValueError):
    """A result line is not valid against the schema."""


@dataclass
class Record:
    """One case run. ``passed`` is serialized as ``pass`` (a Python keyword)."""

    suite: str
    case: str
    role: str
    preset: str
    profile: str
    passed: bool
    score: float
    latency_ms: float
    tokens: dict[str, int | None]
    gpu_profile: str | None
    commit: str
    run_id: str | None = None
    timestamp: str | None = None
    error: str | None = None
    detail: dict[str, Any] | None = None
    gpu_memory_mb: float | None = None

    def to_json(self) -> dict[str, Any]:
        """Return the JSON object, omitting unset optional fields."""
        out: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "suite": self.suite,
            "case": self.case,
            "role": self.role,
            "preset": self.preset,
            "profile": self.profile,
            "pass": self.passed,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "tokens": {"input": self.tokens.get("input"), "output": self.tokens.get("output")},
            "gpu_profile": self.gpu_profile,
            "commit": self.commit,
        }
        for key in ("run_id", "timestamp", "error", "detail", "gpu_memory_mb"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def from_json(cls, obj: Any) -> Record:
        """Parse and validate one object. Unknown fields are ignored."""
        if not isinstance(obj, dict):
            raise SchemaError("a result line must be a JSON object")
        required = {
            "suite": str,
            "case": str,
            "role": str,
            "preset": str,
            "profile": str,
            "pass": bool,
            "commit": str,
        }
        for key, kind in required.items():
            if not isinstance(obj.get(key), kind):
                raise SchemaError(f"field {key!r} is required and must be {kind.__name__}")
        score = obj.get("score")
        if isinstance(score, bool) or not isinstance(score, int | float) or not 0.0 <= score <= 1.0:
            raise SchemaError("field 'score' must be a number from 0.0 to 1.0")
        latency = obj.get("latency_ms")
        if isinstance(latency, bool) or not isinstance(latency, int | float):
            raise SchemaError("field 'latency_ms' must be a number")
        tokens = obj.get("tokens")
        if not isinstance(tokens, dict):
            raise SchemaError("field 'tokens' must be an object")
        gpu_profile = obj.get("gpu_profile")
        if "gpu_profile" not in obj or not (gpu_profile is None or isinstance(gpu_profile, str)):
            raise SchemaError("field 'gpu_profile' is required and must be a string or null")
        return cls(
            suite=obj["suite"],
            case=obj["case"],
            role=obj["role"],
            preset=obj["preset"],
            profile=obj["profile"],
            passed=obj["pass"],
            score=float(score),
            latency_ms=float(latency),
            tokens={"input": tokens.get("input"), "output": tokens.get("output")},
            gpu_profile=gpu_profile,
            commit=obj["commit"],
            run_id=obj.get("run_id"),
            timestamp=obj.get("timestamp"),
            error=obj.get("error"),
            detail=obj.get("detail"),
            gpu_memory_mb=obj.get("gpu_memory_mb"),
        )


def read_records(path: Path) -> Iterator[Record]:
    """Yield every record in a JSONL file; a bad line raises ``SchemaError`` naming the line."""
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield Record.from_json(json.loads(line))
            except (json.JSONDecodeError, SchemaError) as exc:
                raise SchemaError(f"{path}:{number}: {exc}") from exc


def append_records(path: Path, records: Iterable[Record]) -> None:
    """Append records to a JSONL file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_json(), sort_keys=False) + "\n")
