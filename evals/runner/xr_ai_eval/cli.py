# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``xr-ai-eval run | matrix | report``.

Exit codes follow ``evals/EVAL_SCHEMA.md``: 0 when every executed case passed, 1 when any failed,
2 for invalid cases, profiles, result files or arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys
import tempfile
import uuid
from pathlib import Path

from xr_ai_eval.adapters import AdapterContext, adapter_names, find_adapter
from xr_ai_eval.cases import CaseError, load_suite
from xr_ai_eval.client import ChatClient
from xr_ai_eval.metrics import GpuMemoryReader, git_commit, nvidia_smi_memory_mb
from xr_ai_eval.profiles import Profile, ProfileError, load_profile
from xr_ai_eval.report import build_scorecard
from xr_ai_eval.runner import run_suite, utc_now
from xr_ai_eval.schema import Record, SchemaError, append_records, read_records

EXIT_OK, EXIT_FAILED, EXIT_INVALID = 0, 1, 2


def default_evals_dir() -> Path:
    """``./evals`` when present, else the ``evals`` directory this package lives in."""
    local = Path.cwd() / "evals"
    return local if local.is_dir() else Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xr-ai-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--evals-dir", type=Path, help="directory holding <suite>/cases (default: ./evals)")
        p.add_argument("--profiles-dir", type=Path, action="append", default=[], help="where to look up named profiles")
        p.add_argument("--no-gpu-memory", action="store_true", help="do not call nvidia-smi")
        p.add_argument("--timeout-s", type=float, default=120.0)

    run = sub.add_parser("run", help="run one suite against one profile")
    common(run)
    run.add_argument("--suite", required=True, help="a suite under evals/, or a registered adapter name")
    run.add_argument("--profile", required=True, help="profile name or path")
    run.add_argument("--out", type=Path, help="JSONL to append to (default evals/results/<date>-<commit>.jsonl)")
    run.add_argument("--tags", default="", help="comma-separated; run cases with any of these tags")
    run.add_argument("--roles", default="", help="comma-separated; run cases of these roles")

    matrix = sub.add_parser("matrix", help="run suites across profiles and presets")
    common(matrix)
    matrix.add_argument("--suite", action="append", required=True, help="repeat for several suites")
    matrix.add_argument("--profiles", required=True, help="comma-separated profile names or paths")
    matrix.add_argument("--presets", default="", help="comma-separated presets replacing the profile's own")
    matrix.add_argument("--out", type=Path, help="JSONL to append to")
    matrix.add_argument("--tags", default="")
    matrix.add_argument("--roles", default="")

    report = sub.add_parser("report", help="write the role-by-model scorecard as Markdown")
    report.add_argument("--results", type=Path, nargs="+", required=True, help="result JSONL files")
    report.add_argument("--out", type=Path, required=True, help="Markdown file to write")
    return parser


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _search_dirs(args: argparse.Namespace, evals_dir: Path) -> list[Path]:
    return [*args.profiles_dir, Path.cwd(), evals_dir / "profiles"]


def _gpu_reader(args: argparse.Namespace) -> GpuMemoryReader | None:
    return None if args.no_gpu_memory else nvidia_smi_memory_mb


async def _run_one(
    args: argparse.Namespace,
    evals_dir: Path,
    suite_name: str,
    profile: Profile,
    commit: str,
    run_id: str,
    gpu_memory: GpuMemoryReader | None,
) -> list[Record]:
    if (evals_dir / suite_name / "cases").is_dir():
        suite = load_suite(evals_dir, suite_name)
        client = ChatClient(timeout_s=args.timeout_s)
        try:
            return await run_suite(
                suite,
                profile,
                client,
                commit=commit,
                run_id=run_id,
                tags=_split(args.tags),
                roles=_split(args.roles),
                gpu_memory=gpu_memory,
            )
        finally:
            await client.close()
    adapter = find_adapter(suite_name)
    if adapter is None:
        known = ", ".join(adapter_names()) or "none installed"
        raise CaseError(f"no suite {suite_name!r} under {evals_dir} and no adapter of that name ({known})")
    ctx = AdapterContext(
        repo_root=evals_dir.parent,
        profile_name=profile.name,
        profile_path=profile.path,
        profile=profile,
        commit=commit,
        run_id=run_id,
        timestamp=utc_now(),
        work_dir=Path(tempfile.mkdtemp(prefix=f"xr-ai-eval-{suite_name}-")),
    )
    return await asyncio.to_thread(adapter.run, ctx)


def _default_out(evals_dir: Path, commit: str) -> Path:
    return evals_dir / "results" / f"{dt.datetime.now(dt.UTC):%Y-%m-%d}-{commit}.jsonl"


def _execute(args: argparse.Namespace, suites: list[str], profile_values: list[str], presets: list[str]) -> int:
    evals_dir = (args.evals_dir or default_evals_dir()).resolve()
    commit = git_commit(evals_dir.parent)
    out = args.out or _default_out(evals_dir, commit)
    run_id = uuid.uuid4().hex[:12]
    gpu_memory = _gpu_reader(args)
    records: list[Record] = []
    for value in profile_values:
        base = load_profile(value, _search_dirs(args, evals_dir))
        variants = [base.with_preset(p) for p in presets] or [base]
        for profile in variants:
            for suite_name in suites:
                records += asyncio.run(_run_one(args, evals_dir, suite_name, profile, commit, run_id, gpu_memory))
    append_records(out, records)
    failed = sum(not r.passed for r in records)
    print(f"{len(records) - failed}/{len(records)} cases passed; wrote {out}")
    if not records:
        print("warning: no cases were selected", file=sys.stderr)
    return EXIT_FAILED if failed else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the exit code (also raised through ``SystemExit`` as a script)."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _execute(args, [args.suite], [args.profile], [])
        if args.command == "matrix":
            return _execute(args, args.suite, _split(args.profiles), _split(args.presets))
        records = [r for path in args.results for r in read_records(path)]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(build_scorecard(records), encoding="utf-8")
        print(f"wrote {args.out} from {len(records)} records")
        return EXIT_OK
    except (CaseError, ProfileError, SchemaError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
