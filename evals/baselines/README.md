<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Eval baselines

Recorded results of every existing eval harness on today's (original) samples, so each sample migration can
show parity.

> **Status: PENDING. No baseline has been recorded.** The eval part needs a free GPU host. When this procedure was
> written the GPUs were occupied by model servers from another project, and starting a model stack on top of them was
> ruled out. No `*.jsonl` file exists here. Everything below is the exact procedure and the layout to commit; none of
> it has been run against real models. The CPU part (API snapshot, lifecycle characterization tests) is complete and
> independent of this.

## Expected layout

```text
evals/baselines/
  README.md
  <suite>.<profile>.jsonl         one record per case, schema in evals/EVAL_SCHEMA.md (xr-ai-eval output)
  <suite>.<profile>.meta.json     provenance of that file (below)
  native/<suite>.<profile>.txt    only when the runner is unavailable: the harness's own output, verbatim
```

`<profile>` is `local` for the committed baselines. One file per suite, never merged, so a migration can diff
exactly the suite it touched.

`<suite>.<profile>.meta.json`:

```json
{
  "suite": "tea-making",
  "profile": "local",
  "gpu_profile": "dual_48G_ada",
  "gpus": ["NVIDIA RTX 6000 Ada Generation x2"],
  "driver": "<nvidia-smi driver version>",
  "commit": "<git rev-parse --short HEAD of the code that ran>",
  "date": "<UTC ISO 8601>",
  "models": {"vlm": "<model id served>", "llm": "<model id served>", "stt": "<...>", "tts": "<...>"},
  "profile_file": "<path to the models file used, and its sha256>",
  "command": "<the exact command line>",
  "honored_profile": true
}
```

`gpu_profile`, `commit` and `timestamp` are also in every JSONL record, and `preset` carries the served model;
the meta file adds the hardware, the profile file and the command. `honored_profile` is false for harnesses that
read their own `yaml/models.json` and ignore `--profile` (the `profile_honored` field in each record's `detail`).

## Suites

Original harnesses, as adapted for the runner (`evals/adapters`, experimental; see `evals/README.md`):

| Suite | Harness it runs | Model config it really uses |
| --- | --- | --- |
| `tea-making` | `agent-samples/tea-making-sample/eval/eval.py` | the sample's `yaml/models.local.json` |
| `lab-instrument-monitoring` | `eval/eval.py` (tools) and `eval/visual_eval.py` (visual) | `yaml/models.json` |
| `conversation-example` | `eval/run_decision_eval.py` | `yaml/models.json` |
| `conversation-context` | `agent-sdk/xr-ai-conversation/eval/conflict_eval.py`, `compaction_eval.py` | the profile (`--models`) |
| `xr-render-demo` | `xr_render_demo_eval.harness` | its own fakes plus the model it is pointed at |
| `procedure-capture-guide` | `evaluate_visual_checks.py` (`--models`) and `evaluate_captions.py` | profile for visual checks, `yaml/models.json` for captions |

`simple-vlm-example` has no eval harness. The `tea-visual` and `speech` suites (under `evals/suites/`) are
new content, not baselines of existing behavior; record them with the same commands when their live runs are done.

## Regenerate

### 1. Preconditions

```bash
nvidia-smi                      # both GPUs must show no processes you do not own
git status --short              # clean tree; baselines record the commit
git rev-parse --short HEAD      # this goes in the meta files
```

Record baselines only on a checkout whose `agent-samples/` code equals `main` (no migrated sample), because the
point is to describe the original behavior.

### 2. Start the local model stack

The `local` profile starts every model it needs. For the tea sample, which has the three standard profiles:

```bash
uv --config-file uv.toml run --project utils/xr-ai-model-server-stack xr-ai run tea-making-sample --profile local
# or, only the servers, from the sample folder:
uv --config-file uv.toml run --project agent-samples/tea-making-sample/model-servers tea_making_sample_model_servers
```

The other original samples have no profile set yet: start their servers from their own `model-servers/` project
(`uv --config-file uv.toml run --project agent-samples/<sample>/model-servers <entry point>`, see each sample's README) and
point the harnesses at the same endpoints. Wait for the servers' health endpoints before running anything.

### 3. Run each suite through the runner

From the repository root; `--profile` is a path or a name found as `models.<name>.json`:

```bash
P=deployment/profiles/tea-making-sample/models.local.json     # the local profile
mkdir -p evals/baselines
for s in tea-making lab-instrument-monitoring conversation-example conversation-context \
         xr-render-demo procedure-capture-guide; do
  uv --config-file uv.toml run --project tests xr-ai-eval run \
      --suite "$s" --profile "$P" --out "evals/baselines/$s.local.jsonl"
done
```

`xr-ai-eval run` appends to `--out`; delete a stale file first. Exit code 1 only means some cases failed, which
is expected for a baseline of imperfect behavior: keep the file. Exit code 2 (invalid case, profile or result
file) means no valid baseline.

Then write `<suite>.local.meta.json` for each file from the template above (`nvidia-smi`,
`git rev-parse`, the profile file's `sha256sum`).

### 4. Fallback when the runner cannot be used

Store each harness's native output unchanged, under `evals/baselines/native/`, with the same meta file:

```bash
uv --config-file uv.toml run --project agent-samples/tea-making-sample/worker \
    python agent-samples/tea-making-sample/eval/eval.py | tee evals/baselines/native/tea-making.local.txt
```

The exact argv for every harness is what the adapters build: `HarnessAdapter.command()` in
`evals/adapters/xr_ai_eval_adapters/base.py`, with the scripts and projects listed in `harnesses.py`.

### 5. Check before committing

```bash
python3 - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("evals/baselines").glob("*.local.jsonl")):
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert rows and all(r["profile"] and r["commit"] and r["gpu_profile"] for r in rows), p
    print(p.name, len(rows), "cases", sum(r["pass"] for r in rows), "passed")
PY
```

Every suite needs a `.jsonl` and a `.meta.json`.

## Expected results

There are no recorded numbers to expect yet. When baselines exist, a migrated sample passes parity when, per
suite, on the same profile and GPU profile:

- every case that passed in the baseline still passes (a new pass is fine);
- `error` stays empty for cases that had none;
- median latency per role stays within the tolerance the migration PR states (record it there; baselines are
  from one host and are indicative, not a benchmark).

Suites that only print lines (`tea-making`, `lab-instrument-monitoring`, `xr-render-demo`, the conversation-context harnesses) are scored by the adapters'
parsers, which are tested on fixtures, not recordings (`evals/README.md`). Treat their per-case scores as indicative until a first real
run confirms each parser against live output, and fix the parser rather than the baseline if they disagree.

## Checks that ran without a GPU

Only the runner path was exercised, against a stub OpenAI-compatible server (no model stack):

```bash
# serve a stub (tests/app/eval/eval_support.StubServer) and write models.stub.json pointing at it, then
uv --config-file uv.toml run --project tests xr-ai-eval run --suite conversation-context \
    --profile <scratch>/models.stub.json --out <scratch>/out.jsonl
```

Result: 10 records written with the profile, commit and latency fields filled and every case failing, as expected for a canned
answer. This is not a baseline and was written outside this directory. The runner tests are in `tests/app/eval/`.
