<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Step accuracy on a labeled recording

`step_eval.py` measures each visual check of the tea procedure on frames
of a real recording whose steps are labeled with times. It calls exactly
what runs live: `xr_ai_kit.sop.judge` (and, for identification, the same
label schema), with `steps.yaml`'s wording, `app.yaml`'s tea settings and
`yaml/models.json`'s roles.

## Labels

A label file (`labels/green-tea-04.yaml` is one recording) gives, for each
step, intervals in seconds from the start of the video:

| Key | Meaning |
|---|---|
| `positive` | the step's condition holds: the check should judge positive |
| `negative` | it visibly does not hold: the check should not |
| `exclude` | ambiguous (in transition, flickering); never scored |
| `traps` | frames that look done and are not (a kettle's target setting shown on its display); a trap judged positive is a false completion |
| `watch` | the part of the video the step watches, to replay the live rule |
| `no_data` | the recording does not show this step |

Frames within `margin_s` of a positive or negative boundary are not
scored. The file says where its labels came from; the green-tea labels were
derived from timestamped captions of the video and checked by looking at
the frames at each boundary, so they are good to about a second, not
frame-exact.

## Run

The recording's frames are not in the repository. Extract them at 1 frame
per second (or more) and write an index CSV with the columns
`source_time_s` and `frame_image` (paths relative to the CSV). Then, from
the sample's folder, with the models in `yaml/models.json` running (the vlm
and decision roles):

```bash
uv --config-file ../../uv.toml run python eval/step_eval.py \
    --labels eval/labels/green-tea-04.yaml \
    --frames-index /path/to/frames/index.csv --out runs/eval/
```

`--fakes eval/smoke.fakes.yaml` runs the same plumbing on scripted replies
(no models; the numbers then mean nothing). Frames are judged every
`interval_s` of `app.yaml`'s tea settings (2 s), as the checks look live;
`--concurrency` bounds the model calls in flight.

## Output

`runs/eval/summary.md` (and `summary.json`):

| Column | Meaning |
|---|---|
| Recall | positive frames judged positive |
| False pos. | negative frames judged positive |
| Precision | of the frames judged positive, how many were positive frames |
| Unknown | scored frames the model could not tell (they neither confirm nor reset a live check) |
| Traps judged pos. | trap frames judged positive |
| Live rule | the 2-of-the-last-4 rule replayed over the watch window: `ok` at a time at or after the labeled start of done, `early` (a false completion), or `never` |

For identification, positive means a tea type was read; the summary also
gives how often the type and the name were right. `frames.jsonl` has every
frame's verdict with the model's observation, for reading why.
