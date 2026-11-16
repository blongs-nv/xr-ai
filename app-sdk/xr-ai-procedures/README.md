<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-procedures

A kit for procedure (SOP) apps, on top of the `xr-ai-app` SDK (import name
`xr_ai_procedures`). It holds what is about procedures, including how a step
watches the camera over time (`xr_ai_procedures.perception`); the SDK gives the
camera stream (`p.frames`) and the perception role. It uses only the SDK's
public names, and the kit's (`xr-ai-kit`): a guided procedure's `control` is a
router destination. It is being replaced by the kit's `xr_ai_kit.sop`; new apps
should use that.

| Part | Main names | What it does |
|---|---|---|
| Capture | `ProcedureCapture` | Records a narrated demonstration as an evidence packet: frames, captions (with the activity and phase they show), narration, and a summary. |
| Guide | `GuidedProcedure`, `Sop`, `StepSpec`, `StepDef`, `Run`, `Facts`, `Timer` | Walks someone through an approved procedure step by step. A step is any `Runnable`: a camera check (`read_check`), a read-and-look-up (`VisionRetrieval`), a `Timer`, or your own. |
| Watching the camera | `xr_ai_procedures.perception`: `Done`/`Needs`/`Stuck`, `Eyes`, `Watching`, `Confirmation`, `Verify`, `read_check`, the plugs, `VisionRetrieval` | What a camera step is made of: fresh frames at a steady pace, a structured read judged in code, confirmation over several looks, escalation, and giving up. |

The capture packet uses the format of `agent-samples/procedure-capture-guide`,
whose `guide-generation-skills` turn a packet into a guide.

Samples: `app-samples/sdk-tour/12-procedure-capture` (capture) and
`app-samples/sdk-tour/10-guided-procedure` (guide). Tests:
`tests/app/surface/test_procedures.py` and `test_capture.py`.
