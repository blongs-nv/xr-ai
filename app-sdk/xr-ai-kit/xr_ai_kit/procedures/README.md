<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Procedures (not here yet)

The procedures kit will fold into `xr_ai_kit` here later: capturing a
demonstrated procedure as an evidence packet, guiding someone through an
approved one, and watching the camera over time (`Verify`, `read_check`,
`Watching` and the outcomes).

Today it is the separate package `xr-ai-procedures` (import name
`xr_ai_procedures`), in `../../../xr-ai-procedures/`. It depends on this kit
for routing (`GuidedProcedure.control` is a router destination). There is no
code in this folder and nothing to import from `xr_ai_kit.procedures`.
