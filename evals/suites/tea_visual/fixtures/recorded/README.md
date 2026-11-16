<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Recorded frames (pending)

Put JPEG frames here, named as in `../../cases/recorded.yaml` (`kettle_at_target.jpg`, ...). Extract them from
raw capture bundles with `../../extract_frames.py`. What each frame must show is in the case's description
in `../../cases/recorded.yaml`. Until the files exist the recorded cases are skipped and write no records.
